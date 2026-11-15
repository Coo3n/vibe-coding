"""Компактная Rich-консоль: меню, диалог и сравнение стратегий."""

import argparse
from contextlib import contextmanager
from datetime import datetime
import json
from pathlib import Path
import sys

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .agent import Agent
from .client import ChatClient, ClientError, load_settings
from .context import STRATEGY_CLASSES
from .experiment import run_experiment
from .report import load_report


COLORS = {'user': '#92b7ff', 'answer': '#76ddbd', 'tool': '#e7ba76', 'error': '#ff8a96'}
LABELS = {'sliding': 'Sliding Window', 'sticky': 'Sticky Facts', 'summary': 'Summary'}
ROOT = Path(__file__).resolve().parent.parent
HELP = ('/strategy sliding|sticky|summary · /window N\n'
        '/checkpoint имя · /branch имя · /switch имя · /tree\n'
        '/history · /facts · /summary · /reset · /help · /exit')


def number(value):
    return '?' if value is None else f'{value:,}'.replace(',', ' ')


def clean(value):
    return ''.join(c if c in '\n\t' or c.isprintable() else '�' for c in str(value))


class Terminal:
    def __init__(self, console=None):
        self.console = console or Console(force_terminal=True, color_system='truecolor',
                                          force_interactive=sys.stdout.isatty())
        self.console.width = min(self.console.width, 110)
        self.current_strategy = None

    def panel(self, title, text, kind='answer'):
        self.console.print(Panel(Text(clean(text)), title=Text(clean(title), style='bold'),
                                 title_align='left', box=box.ROUNDED, padding=(0, 2),
                                 border_style=COLORS[kind]))

    @contextmanager
    def waiting(self, title='Агент работает · запрос к API'):
        original, termios = None, None
        if sys.stdin.isatty():
            try:
                import termios
                original = termios.tcgetattr(sys.stdin.fileno())
                muted = original.copy()
                muted[3] &= ~(termios.ECHO | termios.ECHONL)
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, muted)
            except (ImportError, OSError):
                original = None
        try:
            with self.console.status(title, spinner='dots'):
                yield
        finally:
            if original is not None:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSAFLUSH, original)

    def usage(self, usage, side_tokens=0, side_calls=0):
        self.console.print(Text.assemble(
            ('  Запрос к модели: ', 'bold ' + COLORS['user']),
            (f"вход {number(usage.get('prompt_tokens'))} + выход {number(usage.get('completion_tokens'))}", COLORS['user']),
            (f" = {number(usage.get('total_tokens'))} токенов · 1 вызов API", 'bold ' + COLORS['user'])))
        if side_calls:
            self.console.print(Text(
                f'  Обновление памяти: {number(side_tokens)} токенов · дополнительных вызовов API: {side_calls}',
                style=COLORS['tool']))
        else:
            self.console.print(Text('  Обновление памяти: не выполнялось · дополнительных запросов нет',
                                    style=COLORS['tool']))

    def progress(self, phase, index, total, usage, side_tokens, side_calls, running, ok=None, answer=None):
        if answer is not None:
            self.panel('АГЕНТ', answer, 'answer' if ok is not False else 'error')
        if ok is not None:
            self.console.print(Text('  ✓ Ответ совпал по ключевым словам' if ok else
                                    '  ! Ответ не прошёл проверку по ключевым словам — нужна сверка',
                                    style='bold ' + COLORS['answer' if ok else 'error']))
        self.usage(usage, side_tokens, side_calls)
        tokens = usage.get('total_tokens')
        step_tokens = None if tokens is None else tokens + side_tokens
        strategy = LABELS.get(self.current_strategy, 'текущей стратегии')
        self.console.print(Text.assemble(
            (f'  Расход шага: {number(step_tokens)} токенов', 'bold ' + COLORS['answer']),
            (f' · всего за прогон {strategy}: {number(running)} токенов', 'bold ' + COLORS['tool'])))
        self.console.print()

    def comparison(self, results, model, window):
        if not results:
            self.panel('НЕТ ЗАВЕРШЁННЫХ СТРАТЕГИЙ',
                       'Сравнение пока недоступно. Расход незавершённого прогона неизвестен.', 'tool')
            return
        self.console.rule(Text('СРАВНЕНИЕ СТРАТЕГИЙ', style='bold ' + COLORS['answer']))
        self.console.print(Text(f'Модель: {model} · окно: {window} сообщений', style='dim'))
        table = Table(box=box.SIMPLE_HEAVY)
        for title in ('Стратегия', 'Детали', 'Токены всего', 'Из них память¹', 'Вызовы API', 'Из них память¹', 'Строк²'):
            table.add_column(title, justify='left' if title == 'Стратегия' else 'right',
                             header_style='bold ' + COLORS['user'],
                             style=COLORS['tool'] if title == 'Из них память¹' else COLORS['answer'])
        for r in results:
            table.add_row(LABELS[r['name']], f"{r['correct']}/{r['questions']}",
                          number(r['tokens']), number(r['side_tokens']), str(r['calls']),
                          str(r['side_calls']), str(r['loc']))
        self.console.print(table)
        self.console.print(Text('¹ Служебные токены и вызовы уже включены в итог, повторно не прибавлять.\n'
                                '² Строки класса, включая комментарии: ориентир, а не оценка качества.\n'
                                'Проверка по ключевым словам приблизительная; ответы сохранены для сверки.', style='dim'))
        self.panel('КАКУЮ СТРАТЕГИЮ ВЫБРАТЬ',
                   'Sliding Window · короткие независимые вопросы, минимум расходов.\n'
                   'Sticky Facts · долгие рабочие диалоги с бюджетом, ограничениями и решениями.\n'
                   'Summary · связный рассказ и нюансы, которые сложно свести к ключам.', 'tool')

    def event(self, event, verbose=False):
        if event['kind'] == 'case_start':
            self.current_strategy = event['name']
            self.console.rule(Text('ТЕСТ · ' + LABELS[event['name']].upper(), style='bold ' + COLORS['user']))
            self.panel('КАК ПРОХОДИТ ПРОВЕРКА', STRATEGY_CLASSES[event['name']].description + '\n'
                       'Сначала проигрываем диалог о поездке, затем задаём вопросы по его деталям.\n'
                       'Каждый шаг: 1 основной запрос к модели + отдельные запросы поддержки памяти, если нужны.\n'
                       'Расход — вход и выход модели; накопленный итог считается отдельно для этой стратегии.', 'tool')
            return
        phase = 'ДИАЛОГ' if event['phase'] == 'dialogue' else 'ПРОВЕРКА ПАМЯТИ'
        title = f'{LABELS[self.current_strategy]} · {phase} {event["index"]:02}/{event["total"]} · ВЫ'
        self.panel(title, event['prompt'], 'user')
        self.progress('Диалог' if event['phase'] == 'dialogue' else 'Вопрос',
                      event['index'], event['total'], event['usage'], event['side_tokens'],
                      event['side_calls'], event['running'], ok=event['ok'],
                      answer=event['answer'] if verbose or event['phase'] == 'probe' else None)


def chat(client, terminal, strategy='sliding', window=4):
    agent = Agent(client, strategy, window)
    terminal.panel('ДЕНЬ 10 · ПЛАНИРУЕМ ЯПОНИЮ',
                   f'Стратегия: {LABELS[strategy]} · окно {window} сообщений\n'
                   'Синий — вы · зелёный — агент · жёлтый — память · красный — ошибка\n\n' + HELP)
    while True:
        try:
            text = terminal.console.input(
                f'[bold {COLORS["user"]}]{agent.current} · {agent.strategy.name} · Вы › [/]').strip()
        except EOFError:
            return 0
        if not text:
            continue
        cmd, _, arg = text.partition(' ')
        arg = arg.strip()
        try:
            if cmd in ('/exit', '/quit'):
                return 0
            if cmd == '/help':
                terminal.panel('КОМАНДЫ', HELP, 'user')
            elif cmd in ('/reset', '/new'):
                agent.reset()
                terminal.panel('НОВЫЙ ДИАЛОГ', 'История и память текущей ветки очищены. Остальные ветки сохранены.')
            elif cmd == '/strategy':
                agent.set_strategy(arg)
                terminal.panel('СТРАТЕГИЯ', f'{LABELS[arg]} · окно {agent.strategy.window}\n'
                               'Память новой стратегии начинается с нуля; история не изменяется.', 'tool')
            elif cmd == '/window':
                if not arg.isdigit():
                    raise ValueError('Нужно число ≥ 2, например: /window 6')
                agent.set_window(int(arg))
                terminal.panel('ОКНО', f'{agent.strategy.window} сообщений. Ранее забытое не восстанавливается.', 'tool')
            elif cmd in ('/checkpoint', '/branch'):
                agent.checkpoint(arg, switch=cmd == '/branch')
                terminal.panel('CHECKPOINT' if cmd == '/checkpoint' else 'НОВАЯ ВЕТКА',
                               f'Снимок «{arg}» сохранён · текущая ветка: {agent.current}\n'
                               'История, стратегия и её память копируются независимо.', 'tool')
            elif cmd == '/switch':
                agent.switch(arg)
                terminal.panel('ВЕТКА ПЕРЕКЛЮЧЕНА',
                               f'{agent.current} · {LABELS[agent.strategy.name]} · {len(agent.messages) - 1} сообщений', 'tool')
            elif cmd == '/tree':
                terminal.panel('ВЕТКИ И CHECKPOINTS', '\n'.join(
                    f'{"→" if row["current"] else "·"} {row["name"]} · {row["messages"]} сообщений · '
                    f'{LABELS[row["strategy"]]}' for row in agent.tree()), 'tool')
            elif cmd == '/history':
                terminal.panel('ИСТОРИЯ ДИАЛОГА', json.dumps(agent.messages, ensure_ascii=False, indent=2), 'user')
                terminal.panel('КОНТЕКСТ ЗАПРОСА',
                               json.dumps(agent.strategy.context(agent.messages), ensure_ascii=False, indent=2), 'tool')
            elif cmd in ('/facts', '/summary'):
                value = getattr(agent.strategy, 'facts' if cmd == '/facts' else 'summary', None)
                terminal.panel(cmd[1:].upper(), json.dumps(value, ensure_ascii=False, indent=2)
                               if value is not None else 'У текущей стратегии такого блока нет.', 'tool')
            elif cmd.startswith('/'):
                raise ValueError('Неизвестная команда. Список: /help')
            else:
                terminal.panel('ВЫ', text, 'user')
                with terminal.waiting():
                    reply = agent.ask(text)
                terminal.panel('АГЕНТ', reply['answer'])
                if reply['warning']:
                    terminal.panel('ПАМЯТЬ НЕ ОБНОВЛЕНА', reply['warning'], 'error')
                terminal.usage(reply['usage'], reply['side_tokens'], reply['side_calls'])
        except (ClientError, ValueError) as error:
            terminal.panel('ОШИБКА', str(error), 'error')


def main():
    parser = argparse.ArgumentParser(description='День 10 · управление контекстом и ветвление диалога')
    parser.add_argument('--env', type=Path, default=ROOT / '.env', help='Файл настроек API')
    parser.add_argument('--model', help='Модель, иначе LLM_MODEL из .env')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--demo', action='store_true', help='Автоматический прогон трёх стратегий')
    modes.add_argument('--chat', action='store_true', help='Ручной диалог с ветвлением')
    modes.add_argument('--show-results', type=Path, help='Показать сохранённый отчёт без API')
    parser.add_argument('--strategy', choices=tuple(STRATEGY_CLASSES), default='sliding', help='Стратегия ручного чата')
    parser.add_argument('--case', choices=tuple(STRATEGY_CLASSES), help='Прогнать только одну стратегию')
    parser.add_argument('--window', type=int, default=4, help='Число последних сообщений, без system и памяти')
    parser.add_argument('--output-dir', type=Path, help='Новая папка результатов эксперимента')
    parser.add_argument('--verbose', action='store_true', help='Показать полные реплики эксперимента')
    args = parser.parse_args()
    if args.window < 2:
        parser.error('--window должен быть ≥ 2')
    if args.case and (args.chat or args.show_results):
        parser.error('--case используется только для эксперимента')
    if args.output_dir and (args.chat or args.show_results):
        parser.error('--output-dir используется только для эксперимента')
    terminal = Terminal()
    try:
        if args.show_results:
            report = load_report(args.show_results)
            terminal.panel('СОХРАНЁННЫЙ ПРОГОН · БЕЗ API', str(args.show_results), 'user')
            terminal.comparison(report['runs'], report['model'], report['window'])
            if report.get('error'):
                terminal.panel('ПРОГОН НЕ ЗАВЕРШЁН', report['error'], 'error')
            return 0
        if args.case:
            args.demo = True
        if not args.demo and not args.chat:
            terminal.panel('ДЕНЬ 10 · УПРАВЛЕНИЕ КОНТЕКСТОМ',
                           'Япония · 8 ходов · 10 контрольных вопросов\n\n'
                           '1 · Все три стратегии и сравнение\n'
                           '2 · Только Sliding Window\n3 · Только Sticky Facts\n4 · Только Summary\n'
                           '5 · Ручной чат и ветвление\n6 · Сохранённые результаты, без API\n0 · Выход')
            while True:
                choice = terminal.console.input('[bold #92b7ff]Режим › [/]').strip()
                if choice in ('0', '1', '2', '3', '4', '5', '6'):
                    break
                terminal.console.print('Введите число от 0 до 6.', style=COLORS['error'])
            if choice == '0':
                return 0
            if choice == '6':
                report = load_report(ROOT / 'results' / 'experiment-1' / 'report.json')
                terminal.panel('СОХРАНЁННЫЙ ПРОГОН', 'Модель coder · сравнение трёх стратегий, без новых запросов API.', 'user')
                terminal.comparison(report['runs'], report['model'], report['window'])
                return 0
            args.chat = choice == '5'
            args.demo = not args.chat
            args.case = {'2': 'sliding', '3': 'sticky', '4': 'summary'}.get(choice)
        if args.output_dir and args.chat:
            parser.error('--output-dir используется только для эксперимента')
        settings = load_settings(args.env)
        if args.model:
            settings['LLM_MODEL'] = args.model
        client = ChatClient(settings)
        if args.chat:
            return chat(client, terminal, args.strategy, args.window)
        directory = args.output_dir or ROOT / 'results' / ('local-day10-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
        terminal.panel('ДЕНЬ 10 · ТЕСТ ПАМЯТИ',
                       f'Модель: {client.model_name} · окно {args.window} сообщений\n'
                       'Япония · 8 ходов · 10 контрольных вопросов\n'
                       'Диалог, вопросы и служебная память включены в расход.', 'user')
        with terminal.waiting('Эксперимент · реальные запросы API'):
            report = run_experiment(client, [args.case] if args.case else list(STRATEGY_CLASSES),
                                    directory, args.window, lambda event: terminal.event(event, args.verbose))
        terminal.comparison(report['runs'], report['model'], report['window'])
        terminal.panel('ОТЧЁТЫ', f'{directory / "report.md"}\n{directory / "report.json"}')
        if report.get('error'):
            terminal.panel('ПРОГОН ОСТАНОВЛЕН', report['error'] + '\nНезавершённая стратегия не включена в итог.', 'error')
        return 0 if report['status'] == 'completed' else 1
    except (ClientError, OSError, ValueError, KeyError, TypeError) as error:
        terminal.panel('ОШИБКА', str(error), 'error')
        return 1
    except (KeyboardInterrupt, EOFError):
        terminal.console.print('\nРабота остановлена.', style='dim')
        return 130
