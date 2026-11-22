"""Компактная Rich-консоль: меню, диалог со слоями памяти и эксперимент."""

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
from .codex_client import CodexClient
from .memory import MemoryStore
from .experiment import run_experiment
from .report import LABELS, PHASE_LABELS, initial_memory_sections, load_report


COLORS = {'user': '#92b7ff', 'answer': '#76ddbd', 'tool': '#e7ba76', 'error': '#ff8a96'}
ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / '.env'
DATA_PATH = ROOT / 'data' / 'long_term.json'
HELP = ('/layers · /short · /working · /long\n'
        '/clear short|working|long · /remember текст · /forget ключ\n'
        '/context · /history · /window N · /reset · /help · /exit')

PHASES = {'dialogue': 'ДИАЛОГ', 'extra': 'ДОПОЛНЕНИЕ', 'probe': 'ПРОВЕРКА ПАМЯТИ',
          'reset': 'НОВЫЙ ДИАЛОГ', 'remember': 'ПОСЛЕ /REMEMBER'}

CONFIG_HELP = {
    'full': 'Все три слоя в промпте: long-term + working + окно последних сообщений. '
            'Это база, с которой сравниваем очистку слоёв.',
    'no-short': 'Short-term отключён: в промпт уходит только текущая реплика. '
                'Диалог помнят только working и long-term.',
    'no-working': 'Working отключён: данные задачи доступны только из окна сообщений '
                  'и долговременной памяти.',
    'no-long': 'Long-term отключён: факты Дня 10 не попадают в промпт, '
               'диалог держат только short-term и working.',
    'new-dialog': 'Новый диалог после /reset: short и working очищены. '
                  'На вопросы отвечает только long-term с фактами Дня 10.',
    'after-remember': 'Решения явно сохранены /remember в long-term, затем /reset. '
                      'Проверка, что явное сохранение переживает новый диалог.',
    'new-dialog-no-long': 'Финальная очистка: long-term сброшен, новый диалог. '
                          'Ни один слой не помнит поездку — агент не должен отвечать.',
}


def number(value):
    return '?' if value is None else f'{value:,}'.replace(',', ' ')


def clean(value):
    return ''.join(c if c in '\n\t' or c.isprintable() else '�' for c in str(value))


def dump_json(value):
    return json.dumps(value, ensure_ascii=False, indent=2)


class Terminal:
    def __init__(self, console=None):
        self.console = console or Console(force_terminal=True, color_system='truecolor',
                                          force_interactive=sys.stdout.isatty())
        self.console.width = min(self.console.width, 110)
        self.current_case = None
        self.call_label = 'вызов API'

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
            (f" = {number(usage.get('total_tokens'))} токенов · 1 {self.call_label}", 'bold ' + COLORS['user'])))
        if side_calls:
            self.console.print(Text(
                f'  Обновление памяти: {number(side_tokens)} токенов · дополнительных вызовов: {side_calls}',
                style=COLORS['tool']))
        else:
            self.console.print(Text('  Обновление памяти: не выполнялось · дополнительных запросов нет',
                                    style=COLORS['tool']))

    def progress(self, usage, side_tokens, side_calls, running, ok=None, answer=None):
        if answer is not None:
            self.panel('АГЕНТ', answer, 'answer' if ok is not False else 'error')
        if ok is not None:
            self.console.print(Text('  ✓ Ответ совпал по ключевым словам' if ok else
                                    '  ! Ответ не прошёл проверку по ключевым словам — нужна сверка',
                                    style='bold ' + COLORS['answer' if ok else 'error']))
        self.usage(usage, side_tokens, side_calls)
        tokens = usage.get('total_tokens')
        step_tokens = None if tokens is None else tokens + side_tokens
        self.console.print(Text.assemble(
            (f'  Расход шага: {number(step_tokens)} токенов', 'bold ' + COLORS['answer']),
            (f' · всего за прогон: {number(running)} токенов', 'bold ' + COLORS['tool'])))
        self.console.print()

    def comparison(self, report):
        runs = report.get('runs') or []
        measure = report.get('token_measure')
        self.console.rule(Text('РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА', style='bold ' + COLORS['answer']))
        self.console.print(Text(f"Модель: {report['model']} · окно: {report['window']} сообщений",
                                style='dim'))
        if report.get('mode') == 'codex':
            self.console.print(Text('Codex CLI · общий расход включает служебный контекст Codex. '
                                    'Дельты получены внутри одного прогона.', style='dim'))
        if report.get('tokens') is not None:
            self.console.print(Text(f"Расход прогона · {number(report['tokens'])} токенов · "
                                    f"{report.get('main_calls', '?')} основных и "
                                    f"{report.get('side_calls', 0)} служебных вызовов", style='dim'))
        if measure:
            table = Table(box=box.SIMPLE_HEAVY, title='Токены каждого слоя',
                          title_style='bold ' + COLORS['user'])
            for title in ('Слой', 'На запрос', 'За прогон (оценка¹)', 'prompt_tokens полных'):
                table.add_column(title, justify='left' if title == 'Слой' else 'right',
                                 header_style='bold ' + COLORS['user'],
                                 style=COLORS['tool'] if 'оценка' in title else COLORS['answer'])
            main_calls = report.get('main_calls')
            for layer, title in (('long', 'long-term'), ('working', 'working'),
                                 ('short', 'short-term (окно)')):
                value = measure['layer_tokens'][layer]
                estimate = number(value * main_calls) if main_calls else '—'
                table.add_row(title, str(value), estimate,
                              str(measure['prompt_tokens'].get('full')))
            self.console.print(table)
            self.console.print(Text(
                f"Сопровождение working: {number(measure.get('working_side_tokens'))} токенов в "
                f"{measure.get('working_side_calls')} служебных вызовах — измерено точно.\n"
                '¹ Токены слоя на репрезентативном запросе × основные вызовы; способ сравнения, '
                'не тарификация.', style='dim'))
            self.console.print()
        if runs:
            table = Table(box=box.SIMPLE_HEAVY, title='Ablation: очистка слоёв по очереди',
                          title_style='bold ' + COLORS['user'])
            for title in ('Конфигурация', 'Детали'):
                table.add_column(title, justify='left',
                                 header_style='bold ' + COLORS['user'], style=COLORS['answer'])
            for run in runs:
                label = LABELS.get(run['name'], run['name'])
                if run['phase'] != 'after-reset':
                    label += ' · ' + PHASE_LABELS.get(run['phase'], run['phase'])
                table.add_row(label, f"{run['correct']}/{run['total']}")
            self.console.print(table)
            self.console.print(Text('Проверка по ключевым словам приблизительная; '
                                    'полные ответы сохранены в report.json.', style='dim'))
        self.panel('ЧТО ПОКАЗЫВАЮТ СЛОИ',
                   'Short-term · дословные последние реплики: нюансы текущего разговора.\n'
                   'Working · данные задачи после каждой реплики: бюджет, ограничения, маршрут.\n'
                   'Long-term · явные записи на диске: решения переживают новый диалог.\n'
                   'Факты могут повторяться в нескольких слоях. После /reset '
                   'из предыдущего диалога остаётся только long-term.', 'tool')

    def event(self, event):
        kind = event['kind']
        if kind == 'initial_memory':
            self.initial_memory(event['memory'])
            return
        if kind == 'measure_prompt':
            self.panel(f'ЗАМЕР ТОКЕНОВ · {LABELS[event["name"]]} · ВЫ', event['prompt'], 'user')
            return
        if kind == 'measure':
            self.progress(event['usage'], 0, 0, event['running'], answer=event['answer'])
            return
        if kind == 'case_start':
            self.current_case = event['name']
            self.console.rule(Text('ТЕСТ · ' + LABELS[event['name']].upper(),
                                   style='bold ' + COLORS['user']))
            self.panel('КАК ПРОХОДИТ ПРОВЕРКА', CONFIG_HELP[event['name']] + '\n'
                       'Каждый шаг: 1 основной запрос к модели + обновление working-памяти, '
                       'если слой включён. Накопленный итог охватывает весь прогон.', 'tool')
            return
        if kind == 'prompt':
            phase = PHASES[event['phase']]
            case = LABELS.get(self.current_case, 'Диалог') if self.current_case else 'Диалог'
            title = f'{case} · {phase} {event["index"]:02}/{event["total"]} · ВЫ'
            self.panel(title, event['prompt'], 'user')
            return
        self.progress(event['usage'], event['side_tokens'], event['side_calls'],
                      event['running'], ok=event['ok'], answer=event['answer'])

    def initial_memory(self, memory):
        sections = ('\n'.join([title, *(f'  {entry}' for entry in entries)])
                    for title, entries in initial_memory_sections(memory))
        self.panel('ИСХОДНЫЕ ДАННЫЕ В ПАМЯТИ', '\n\n'.join(sections), 'tool')


def show_layer(terminal, name, title, value, window=4):
    if name == 'short':
        tail = value[-window:]
        text = (f'{len(value)} сообщений текущего диалога (в окно уходит {len(tail)}):\n\n'
                + (dump_json([{'role': m["role"], 'content': m["content"]} for m in tail])
                   if tail else '(пусто)'))
    else:
        text = dump_json(value) if value else '(пусто)'
    terminal.panel(title, text, 'tool')


def chat(client, terminal, window=4):
    memory = MemoryStore(DATA_PATH, window)
    agent = Agent(client, memory)
    terminal.panel('ДЕНЬ 11 · ПЛАНИРУЕМ ЯПОНИЮ',
                   f'Окно {window} сообщений · long-term: {len(memory.long_term)} записей на диске\n'
                   'Синий — вы · зелёный — агент · жёлтый — память · красный — ошибка\n\n' + HELP)
    while True:
        try:
            text = terminal.console.input(
                f'[bold {COLORS["user"]}]окно {memory.window} · Вы › [/]').strip()
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
            elif cmd == '/layers':
                terminal.panel('СЛОИ ПАМЯТИ',
                               f'short-term: {len(memory.short_term)} сообщений (окно {memory.window})\n'
                               f'working: {len(memory.working)} ключей данных задачи\n'
                               f'long-term: {len(memory.long_term)} записей · файл '
                               f'{DATA_PATH}', 'tool')
            elif cmd == '/short':
                show_layer(terminal, 'short', 'SHORT-TERM', memory.short_term)
            elif cmd == '/working':
                show_layer(terminal, 'working', 'WORKING', memory.working)
            elif cmd == '/long':
                show_layer(terminal, 'long', 'LONG-TERM', memory.long_term)
            elif cmd == '/clear':
                if arg not in ('short', 'working', 'long'):
                    raise ValueError('Укажите слой: /clear short|working|long')
                memory.clear(arg)
                terminal.panel('СЛОЙ ОЧИЩЕН',
                               {'short': 'short-term: сообщения текущего диалога удалены.',
                                'working': 'working: данные задачи сброшены.',
                                'long': 'long-term: записи удалены, файл на диске перезаписан.'}[arg])
            elif cmd == '/remember':
                key = memory.remember(arg)
                terminal.panel('ЗАПОМНЕНО НАВСЕГДА',
                               f'Ключ «{key}» → long-term ({len(memory.long_term)} записей).\n'
                               'Запись переживает /reset и перезапуск процесса.', 'tool')
            elif cmd == '/forget':
                memory.forget(arg)
                terminal.panel('УДАЛЕНО', f'Ключ «{arg}» удалён из long-term '
                               f'({len(memory.long_term)} записей).', 'tool')
            elif cmd == '/context':
                terminal.panel('КОНТЕКСТ ЗАПРОСА',
                               dump_json(memory.context(agent.system_prompt)), 'tool')
            elif cmd == '/history':
                terminal.panel('ИСТОРИЯ ДИАЛОГА', dump_json(memory.short_term), 'user')
            elif cmd == '/window':
                if not arg.isdigit() or int(arg) < 1:
                    raise ValueError('Нужно число ≥ 1, например: /window 6')
                memory.window = int(arg)
                terminal.panel('ОКНО', f'{memory.window} сообщений. Ранее выпавшее из окна '
                               'в промпт не вернётся.', 'tool')
            elif cmd == '/reset':
                agent.reset()
                terminal.panel('НОВЫЙ ДИАЛОГ',
                               'Short-term и working очищены. Long-term '
                               f'({len(memory.long_term)} записей) пережил сброс.', 'tool')
            elif cmd.startswith('/'):
                raise ValueError('Неизвестная команда. Список: /help')
            else:
                terminal.panel('ВЫ', text, 'user')
                with terminal.waiting(f'Агент работает · {terminal.call_label}'):
                    reply = agent.ask(text)
                terminal.panel('АГЕНТ', reply['answer'])
                terminal.usage(reply['usage'], reply['side_tokens'], reply['side_calls'])
        except (ClientError, ValueError) as error:
            terminal.panel('ОШИБКА', str(error), 'error')


def main():
    parser = argparse.ArgumentParser(description='День 11 · модель памяти ассистента')
    providers = parser.add_mutually_exclusive_group()
    providers.add_argument('--live', action='store_true', help='Использовать прямой API (режим по умолчанию)')
    providers.add_argument('--codex', action='store_true', help='Использовать вход ChatGPT в Codex CLI')
    parser.add_argument('--model', help='Имя модели для выбранного клиента')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--demo', action='store_true', help='Автоматический прогон эксперимента')
    modes.add_argument('--chat', action='store_true', help='Ручной диалог со слоями памяти')
    modes.add_argument('--show-results', type=Path, help='Показать сохранённый отчёт без API')
    parser.add_argument('--window', type=int, default=4,
                        help='Число последних сообщений short-term в промпте')
    parser.add_argument('--output-dir', type=Path, help='Новая папка результатов эксперимента')
    parser.add_argument('--verbose', action='store_true', help='Полные реплики эксперимента выводятся по умолчанию')
    args = parser.parse_args()
    if args.window < 1:
        parser.error('--window должен быть ≥ 1')
    if args.output_dir and (args.chat or args.show_results):
        parser.error('--output-dir используется только для эксперимента')
    if args.show_results and (args.live or args.codex or args.model):
        parser.error('--show-results читает сохранённый отчёт')
    provider_label = 'Codex CLI' if args.codex else 'API'
    terminal = Terminal()
    try:
        if args.show_results:
            report = load_report(args.show_results)
            terminal.panel('СОХРАНЁННЫЙ ПРОГОН · БЕЗ API', str(args.show_results), 'user')
            if 'initial_memory' in report:
                terminal.initial_memory(report['initial_memory'])
            terminal.comparison(report)
            if report.get('error'):
                terminal.panel('ПРОГОН НЕ ЗАВЕРШЁН', report['error'], 'error')
            return 0
        if not args.demo and not args.chat:
            terminal.panel('ДЕНЬ 11 · МОДЕЛЬ ПАМЯТИ',
                           'Япония · 8 ходов + 2 новых решения · 9+4+2 контрольных вопроса\n\n'
                           '1 · Эксперимент: слои, токены, ablation\n'
                           '2 · Ручной чат со слоями памяти\n'
                           '3 · Сохранённые результаты, без API\n0 · Выход')
            while True:
                choice = terminal.console.input('[bold #92b7ff]Режим › [/]').strip()
                if choice in ('0', '1', '2', '3'):
                    break
                terminal.console.print('Введите число от 0 до 3.', style=COLORS['error'])
            if choice == '0':
                return 0
            if choice == '3':
                report = load_report(ROOT / 'results' / 'experiment-1' / 'report.json')
                terminal.panel('СОХРАНЁННЫЙ ПРОГОН', 'Модель ' + report['model'] +
                               ' · слои, токены и ablation, без новых запросов API.', 'user')
                if 'initial_memory' in report:
                    terminal.initial_memory(report['initial_memory'])
                terminal.comparison(report)
                return 0
            args.chat = choice == '2'
            args.demo = not args.chat
        if args.codex:
            client = CodexClient(args.model)
            terminal.call_label = 'ход Codex'
        else:
            settings = load_settings(ENV_PATH)
            if args.model:
                settings['LLM_MODEL'] = args.model
            client = ChatClient(settings)
        if args.chat:
            return chat(client, terminal, args.window)
        directory = args.output_dir or ROOT / 'results' / ('local-day11-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
        terminal.panel('ДЕНЬ 11 · ТЕСТ ПАМЯТИ',
                       f'{client.model_name} · {provider_label} · окно {args.window} сообщений\n'
                       'Япония · 8 ходов + 2 решения · 9 вопросов ablation × 4 конфигурации\n'
                       'Новый диалог после /reset · явное /remember · финальная очистка.\n'
                       'Диалог, вопросы, замеры и служебная память включены в расход.', 'user')
        with terminal.waiting(f'Эксперимент · {provider_label}'):
            report = run_experiment(client, directory, directory / 'long_term.json', args.window,
                                    terminal.event)
        terminal.panel('ОТЧЁТЫ', f'{directory / "report.md"}\n{directory / "report.json"}')
        if report.get('error'):
            terminal.panel('ПРОГОН ОСТАНОВЛЕН', report['error'], 'error')
        terminal.comparison(report)
        return 0 if report['status'] == 'completed' else 1
    except (ClientError, OSError, ValueError, KeyError, TypeError) as error:
        terminal.panel('ОШИБКА', str(error), 'error')
        return 1
    except (KeyboardInterrupt, EOFError):
        terminal.console.print('\nРабота остановлена.', style='dim')
        return 130
