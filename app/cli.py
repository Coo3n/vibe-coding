"""Compact Rich terminal, adapted from task-6 panels and network indication."""

import argparse
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import sys

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .agent import Agent
from .client import ChatClient, ClientError, load_settings
from .context import Context, ContextError
from .experiment import run_experiment
from .protocol import ProtocolError, dumps
from .report import number, write_report
from .scenario import FACTS
from .tokens import usage_total


ROOT = Path(__file__).resolve().parent.parent
COLORS = {'user': '#92b7ff', 'answer': '#76ddbd', 'tool': '#e7ba76', 'error': '#ff8a96'}
ERRORS = (ClientError, ContextError, ProtocolError, OSError, ValueError)


def clean(value):
    return ''.join(c if c in '\n\t' or c.isprintable() else '�' for c in str(value))


class Terminal:
    def __init__(self):
        self.console = Console()
        self.console.width = min(self.console.width, 110)

    def panel(self, title, text, kind='answer'):
        self.console.print(Panel(Text(clean(text)), title=Text(clean(title), style='bold'),
                                 title_align='left', box=box.ROUNDED, padding=(0, 2),
                                 border_style=COLORS[kind]))

    @contextmanager
    def waiting(self, title):
        # On a terminal, discard input typed while the request is being processed.
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

    def reply(self, reply):
        text = reply['answer']
        if reply['questions']:
            text += '\n\n' + '\n'.join(f'{i}. {q}' for i, q in enumerate(reply['questions'], 1))
        if reply['next_step']:
            text += '\n\nСледующий шаг: ' + reply['next_step']
        self.panel('АГЕНТ', text)

    def event(self, event):
        kind = event['kind']
        if kind == 'dialogue':
            self.panel(f"ВЫ · {event['number']}/{event['total']}", event['prompt'], 'user')
        elif kind == 'request':
            self.console.print(Text(f"  → LLM · шаг {event['step']}", style='dim'))
        elif kind == 'tool':
            result = event['result']
            if 'error' in result:
                self.panel('ОШИБКА ИНСТРУМЕНТА', dumps(result), 'error')
            elif event['name'] == 'get_products':
                text = '\n'.join(f"{p['name']} ({p['id']}) · {p['purchase_rub_per_kg']} ₽/кг · "
                                 f"остаток {p['stock_kg']} кг" for p in result['products'])
                self.panel('TOOL CALL · get_products', text + '\nУчебный каталог', 'tool')
            else:
                text = '\n'.join(f"{item['product_id']} · {item['kg']} кг → {item['amount_rub']:g} ₽"
                                 for item in result['items'])
                text += (f"\nОвощи: {result['goods_rub']:g} ₽ · упаковка: {result['packaging_rub']:g} ₽"
                         f" · доставка: {result['delivery_rub']:g} ₽\nИтого: {result['total_rub']:g} ₽")
                self.panel('TOOL CALL · calculate_batch', text, 'tool')
        elif kind == 'answer':
            self.reply(event['reply'])
            self.usage(event['trace'])
        elif kind == 'compression_start':
            self.console.print(Text(f"  → Summary · после {event['after']} видимых сообщений", style=COLORS['tool']))
        elif kind == 'compression':
            data = event['event']
            self.panel('ИСТОРИЯ СЖАТА', f"Старые ходы заменены JSON-summary. "
                       f"Хвост: {data['tail_visible_messages']} видимых сообщений.\n" +
                       '\n'.join('• ' + s for s in data['summary']['facts']), 'tool')
            self.usage(data['trace'])
        elif kind == 'probe_start':
            if event['number'] == 1:
                self.panel('СЛУЧАЙ 1 · БЕЗ СЖАТИЯ' if event['mode'] == 'full' else 'СЛУЧАЙ 2 · СО СЖАТИЕМ',
                           'Каждый вопрос получает всю исходную переписку.' if event['mode'] == 'full' else
                           'Каждый вопрос получает summary и последние сообщения.\n'
                           'Расход создания summary включён в итог этого случая.')
            label = 'полная история' if event['mode'] == 'full' else 'summary + хвост'
            self.console.print(Text(f"  → Память {event['number']}/20 · {label} · {event['fact']['label']}", style='dim'))
        elif kind == 'probe':
            evaluation = event['probe']['evaluation']
            mark = '✓' if evaluation['correct'] else 'нет ответа' if evaluation['value'] is None else '≠ сверить'
            self.console.print(Text(f"    {mark}  {evaluation['value']}",
                                    style=COLORS['answer' if evaluation['correct'] else 'error']))

    def usage(self, traces):
        stats = usage_total(traces)
        self.console.print(Text(f"  API: {stats['requests']} · вход {number(stats['prompt_tokens'])} · "
                                f"выход {number(stats['completion_tokens'])} · всего {number(stats['total_tokens'])}", style='dim'))

    def comparison(self, report):
        table = Table(title='ПАМЯТЬ · 20 заранее заданных деталей', box=box.SIMPLE_HEAVY)
        for heading in ('Деталь', 'Эталон', 'Без сжатия', 'Со сжатием'):
            table.add_column(heading)
        probes = {mode: {p['fact_id']: p for p in report['probes'][mode]} for mode in ('full', 'compressed')}
        for fact in report['facts']:
            cells = []
            for mode in ('full', 'compressed'):
                evaluation = probes[mode].get(fact['id'], {}).get('evaluation')
                value = '—' if evaluation is None else str(evaluation['value'])
                mark = '' if evaluation is None else '✓ ' if evaluation['correct'] else '≠ '
                cells.append(Text(clean(mark + value), style='green' if evaluation and evaluation['correct'] else 'yellow'))
            table.add_row(Text(fact['label']), Text(fact['expected']), *cells)
        self.console.print(table)
        scores = report['score']
        comparison = Table(title='ДВА СЛУЧАЯ · одинаковые 20 вопросов', box=box.SIMPLE_HEAVY)
        comparison.add_column('Показатель')
        comparison.add_column('1. Без сжатия', justify='right')
        comparison.add_column('2. Со сжатием', justify='right')
        comparison.add_row('Контекст каждого вопроса', 'Вся переписка', 'Summary + хвост')
        comparison.add_row('Детали: точные совпадения',
                           f"{scores['full']['correct']}/20", f"{scores['compressed']['correct']}/20")
        for key, label in (('prompt_tokens', 'Входные токены'), ('completion_tokens', 'Выходные токены'),
                           ('total_tokens', 'Всего токенов')):
            comparison.add_row(label, number(report['usage']['full_comparison'][key]),
                                number(report['usage']['compressed_comparison'][key]))
        comparison.add_row('Из них создание summary', '0', number(report['usage']['summary']['total_tokens']))
        self.console.print(comparison)
        self.console.print(Text('  Summary уже включено в итог второго случая — повторно не прибавлять.\n'
                                '  Сравниваем проверки одной готовой переписки. Общая подготовка учтена отдельно в report.json.\n'
                                '  Несовпадение формулировки требует сверки: это не обязательно забытый факт.', style='dim'))
        saving = report['savings']
        self.panel('ИТОГ', f"Экономия второго случая с учётом summary: {number(saving['tokens'])} токенов "
                   f"({saving['percent'] if saving['percent'] is not None else '?'}%).\n"
                   f"Статус: {report['status']}. Отсутствующий usage обозначен «?».",
                   'answer' if report['status'] == 'completed' else 'error')


def chat(client, terminal, args):
    path = args.session or ROOT / 'data' / f'session-{args.mode}.json'
    context = Context.load(path) if path.exists() else Context(args.mode, args.keep_last, args.compress_every)
    if (context.mode, context.keep_last, context.compress_every) != (args.mode, args.keep_last, args.compress_every):
        raise ContextError('Настройки сохранённой сессии отличаются. Укажите другой --session или прежние настройки.')
    agent = Agent(client, context, path, on_event=terminal.event)
    terminal.panel('ОВОЩНОЙ ПРИЛАВОК · ЧАТ',
                   f"Режим: {args.mode} · последние {args.keep_last} сообщений · порция {args.compress_every}\n"
                   '/summary — JSON-summary · /history — хвост · /new — новый диалог · /exit — выход\n'
                   'Введите обычный текст: программа отправит его в JSON-обёртке.')
    while True:
        try:
            text = terminal.console.input('[bold #92b7ff]Вы › [/]').strip()
        except EOFError:
            return 0
        if text.lower() in ('/exit', '/quit'):
            return 0
        if text.lower() == '/summary':
            terminal.panel('SUMMARY · JSON', dumps(agent.context.summary))
        elif text.lower() == '/history':
            terminal.panel('НЕСЖАТЫЙ ХВОСТ · JSON', dumps(agent.context.turns))
        elif text.lower() == '/new':
            fresh = Context(args.mode, args.keep_last, args.compress_every)
            fresh.save(path)
            agent = Agent(client, fresh, path, on_event=terminal.event)
            terminal.panel('НОВЫЙ ДИАЛОГ', 'История и summary очищены.')
        elif text:
            terminal.panel('ВЫ', text, 'user')
            before = agent.context.compressed_messages
            try:
                with terminal.waiting('Агент работает · ввод временно отключён'):
                    agent.ask(text)
                terminal.reply(agent.reply)
                if agent.context.compressed_messages != before:
                    terminal.panel('SUMMARY ОБНОВЛЕНО', f"Всего сжато: {agent.context.compressed_messages} сообщений. "
                                   f"В хвосте: {agent.context.visible_messages}. Просмотр: /summary", 'tool')
            except ERRORS as error:
                terminal.panel('ОШИБКА · ХОД НЕ СОХРАНЁН', str(error), 'error')
            terminal.usage(agent.trace)


def main():
    parser = argparse.ArgumentParser(description='День 9 · овощной агент, tool calling и JSON-summary')
    parser.add_argument('--env', type=Path, default=ROOT / '.env')
    parser.add_argument('--model', help='Маршрут модели, иначе LLM_MODEL из .env')
    parser.add_argument('--demo', action='store_true', help='Автономный эксперимент: диалог, summary и проверки памяти')
    parser.add_argument('--turns', type=int, choices=(15, 30), default=15,
                        help='Ходов user+agent: 15 = 30 сообщений по ТЗ (по умолчанию), 30 = 60 сообщений')
    parser.add_argument('--chat', action='store_true', help='Ручной чат вместо меню')
    parser.add_argument('--mode', choices=('full', 'compressed'), default='compressed')
    parser.add_argument('--keep-last', type=int, default=8)
    parser.add_argument('--compress-every', type=int, default=10)
    parser.add_argument('--output-dir', type=Path, help='Новая папка результатов эксперимента')
    parser.add_argument('--session', type=Path, help='Файл ручного чата')
    args = parser.parse_args()
    if args.demo and args.chat:
        parser.error('Выберите --demo или --chat.')
    if args.output_dir and not args.demo:
        parser.error('--output-dir используется с --demo.')
    terminal = Terminal()
    try:
        Context(args.mode, args.keep_last, args.compress_every)
        settings = load_settings(args.env)
        if args.model:
            settings['LLM_MODEL'] = args.model
        client = ChatClient(settings)
        if not args.demo and not args.chat:
            terminal.panel('ДЕНЬ 9 · ОВОЩНОЙ ПРИЛАВОК',
                           f'1 · Эксперимент: {args.turns * 2} сообщений, tools, summary, 20 проверок × 2\n'
                           '2 · Ручной чат со сжатием\n3 · Ручной чат без сжатия\n0 · Выход')
            choice = terminal.console.input('Выберите режим › ').strip()
            if choice == '0':
                return 0
            if choice not in ('1', '2', '3'):
                raise ValueError('Выберите 0, 1, 2 или 3.')
            args.demo = choice == '1'
            args.mode = 'full' if choice == '3' else 'compressed'
        if args.demo:
            directory = args.output_dir or ROOT / 'results' / ('local-day9-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
            terminal.panel('ЭКСПЕРИМЕНТ · ЧТО СРАВНИВАЕМ',
                           f'Подготавливаем один диалог на {args.turns * 2} сообщений.\n'
                           'Случай 1: без сжатия — вся переписка.\n'
                           'Случай 2: со сжатием — summary + последние сообщения.\n'
                           '20 вопросов к каждому контексту; ответы проверок не освежают память.\n'
                           'Расход summary включён в сравнение. Всё общение с LLM — JSON.\n'
                           f'Модель: {client.model_name} · N={args.keep_last} · порция={args.compress_every}')
            terminal.panel('ФАКТЫ ЗАДАНЫ ЗАРАНЕЕ', '\n'.join(f"{i}. {f['label']}: {f['expected']}"
                           for i, f in enumerate(FACTS, 1)), 'user')
            with terminal.waiting('Эксперимент · реальные запросы API'):
                report = run_experiment(client, directory, terminal.event, args.keep_last, args.compress_every, args.turns)
            write_report(report, directory)
            terminal.comparison(report)
            terminal.panel('ОТЧЁТЫ', f'{directory / "report.md"}\n{directory / "report.json"}')
            if report.get('error'):
                terminal.panel('ПРИЧИНА ОСТАНОВКИ', report['error'], 'error')
            return 0 if report['status'] == 'completed' else 1
        return chat(client, terminal, args)
    except ERRORS as error:
        terminal.panel('ОШИБКА', str(error), 'error')
        return 1
    except (KeyboardInterrupt, EOFError):
        terminal.console.print('\nРабота остановлена.', style='dim')
        return 130
