"""Rich-консоль Lab11 с меню, чатом и отчётом по инвариантам."""

import argparse
from contextlib import contextmanager
from datetime import datetime
import json
from pathlib import Path
import sqlite3
import sys

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .agent import Agent
from .experiment import run_experiment
from .invariants import FIELDS, InvariantStore, display
from .memory import MemoryStore
from .report import GROUPS, load_report
from .scenario import OfflineClient


COLORS = {'user': '#92b7ff', 'answer': '#76ddbd', 'tool': '#e7ba76', 'error': '#ff8a96'}
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / 'data' / 'long_term.json'
RULES_PATH = ROOT / 'data' / 'invariants.sqlite3'
ARCHIVED = ROOT / 'results' / 'experiment-1' / 'report.json'
HELP = ('/invariants · /invariant set <ключ> <значение> · /audit\n'
        '/layers · /short · /working · /long\n'
        '/clear short|working|long · /remember текст · /forget ключ\n'
        '/context · /history · /window N · /reset · /help · /exit')


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

    def usage(self, usage, provider='API'):
        if usage is None:
            return
        self.console.print(Text.assemble(
            ('  Запрос к модели: ', 'bold ' + COLORS['user']),
            (f"вход {number(usage.get('prompt_tokens'))} + выход {number(usage.get('completion_tokens'))}", COLORS['user']),
            (f" = {number(usage.get('total_tokens'))} токенов · 1 вызов {provider}", 'bold ' + COLORS['user'])))

    def begin_case(self, name, prompt, source):
        origin = {'codex': 'Codex CLI', 'offline': 'офлайн', 'local': 'команда',
                  'fixture': 'локальная фикстура'}[source]
        self.panel(f'ВЫ · {name} · {origin}', prompt, 'user')

    def end_case(self, case):
        if not case['prompt'] or not case['reply']:
            return
        reply = case['reply']
        kind = 'error' if reply['status'] in ('invalid', 'conflict', 'error') else 'answer'
        self.panel('АГЕНТ', reply['answer'], kind)
        self.usage(reply.get('usage'), 'Codex CLI' if case.get('source') == 'codex' else 'API')

    def comparison(self, report, verbose=False):
        if verbose:
            for case in report['cases']:
                if case['prompt']:
                    self.begin_case(case['name'], case['prompt'], case.get('source', 'offline'))
                    self.end_case(case)
        self.console.rule(Text('РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА', style='bold ' + COLORS['answer']))
        codex = report.get('mode') == 'codex'
        provider = (f"Codex CLI · {report['model']} · вызовов {report['codex_calls']}/{report['codex_attempts']} · "
                    f"{number(report['tokens'])} токенов с контекстом Codex" if codex else
                    f"Офлайн · вызовов API {report['api_calls']}")
        self.console.print(Text(f"{provider} · размер окна {report['window']}", style='dim'))
        table = Table(box=box.SIMPLE_HEAVY)
        for title in ('Группа', 'Проверка', 'Результат'):
            table.add_column(title, header_style='bold ' + COLORS['user'])
        for case in report['cases']:
            passed = case['passed']
            status = Text('✓' if passed else '✗', style=COLORS['answer' if passed else 'error'])
            table.add_row(GROUPS[case['group']], case['name'], status)
        self.console.print(table)
        code_passed = report.get('application_passed', report['passed'])
        summary = (f"Проверки приложения: {code_passed}/{report['total']}\n"
                   if codex else f"Проверки: {report['passed']}/{report['total']}\n")
        if codex:
            summary += (f"Реплики модели по сценарию: {report['model_matched']}/{report['model_total']}\n"
                        f"Отказ с ожидаемой причиной: {report['bypasses_blocked']}/{report['bypasses']}\n")
        summary += (f"Конфликтных вариантов выдано: {report['bypasses_accepted']}/{report['bypasses']}\n"
                    f"Локальных фикстур вне формата заблокировано: {report['malformed_blocked']}/{report['malformed']}")
        if report.get('error'):
            summary += '\nПрогон прерван. ' + report['error']
        ok = not report.get('error') and code_passed == report['total']
        self.panel('ИТОГ', summary, 'answer' if ok else 'error')


def show_rules(terminal, store):
    rules = store.active()
    lines = []
    for field, (label, options) in FIELDS.items():
        value = rules[field]
        lines.append(f'{label} · {display(field, value)}\n'
                     f'  {field} = {value} · значения: {" / ".join(options)}')
    terminal.panel('ИНВАРИАНТЫ', '\n'.join(lines), 'tool')


def chat(client, terminal, window=4):
    memory = MemoryStore(DATA_PATH, window)
    store = InvariantStore(RULES_PATH)
    agent = Agent(client, memory, store)
    mode = client.model_name if client.mode != 'offline' else 'офлайн'
    terminal.panel('ДЕНЬ 14 · ИНВАРИАНТЫ ПРОЕКТА',
                   f'Журнал проектов команды · {mode} · окно {window} сообщений\n'
                   'Синий — вы · зелёный — агент · жёлтый — память · красный — отказ\n\n' + HELP)
    while True:
        try:
            text = terminal.console.input(
                f'[bold {COLORS["user"]}]окно {memory.window} · Вы › [/]').strip()
        except EOFError:
            return 0
        if not text:
            continue
        line = text.split(maxsplit=1)
        cmd, arg = line[0], line[1] if len(line) == 2 else ''
        try:
            if cmd in ('/exit', '/quit'):
                return 0
            if cmd == '/help':
                terminal.panel('КОМАНДЫ', HELP, 'user')
            elif cmd == '/invariants':
                show_rules(terminal, store)
            elif cmd == '/invariant':
                entry = agent.change_invariant(arg)
                terminal.panel('ИНВАРИАНТ ИЗМЕНЁН',
                               f"{entry['field']}: {entry['old_value']} → {entry['new_value']}\n"
                               f"{entry['changed_at']} · запись доступна в /audit", 'tool')
            elif cmd == '/audit':
                terminal.panel('ЖУРНАЛ ИЗМЕНЕНИЙ', dump_json(store.history()), 'tool')
            elif cmd == '/layers':
                terminal.panel('СЛОИ ПАМЯТИ',
                               f'short-term: {len(memory.short_term)} сообщений (окно {memory.window})\n'
                               f'working: {len(memory.working)} записей\n'
                               f'long-term: {len(memory.long_term)} записей\n'
                               f'Инварианты на диске: {RULES_PATH}', 'tool')
            elif cmd == '/short':
                terminal.panel('SHORT-TERM', dump_json(memory.short_term[-memory.window:]), 'tool')
            elif cmd == '/working':
                agent.context()
                terminal.panel('WORKING', dump_json(memory.working), 'tool')
            elif cmd == '/long':
                terminal.panel('LONG-TERM', dump_json(memory.long_term), 'tool')
            elif cmd == '/clear':
                memory.clear(arg)
                terminal.panel('СЛОЙ ОЧИЩЕН', arg, 'tool')
            elif cmd == '/remember':
                key = memory.remember(arg)
                terminal.panel('ЗАПОМНЕНО', f'«{key}» добавлено в long-term.', 'tool')
            elif cmd == '/forget':
                memory.forget(arg)
                terminal.panel('УДАЛЕНО', f'«{arg}» удалено из long-term.', 'tool')
            elif cmd == '/context':
                terminal.panel('КОНТЕКСТ ЗАПРОСА', dump_json(agent.context()), 'tool')
            elif cmd == '/history':
                terminal.panel('ИСТОРИЯ ДИАЛОГА', dump_json(memory.short_term), 'user')
            elif cmd == '/window':
                if not arg.isdecimal() or int(arg) < 1:
                    raise ValueError('Укажите число ≥ 1, например /window 6.')
                memory.window = int(arg)
                terminal.panel('ОКНО', f'{memory.window} сообщений.', 'tool')
            elif cmd == '/reset':
                agent.reset()
                terminal.panel('НОВЫЙ ДИАЛОГ', 'Short-term и working очищены. Действующие правила доступны в /invariants.', 'tool')
            elif cmd.startswith('/'):
                raise ValueError('Неизвестная команда. Список доступен в /help.')
            else:
                terminal.panel('ВЫ', text, 'user')
                source = {'codex': 'Codex CLI', 'live': 'запрос к API', 'offline': 'офлайн'}[client.mode]
                title = f'Агент работает · {source}'
                with terminal.waiting(title):
                    reply = agent.ask(text)
                kind = 'error' if reply['status'] in ('invalid', 'conflict') else 'answer'
                terminal.panel('АГЕНТ', reply['answer'], kind)
                terminal.usage(reply['usage'], 'Codex CLI' if client.mode == 'codex' else 'API')
        except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
            terminal.panel('ОШИБКА', str(error), 'error')


def main():
    parser = argparse.ArgumentParser(description='День 14 · инварианты и ограничения состояния')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--demo', action='store_true', help='Автоматический эксперимент')
    modes.add_argument('--chat', action='store_true', help='Ручной диалог')
    modes.add_argument('--show-results', type=Path, help='Сохранённый отчёт')
    parser.add_argument('--window', type=int, default=4, help='Размер окна short-term')
    parser.add_argument('--output-dir', type=Path, help='Новая папка результатов эксперимента')
    parser.add_argument('--verbose', action='store_true', help='Полные реплики эксперимента (в --codex по умолчанию)')
    providers = parser.add_mutually_exclusive_group()
    providers.add_argument('--live', action='store_true', help='Использовать API в ручном диалоге')
    providers.add_argument('--codex', action='store_true', help='Использовать вход ChatGPT в Codex CLI')
    parser.add_argument('--model', help='Переопределить модель для --codex или --live')
    args = parser.parse_args()
    if args.window < 1:
        parser.error('--window должен быть ≥ 1')
    if args.output_dir and (args.chat or args.show_results):
        parser.error('--output-dir используется только для эксперимента')
    if args.live and not args.chat:
        parser.error('--live используется вместе с --chat')
    if args.codex and args.show_results:
        parser.error('--codex используется для эксперимента или чата')
    if args.model and not (args.live or args.codex):
        parser.error('--model используется вместе с --codex или --live')

    terminal = Terminal()
    try:
        if not args.demo and not args.chat and not args.show_results:
            terminal.panel('ДЕНЬ 14 · ИНВАРИАНТЫ',
                           '1 · Эксперимент проверки инвариантов\n2 · Ручной чат\n'
                           '3 · Сохранённые результаты\n0 · Выход')
            while True:
                choice = terminal.console.input('[bold #92b7ff]Режим › [/]').strip()
                if choice in ('0', '1', '2', '3'):
                    break
                terminal.console.print('Введите число от 0 до 3.', style=COLORS['error'])
            if choice == '0':
                return 0
            args.chat = choice == '2'
            if choice == '3':
                args.show_results = ARCHIVED
            if args.output_dir and choice != '1':
                parser.error('--output-dir используется только для эксперимента')
        if args.show_results:
            terminal.panel('СОХРАНЁННЫЙ ПРОГОН · БЕЗ API', str(args.show_results), 'user')
            terminal.comparison(load_report(args.show_results), args.verbose)
            return 0
        client = OfflineClient()
        if args.codex:
            from .codex_client import CodexClient

            client = CodexClient(args.model)
        elif args.live:
            from .client import ChatClient, load_settings

            settings = load_settings(ROOT / '.env')
            if args.model:
                settings['LLM_MODEL'] = args.model
            client = ChatClient(settings)
        if args.chat:
            return chat(client, terminal, args.window)

        directory = args.output_dir or ROOT / 'results' / ('local-day14-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
        provider = f'Codex CLI · {client.model_name}' if args.codex else 'офлайн'
        terminal.panel('ДЕНЬ 14 · ПРОВЕРКА ИНВАРИАНТОВ',
                       f'Журнал проектов команды · {provider} · восемь попыток обхода', 'user')
        trace = args.codex or args.verbose
        report = run_experiment(directory, args.window, client,
                                terminal.begin_case if trace else None, terminal.end_case if trace else None)
        terminal.panel('ОТЧЁТЫ', f'{directory / "report.md"}\n{directory / "report.json"}')
        terminal.comparison(report)
        return 0 if not report['error'] and report['application_passed'] == report['total'] else 1
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, sqlite3.Error) as error:
        terminal.panel('ОШИБКА', str(error), 'error')
        return 1
    except (KeyboardInterrupt, EOFError):
        terminal.console.print('\nРабота остановлена.', style='dim')
        return 130
