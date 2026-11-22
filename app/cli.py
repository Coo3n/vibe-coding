"""Rich-консоль с меню, диалогом и проверкой конечного автомата."""

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

from .agent import Agent, ModelReplyError, OfflineClient
from .client import ChatClient, ClientError, load_settings
from .codex_client import CodexClient
from .experiment import run_codex_experiment, run_experiment
from .memory import MemoryStore
from .report import load_report
from .state import StateStore


COLORS = {'user': '#92b7ff', 'answer': '#76ddbd', 'tool': '#e7ba76', 'error': '#ff8a96'}
ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / '.env'
DATA_PATH = ROOT / 'data' / 'long_term.json'
STATE_PATH = ROOT / 'data' / 'task.json'
REPORT_PATH = ROOT / 'results' / 'experiment-1' / 'report.json'
HELP = (
    '/task <цель> · /state · /plan [текст] · /approve · /execute [результат]\n'
    '/validate · /validate pass|fail <результат> · /revise <результат> · /finish\n'
    '/pause · /resume · /reset · /layers · /short · /working · /long\n'
    '/clear short|long · /remember текст · /forget ключ · /context · /history\n'
    '/window N · /help · /exit'
)
TASK_ACTIONS = {'/plan': 'plan', '/approve': 'approve', '/execute': 'execute',
                '/revise': 'revise', '/finish': 'finish', '/pause': 'pause',
                '/resume': 'resume'}


def dump_json(value):
    return json.dumps(value, ensure_ascii=False, indent=2)


def clean(value):
    return ''.join(c if c in '\n\t' or c.isprintable() else '�' for c in str(value))


class Terminal:
    def __init__(self):
        self.console = Console(force_terminal=True, color_system='truecolor',
                               force_interactive=sys.stdout.isatty())
        self.console.width = min(self.console.width, 110)

    def panel(self, title, text, kind='answer'):
        self.console.print(Panel(Text(clean(text)), title=Text(clean(title), style='bold'),
                                 title_align='left', box=box.ROUNDED, padding=(0, 2),
                                 border_style=COLORS[kind]))

    @contextmanager
    def waiting(self, provider):
        title = f'Агент работает · {provider}'
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
        if usage is not None:
            tokens = (f"вход {usage.get('prompt_tokens', '?')} + выход "
                      f"{usage.get('completion_tokens', '?')} = {usage['total_tokens']} токенов")
            self.console.print(Text(f'  Запрос к модели · {tokens} · 1 вызов {provider}',
                                    style=COLORS['user']))

    def comparison(self, report):
        self.console.rule(Text('РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА', style='bold ' + COLORS['answer']))
        calls = (f"вызовов Codex CLI {report['codex_calls']}"
                 if report.get('mode') == 'codex' else 'API-вызовов 0')
        self.console.print(Text(f"Режим · {report['model']} · {calls}", style='dim'))
        table = Table(box=box.SIMPLE_HEAVY, title='Переходы и восстановление состояния')
        for title in ('Проверка', 'Итог'):
            table.add_column(title, header_style='bold ' + COLORS['user'])
        for check in report['checks']:
            style = COLORS['answer' if check['ok'] else 'error']
            table.add_row(Text(check['name']), Text('✓' if check['ok'] else '×', style=style))
        self.console.print(table)
        self.console.print(Text(f"Пройдено {report['passed']}/{len(report['checks'])}",
                                style='bold ' + COLORS['answer']))
        if report.get('usage'):
            self.console.print(Text(f"Токенов за прогон · {report['usage']['total_tokens']}",
                                    style=COLORS['user']))
        if report.get('error'):
            self.panel('ПРОГОН ОСТАНОВЛЕН', report['error'], 'error')

    def check(self, check):
        self.panel(check['name'], check['detail'], 'answer' if check['ok'] else 'error')

    def codex_event(self, kind, data):
        if kind == 'input':
            title = 'ВЫ · ФИКСТУРА' if data['source'] == 'fixture' else 'ВЫ'
            self.panel(title, data['prompt'], 'user')
            return
        if data.get('raw_reply'):
            title = ('ФИКСТУРА · СЫРОЙ ОТВЕТ' if data['source'] == 'fixture'
                     else 'TERRA · СЫРОЙ ОТВЕТ')
            self.panel(title, data['raw_reply'], 'tool')
        if data['result'] == 'accepted':
            title = 'TERRA' if data['source'] == 'codex' else 'СОСТОЯНИЕ'
            self.panel(title, data['detail'])
        else:
            self.panel('ОТКАЗ' if data['result'] == 'denied' else 'ОШИБКА',
                       data['detail'], 'error')
        self.usage(data['usage'], 'Codex CLI')
        mark = '✓' if data['ok'] else '×'
        color = COLORS['answer' if data['ok'] else 'error']
        self.console.print(Text(f"  {mark} {data['name']}", style=color))
        self.console.print()


def task_action(agent, command, argument):
    if command == '/validate':
        if not argument:
            return agent.perform('review')
        outcome, _, detail = argument.partition(' ')
        if outcome not in ('pass', 'fail') or not detail.strip():
            raise ValueError('Используйте /validate pass|fail <результат>.')
        return agent.perform('validate', detail, passed=outcome == 'pass')
    action = TASK_ACTIONS[command]
    if argument and action not in ('plan', 'execute', 'revise'):
        raise ValueError(f'Команда {command} выполняется без аргументов.')
    return agent.perform(action, argument)


def memory_action(agent, terminal, command, argument):
    memory = agent.memory
    if command == '/layers':
        agent.context()
        terminal.panel('СЛОИ ПАМЯТИ',
                       f'short-term · {len(memory.short_term)} сообщений, окно {memory.window}\n'
                       f'working · {len(memory.working)} полей сохранённой задачи\n'
                       f'long-term · {len(memory.long_term)} записей на диске', 'tool')
    elif command == '/short':
        terminal.panel('SHORT-TERM', dump_json(memory.short_term[-memory.window:]), 'tool')
    elif command == '/working':
        agent.context()
        terminal.panel('WORKING', dump_json(memory.working), 'tool')
    elif command == '/long':
        terminal.panel('LONG-TERM', dump_json(memory.long_term), 'tool')
    elif command == '/clear':
        if argument not in ('short', 'long'):
            raise ValueError('Укажите слой через /clear short|long.')
        memory.clear(argument)
        terminal.panel('СЛОЙ ОЧИЩЕН', argument, 'tool')
    elif command == '/remember':
        key = memory.remember(argument)
        terminal.panel('ЗАПОМНЕНО', f'Ключ · {key}', 'tool')
    elif command == '/forget':
        memory.forget(argument)
        terminal.panel('УДАЛЕНО', argument, 'tool')
    elif command == '/context':
        terminal.panel('КОНТЕКСТ ЗАПРОСА', dump_json(agent.context()), 'tool')
    elif command == '/history':
        terminal.panel('ИСТОРИЯ ДИАЛОГА', dump_json(memory.short_term), 'user')
    elif command == '/window':
        if not argument.isdigit() or int(argument) < 1:
            raise ValueError('Укажите положительный размер окна через /window N.')
        memory.window = int(argument)
        terminal.panel('ОКНО', f'{memory.window} сообщений', 'tool')
    else:
        raise ValueError('Неизвестная команда. Список доступен через /help.')


def chat(agent, terminal, provider):
    terminal.panel('ДЕНЬ 13 · СОСТОЯНИЕ ЗАДАЧИ',
                   f'Режим · {agent.client.model_name}\n'
                   'Синий — вы · зелёный — агент · жёлтый — состояние · красный — ошибка\n\n'
                   + HELP)
    while True:
        try:
            text = terminal.console.input(
                f'[bold {COLORS["user"]}]окно {agent.memory.window} · Вы › [/]').strip()
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
            elif cmd == '/task':
                terminal.panel('НОВАЯ ЗАДАЧА', agent.status(agent.start(arg)), 'tool')
            elif cmd == '/state':
                terminal.panel('СОСТОЯНИЕ ЗАДАЧИ', agent.status(agent.state()), 'tool')
            elif cmd == '/reset':
                agent.reset()
                terminal.panel('НОВЫЙ ДИАЛОГ', 'Короткая история очищена. Продолжение через /state.', 'tool')
            elif cmd in TASK_ACTIONS or cmd == '/validate':
                terminal.panel('ВЫ', text, 'user')
                with terminal.waiting(provider):
                    state, answer, usage = task_action(agent, cmd, arg)
                if cmd == '/validate' and not arg:
                    terminal.panel('АГЕНТ · ЧЕК-ЛИСТ ПРОВЕРКИ', answer)
                else:
                    terminal.panel('СОСТОЯНИЕ ЗАДАЧИ', agent.status(state), 'tool')
                terminal.usage(usage, provider)
            elif cmd.startswith('/'):
                memory_action(agent, terminal, cmd, arg)
            else:
                terminal.panel('ВЫ', text, 'user')
                with terminal.waiting(provider):
                    answer, usage = agent.ask(text)
                terminal.panel('АГЕНТ', answer)
                terminal.usage(usage, provider)
        except ModelReplyError as error:
            terminal.panel('АГЕНТ · СЫРОЙ ОТВЕТ', error.raw, 'tool')
            terminal.panel('ОТКАЗ', str(error), 'error')
            terminal.usage(error.usage, provider)
        except (ClientError, ValueError, OSError) as error:
            terminal.panel('ОШИБКА', str(error), 'error')


def choose_mode(terminal):
    terminal.panel('ДЕНЬ 13 · СОСТОЯНИЕ ЗАДАЧИ',
                   'planning → execution → validation → done\n\n'
                   '1 · Эксперимент с конечным автоматом\n'
                   '2 · Ручной чат\n'
                   '3 · Сохранённые результаты\n0 · Выход')
    while True:
        choice = terminal.console.input(f'[bold {COLORS["user"]}]Выберите режим › [/]').strip()
        if choice in ('0', '1', '2', '3'):
            return choice
        terminal.panel('ОШИБКА', 'Введите 0, 1, 2 или 3.', 'error')


def main():
    parser = argparse.ArgumentParser(description='День 13 · состояние задачи')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--demo', action='store_true', help='Эксперимент')
    modes.add_argument('--chat', action='store_true', help='Ручной чат')
    modes.add_argument('--show-results', type=Path, help='Показать сохранённый JSON-отчёт')
    providers = parser.add_mutually_exclusive_group()
    providers.add_argument('--live', action='store_true', help='Включить API в интерактивном чате')
    providers.add_argument('--codex', action='store_true', help='Использовать вход ChatGPT в Codex CLI')
    parser.add_argument('--model', help='Модель для --live или --codex')
    parser.add_argument('--window', type=int, default=4, help='Размер окна сообщений short-term')
    parser.add_argument('--output-dir', type=Path, help='Новая папка результатов эксперимента')
    parser.add_argument('--state-file', type=Path, default=STATE_PATH, help='Файл состояния чата')
    parser.add_argument('--verbose', action='store_true', help='Показать все проверки эксперимента')
    args = parser.parse_args()
    if args.window < 1:
        parser.error('--window должен быть ≥ 1')
    if args.live and not args.chat:
        parser.error('--live используется с --chat')
    if args.model and not (args.live or args.codex):
        parser.error('--model используется с --live или --codex')
    if args.codex and args.show_results:
        parser.error('--codex используется с --demo или --chat')
    if args.output_dir and (args.chat or args.show_results):
        parser.error('--output-dir используется для эксперимента')

    terminal = Terminal()
    try:
        if not any((args.demo, args.chat, args.show_results)):
            choice = choose_mode(terminal)
            if choice == '0':
                return 0
            args.demo, args.chat = choice == '1', choice == '2'
            if choice == '3':
                args.show_results = REPORT_PATH
        if args.show_results:
            report = load_report(args.show_results)
            terminal.panel('СОХРАНЁННЫЙ ПРОГОН', str(args.show_results), 'user')
            terminal.comparison(report)
            return int(bool(report.get('error')) or report['passed'] != len(report['checks']))
        if args.codex:
            client, provider = CodexClient(args.model), 'Codex CLI'
        elif args.live:
            settings = load_settings(ENV_PATH)
            if args.model:
                settings['LLM_MODEL'] = args.model
            client, provider = ChatClient(settings), 'API'
        else:
            client, provider = OfflineClient(), 'локальный режим'
        if args.chat:
            return chat(Agent(client, MemoryStore(DATA_PATH, args.window), StateStore(args.state_file)),
                        terminal, provider)
        directory = args.output_dir or ROOT / 'results' / datetime.now().strftime('local-%Y%m%d-%H%M%S-%f')
        if args.codex:
            report = run_codex_experiment(directory, DATA_PATH, client, args.window,
                                          terminal.codex_event)
        else:
            report = run_experiment(directory, DATA_PATH, args.window,
                                    terminal.check if args.verbose else None)
        terminal.panel('ОТЧЁТ СОХРАНЁН', str(directory), 'tool')
        terminal.comparison(report)
        return int(bool(report.get('error')) or report['passed'] != len(report['checks']))
    except (ValueError, OSError) as error:
        terminal.panel('ОШИБКА', str(error), 'error')
        return 1
    except EOFError:
        return 0
    except KeyboardInterrupt:
        return 130
