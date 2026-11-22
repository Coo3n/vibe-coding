"""Rich-консоль лабораторной 15, с меню и панелями Lab11."""

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

from .agent import Agent, OfflineClient, RECORD_LABELS
from .codex_client import CodexClient
from .experiment import run_experiment
from .memory import MemoryStore
from .report import initial_memory_sections, load_report


COLORS = {'user': '#92b7ff', 'answer': '#76ddbd', 'tool': '#e7ba76', 'error': '#ff8a96'}
ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / '.env'
DATA_PATH = ROOT / 'data' / 'memory.json'
SEED_PATH = ROOT / 'data' / 'long_term_seed.json'
REPORT_PATH = ROOT / 'results' / 'experiment-1' / 'report.json'
HELP = ('/layers · /short · /working · /long · /log\n'
        '/remember ключ: значение · /forget ключ · /note ключ: значение · /done\n'
        '/clear short|working|long · /context · /history · /window N · /reset · /help · /exit')


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
    def waiting(self, title='Агент обрабатывает запрос'):
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

    def usage(self, usage, source=None):
        if usage:
            self.console.print(Text(
                f"  Запрос к модели · вход {usage.get('prompt_tokens', '?')} + "
                f"выход {usage.get('completion_tokens', '?')} = {usage['total_tokens']} токенов",
                style=COLORS['user']))
        elif source == 'memory':
            self.console.print(Text('  Операция памяти · без обращения к модели', style=COLORS['tool']))
        else:
            self.console.print(Text('  Ответ локального клиента', style=COLORS['tool']))

    def event(self, kind, turn, verbose=False):
        if kind == 'initial_memory':
            sections = ('\n'.join([title, *(f'  {entry}' for entry in entries)])
                        for title, entries in initial_memory_sections(turn))
            self.panel('ИСХОДНЫЕ ДАННЫЕ В ПАМЯТИ', '\n\n'.join(sections), 'tool')
        elif kind == 'prompt':
            self.panel(f'ВЫ · {turn["phase"]}', turn['input'], 'user')
        else:
            source = turn['source']
            self.panel('ПАМЯТЬ' if source == 'memory' else 'АГЕНТ', turn['answer'],
                       'tool' if source == 'memory' else 'answer')
            self.usage(turn['usage'], source)
            if verbose and turn['messages']:
                self.panel('КОНТЕКСТ ЗАПРОСА', dump_json(turn['messages']), 'tool')
        self.console.file.flush()

    def comparison(self, report, verbose=False, exchanges_shown=False):
        self.console.rule(Text('РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА', style='bold ' + COLORS['answer']))
        self.console.print(Text(f"Клиент · {report['model']} | Окно · {report['window']}", style='dim'))
        table = Table(box=box.SIMPLE_HEAVY, title='Проверка актуальных значений')
        for name in ('Факт', 'Актуальное значение', 'Проверка' if exchanges_shown else 'Ответ'):
            table.add_column(name, header_style=COLORS['user'])
        for step in report['steps']:
            passed = (step['expected_in_answer'] and not step['stale_in_answer']
                      and not step['stale_in_prompt'])
            result = ('пройдена' if passed else 'не пройдена') if exchanges_shown else step['answer']
            table.add_row(Text(step['key']), Text(step['expected']), Text(result))
            if verbose:
                self.panel('ВЫ · ИЗМЕНЕНИЕ', step['statement'], 'user')
                self.panel('ВЫ · КОНТРОЛЬНЫЙ ВОПРОС', step['question'], 'user')
                self.panel('КОНТЕКСТ ЗАПРОСА', dump_json(step['messages']), 'tool')
        self.console.print(table)
        checks = report['checks']
        usage = report.get('usage')
        spending = ''
        if usage:
            spending = f"\nОпераций с памятью: {usage['memory_actions']}"
            if usage['model_calls']:
                spending += (f"\nВызовов модели: {usage['model_calls']} · "
                             f"{usage['prompt_tokens']:,} вход + {usage['completion_tokens']:,} выход "
                             f"= {usage['total_tokens']:,} токенов")
            else:
                spending += f"\nОтветов локального клиента: {usage['offline_answers']}"
        self.panel('ИТОГ',
                   f"Актуальных ответов: {checks['correct_answers']}/{len(report['steps'])}\n"
                   f"Устаревших ответов: {checks['stale_answers']}\n"
                   f"Промптов с устаревшими значениями: {checks['stale_prompts']}\n"
                   f"Замен в журнале: {checks['replacements']}\n"
                   f"Рабочая память: {checks['working_before_finish']} → {checks['working_after_finish']}\n"
                   f"Долговременная память сохранена: "
                   f"{'да' if checks['durable_unchanged'] else 'нет'}{spending}", 'tool')


def _fact(text):
    key, separator, value = text.partition(':')
    if not separator or not key.strip() or not value.strip():
        raise ValueError('Укажите ключ и значение через двоеточие.')
    return key.strip(), value.strip()


def chat(client, terminal, window, path, seed):
    memory = MemoryStore(path, window, seed)
    agent = Agent(client, memory)
    terminal.panel('ДЕНЬ 15 · ОБНОВЛЯЕМ ПАМЯТЬ',
                   f'Окно {window} сообщений · клиент {client.model_name}\n'
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
                active = memory.durable.read()['active']
                terminal.panel('СЛОИ ПАМЯТИ',
                               f'short-term: {len(memory.short_term)} сообщений (окно {memory.window})\n'
                               f'working: {len(memory.working)} ключей задачи\n'
                               f'long-term: {len(active)} актуальных записей · {path}', 'tool')
            elif cmd in ('/short', '/history'):
                items = memory.short_term[-memory.window:] if cmd == '/short' else memory.short_term
                terminal.panel('ИСТОРИЯ ДИАЛОГА', dump_json(items), 'tool')
            elif cmd == '/working':
                terminal.panel('WORKING', dump_json(memory.working), 'tool')
            elif cmd == '/long':
                state = memory.durable.read()
                terminal.panel('LONG-TERM', dump_json({'active': state['active'],
                                                        'history': state['history']}), 'tool')
            elif cmd == '/log':
                terminal.panel('ЖУРНАЛ ЗАМЕН', dump_json(memory.durable.read()['changes']), 'tool')
            elif cmd == '/remember':
                outcome = memory.durable.record(*_fact(arg))
                terminal.panel('LONG-TERM', RECORD_LABELS[outcome], 'tool')
            elif cmd == '/forget':
                outcome = memory.durable.forget(arg)
                terminal.panel('LONG-TERM', RECORD_LABELS[outcome], 'tool')
            elif cmd == '/note':
                memory.record_working(*_fact(arg))
                terminal.panel('WORKING', 'Значение записано до завершения задачи.', 'tool')
            elif cmd == '/done':
                memory.finish_task()
                terminal.panel('ЗАДАЧА ЗАВЕРШЕНА',
                               'Рабочая память очищена. Долговременные факты сохранены.', 'tool')
            elif cmd == '/clear':
                memory.clear(arg)
                terminal.panel('СЛОЙ ОЧИЩЕН', arg, 'tool')
            elif cmd == '/reset':
                agent.reset()
                terminal.panel('НОВЫЙ ДИАЛОГ', 'Рабочая память и диалог очищены.', 'tool')
            elif cmd == '/context':
                terminal.panel('КОНТЕКСТ ЗАПРОСА', dump_json(memory.context(agent.system_prompt)), 'tool')
            elif cmd == '/window':
                if not arg.isdecimal() or int(arg) < 1:
                    raise ValueError('Нужно число ≥ 1, например /window 6.')
                memory.window = int(arg)
                terminal.panel('ОКНО', f'{memory.window} сообщений.', 'tool')
            elif cmd.startswith('/'):
                raise ValueError('Неизвестная команда. Список доступен через /help.')
            else:
                terminal.panel('ВЫ', text, 'user')
                with terminal.waiting():
                    reply = agent.ask(text)
                terminal.panel('АГЕНТ', reply['answer'])
                terminal.usage(reply['usage'], 'memory' if reply.get('update') else None)
        except (OSError, RuntimeError, ValueError, OverflowError) as error:
            terminal.panel('ОШИБКА', str(error), 'error')


def main():
    parser = argparse.ArgumentParser(description='День 15 · устаревание и конфликты памяти')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--demo', action='store_true', help='Автоматический прогон эксперимента')
    modes.add_argument('--chat', action='store_true', help='Ручной диалог')
    modes.add_argument('--show-results', type=Path, help='Показать сохранённый отчёт без API')
    providers = parser.add_mutually_exclusive_group()
    providers.add_argument('--live', action='store_true', help='Запрашивать ответы через API')
    providers.add_argument('--codex', action='store_true', help='Использовать вход ChatGPT в Codex CLI')
    parser.add_argument('--model', help='Имя модели для режима --live или --codex')
    parser.add_argument('--window', type=int, default=4, help='Размер окна short-term')
    parser.add_argument('--output-dir', type=Path, help='Новая папка результатов эксперимента')
    parser.add_argument('--memory-file', type=Path, help='Отдельное JSON-хранилище для ручного диалога')
    parser.add_argument('--verbose', action='store_true', help='Показать полный запрос эксперимента')
    args = parser.parse_args()
    if args.window < 1:
        parser.error('--window должен быть ≥ 1')
    if args.model and not (args.live or args.codex):
        parser.error('--model используется вместе с --live или --codex')
    if args.show_results and (args.live or args.codex):
        parser.error('--show-results читает готовый отчёт')
    terminal = Terminal()
    try:
        if not (args.demo or args.chat or args.show_results):
            terminal.panel('ДЕНЬ 15 · УСТАРЕВАНИЕ И КОНФЛИКТЫ ПАМЯТИ',
                           '1 · Эксперимент: обновления, журнал и очистка\n'
                           '2 · Ручной чат со слоями памяти\n'
                           '3 · Сохранённые результаты, без API\n0 · Выход')
            while True:
                choice = terminal.console.input('[bold #92b7ff]Режим › [/]').strip()
                if choice in ('0', '1', '2', '3'):
                    break
                terminal.console.print('Введите число от 0 до 3.', style=COLORS['error'])
            if choice == '0':
                return 0
            args.demo, args.chat = choice == '1', choice == '2'
            if choice == '3':
                args.show_results = REPORT_PATH
        if args.output_dir and not args.demo:
            parser.error('--output-dir используется только для эксперимента')
        if args.memory_file and not args.chat:
            parser.error('--memory-file используется только для ручного чата')
        if args.show_results:
            report = load_report(args.show_results)
            terminal.panel('СОХРАНЁННЫЙ ПРОГОН · БЕЗ API', str(args.show_results), 'user')
            if 'initial_memory' in report:
                terminal.event('initial_memory', report['initial_memory'])
            terminal.comparison(report, args.verbose)
            return 0 if report['status'] == 'completed' else 1
        if args.live:
            from .client import ChatClient, load_settings
            settings = load_settings(ENV_PATH)
            if args.model:
                settings['LLM_MODEL'] = args.model
            client = ChatClient(settings)
        elif args.codex:
            client = CodexClient(args.model)
        else:
            client = OfflineClient()
        if args.chat:
            return chat(client, terminal, args.window, args.memory_file or DATA_PATH,
                        None if args.memory_file else SEED_PATH)
        directory = args.output_dir or ROOT / 'results' / (
            'local-day15-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
        terminal.panel('ДЕНЬ 15 · ПРОВЕРКА АКТУАЛЬНЫХ ФАКТОВ',
                       f'{client.model_name} · четыре изменения · время записи и завершение задачи',
                       'user')
        with terminal.waiting('Эксперимент · четыре изменения и завершение задачи'):
            report = run_experiment(client, directory, args.window,
                                    lambda kind, turn: terminal.event(kind, turn, args.verbose))
        terminal.panel('ОТЧЁТЫ', f'{directory / "report.md"}\n{directory / "report.json"}')
        terminal.comparison(report, exchanges_shown=True)
        return 0 if report['status'] == 'completed' else 1
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        terminal.panel('ОШИБКА', str(error), 'error')
        return 1
    except (KeyboardInterrupt, EOFError):
        terminal.console.print('\nРабота остановлена.', style='dim')
        return 130
