"""Компактная Rich-консоль: меню, диалог с профилями и эксперимент Дня 12."""

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
from .offline import OfflineClient
from .profile import UserProfile, seed_profiles
from .experiment import fraction, run_experiment
from .report import FIELDS, LABELS, PHASE_LABELS, load_report
from .scenario import PROFILE_SEEDS


COLORS = {'user': '#92b7ff', 'answer': '#76ddbd', 'tool': '#e7ba76', 'error': '#ff8a96'}
ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / '.env'
DATA_PATH = ROOT / 'data' / 'long_term.json'
PROFILES_DIR = ROOT / 'data' / 'profiles'
DEFAULT_REPORT = ROOT / 'results' / 'offline-1' / 'report.json'
HELP = ('/profile [имя] · /prefer поле: значение\n'
        '/layers · /short · /working · /long\n'
        '/clear short|working|long · /remember текст · /forget ключ\n'
        '/context · /history · /window N · /reset · /help · /exit')

PHASES = {'base': 'БАЗОВЫЙ ДИАЛОГ', 'question': 'ВОПРОС ПРОФИЛЮ', 'stress': 'СТРЕСС-ТЕСТ',
          'before': 'ДО СМЕНЫ', 'switch': 'СМЕНА ПРЕДПОЧТЕНИЙ', 'after': 'ПОСЛЕ СМЕНЫ'}

CONFIG_HELP = {
    'exp1-opposite':
        'Три профиля с противоположными предпочтениями: «Кратко» (кратко, без кода, '
        'неформально), «Подробно» (подробно, с кодом, формально) и «Микс» (кратко, '
        'с кодом, формально). Каждый отвечает на одни и те же 3 вопроса о поездке. '
        'Перед каждым вопросом восстановлена одинаковая память. '
        'Предпочтения передаются отдельным system-блоком.',
    'exp2-stability':
        'Профиль «Кратко»: 10 вопросов подряд, часть соблазняет нарушить формат — '
        '«покажи кодом», «расскажи подробно», «дай подробную инструкцию». '
        'Фиксируем номер первого ответа, где предпочтение нарушено. Разовые просьбы '
        'профиль менять не должны: извлекатель отличает их от установок на будущее.',
    'exp3-replenish':
        'Диалог в профиле «Кратко»: два вопроса, затем явная смена предпочтений '
        '(«теперь отвечай подробно, показывай код, обращайся на Вы»), затем ещё два '
        'вопроса. Проверяем обновление профиля и загрузку с диска.',
}


def number(value):
    return '?' if value is None else f'{value:,}'.replace(',', ' ')


def clean(value):
    text = str(value).replace('\u00a0', ' ').replace('\u202f', ' ')
    return ''.join(c if c in '\n\t' or c.isprintable() else '�' for c in text)


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

    def usage(self, usage, side_tokens=0, side_calls=0,
              profile_side_tokens=0, profile_side_calls=0, profile_changes=None):
        if usage is None:
            self.console.print(Text('  Офлайн · внешних вызовов API 0', style=COLORS['tool']))
            return
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
        if profile_side_calls:
            suffix = (' · ' + '; '.join(profile_changes)) if profile_changes else ' · без изменений'
            self.console.print(Text(
                f'  Обновление профиля: {number(profile_side_tokens)} токенов · '
                f'дополнительных вызовов: {profile_side_calls}{suffix}',
                style=COLORS['tool']))
        elif profile_changes:
            self.console.print(Text('  Обновление профиля: ' + '; '.join(profile_changes),
                                    style=COLORS['tool']))

    def progress(self, usage, side_tokens, side_calls, running,
                 profile_side_tokens=0, profile_side_calls=0, profile_changes=None,
                 checks=None, answer=None):
        if answer is not None:
            self.panel('АГЕНТ', answer)
        if checks:
            parts = []
            for check in checks:
                mark = '✓ ' if check['ok'] else '! '
                text = f"{check['field']}: {check['value']}"
                if check['field'] == 'тон' or not check['ok']:
                    text += f' ({check["detail"]})'
                parts.append((mark + text, 'bold ' + (COLORS['answer'] if check['ok']
                                                      else COLORS['error'])))
            pieces = []
            for index, (text, style) in enumerate(parts):
                if index:
                    pieces.append((' · ', 'dim'))
                pieces.append((text, style))
            self.console.print(Text.assemble(('  ', ''), *pieces))
        self.usage(usage, side_tokens, side_calls,
                   profile_side_tokens, profile_side_calls, profile_changes)
        if usage is None:
            self.console.print()
            return
        tokens = usage.get('total_tokens')
        step_tokens = None if tokens is None else tokens + side_tokens + profile_side_tokens
        self.console.print(Text.assemble(
            (f'  Расход шага: {number(step_tokens)} токенов', 'bold ' + COLORS['answer']),
            (f' · всего за прогон: {number(running)} токенов', 'bold ' + COLORS['tool'])))
        self.console.print()

    def comparison(self, report):
        runs = {run['name']: run for run in report.get('runs') or []}
        measure = report.get('profile_measure')
        self.console.rule(Text('РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА', style='bold ' + COLORS['answer']))
        self.console.print(Text(f"Модель: {report['model']} · окно: {report['window']} сообщений",
                                style='dim'))
        if report.get('mode') == 'offline':
            self.console.print(Text('Офлайн · вызовов API 0 · токены не измерялись', style='dim'))
        elif report.get('mode') == 'codex':
            self.console.print(Text('Codex CLI · токены включают контекст Codex', style='dim'))
        if report.get('tokens') is not None:
            self.console.print(Text(
                f"Расход прогона: {number(report['tokens'])} токенов · "
                f"{report.get('main_calls', '?')} основных и "
                f"{report.get('side_calls', 0) + report.get('profile_side_calls', 0)} служебных "
                f"вызовов (working {number(report.get('side_tokens'))} + "
                f"профиль {number(report.get('profile_side_tokens'))} токенов)",
                style='dim'))
        self.console.print()
        if measure:
            table = Table(box=box.SIMPLE_HEAVY, title='Цена профиля в промпте',
                          title_style='bold ' + COLORS['user'])
            for title in ('Блок', 'На запрос', 'prompt_tokens с блоком', 'без блока'):
                table.add_column(title, justify='left' if title == 'Блок' else 'right',
                                 header_style='bold ' + COLORS['user'],
                                 style=COLORS['tool'] if 'оценка' in title else COLORS['answer'])
            table.add_row(f"Профиль пользователя («{measure['profile']}»)",
                          str(measure['block_tokens']), str(measure['prompt_tokens_with']),
                          str(measure['prompt_tokens_without']))
            self.console.print(table)
            self.console.print(Text('Дельта prompt_tokens на репрезентативном запросе: '
                                    'сколько стоит подключить профиль к каждому запросу.',
                                    style='dim'))
            self.console.print()
        exp1 = runs.get('exp1-opposite')
        if exp1:
            table = Table(box=box.SIMPLE_HEAVY, title='Эксперимент 1 · противоположные профили',
                          title_style='bold ' + COLORS['user'])
            table.add_column('Профиль', justify='left', header_style='bold ' + COLORS['user'],
                             style=COLORS['answer'])
            for title in ('Предпочтения', 'Стиль', 'Формат', 'Тон', 'Ответы'):
                table.add_column(title, justify='left' if title == 'Предпочтения' else 'right',
                                 header_style='bold ' + COLORS['user'],
                                 style=COLORS['tool'] if title == 'Предпочтения' else COLORS['answer'])
            for name, data in exp1['profiles'].items():
                compliance = data['compliance']
                cells = [fraction(*compliance.get(f, (0, 0))) for f in FIELDS]
                answers = data.get('compliant_answers')
                if answers is None:
                    answers = sum(all(check['ok'] for check in q['checks']) for q in data['questions'])
                table.add_row(name, ' · '.join(data['prefs'][f] for f in FIELDS),
                              *cells, f'{answers}/{len(data["questions"])}')
            self.console.print(table)
            self.console.print(Text('Одинаковые вопросы, одинаковая память — разные '
                                    'предпочтения. «с кодом» проверен на 2 из 3 вопросов '
                                    '(код уместен), «без кода» — на всех 3.', style='dim'))
            self.console.print()
        exp2 = runs.get('exp2-stability')
        if exp2:
            table = Table(box=box.SIMPLE_HEAVY,
                          title=f"Эксперимент 2 · устойчивость: 10 вопросов в профиле «{exp2['profile']}»",
                          title_style='bold ' + COLORS['user'])
            for title in ('№', 'Вопрос', 'Стиль', 'Формат', 'Тон'):
                table.add_column(title, justify='left' if title == 'Вопрос' else 'right',
                                 header_style='bold ' + COLORS['user'],
                                 style=COLORS['user'] if title == 'Вопрос' else COLORS['answer'])
            for index, q in enumerate(exp2['questions'], 1):
                marks = {c['field']: ('✓' if c['ok'] else '✗') for c in q['checks']}
                table.add_row(str(index), q['question'],
                              *[marks.get(f, '—') for f in FIELDS])
            self.console.print(table)
            first = exp2.get('first_violation')
            if first:
                self.console.print(Text(
                    f"Первое нарушение — вопрос №{first['index']}: {', '.join(first['fields'])}.",
                    style='bold ' + COLORS['error']))
            else:
                self.console.print(Text('10 из 10 ответов прошли проверку. Первого нарушения нет.',
                                        style='bold ' + COLORS['answer']))
            self.console.print()
        exp3 = runs.get('exp3-replenish')
        if exp3:
            table = Table(box=box.SIMPLE_HEAVY, title='Эксперимент 3 · пополнение из диалога',
                          title_style='bold ' + COLORS['user'])
            table.add_column('Этап', justify='left', header_style='bold ' + COLORS['user'],
                             style=COLORS['answer'])
            for title in ('Стиль', 'Формат', 'Тон'):
                table.add_column(title, justify='right', header_style='bold ' + COLORS['user'],
                                 style=COLORS['answer'])
            for stage, prefs, compliance in (('до смены', exp3['before_prefs'], exp3['before_compliance']),
                                             ('после смены', exp3['after_prefs'], exp3['after_compliance'])):
                table.add_row(stage, *[f"{prefs[f]} · {fraction(*compliance.get(f, (0, 0)))}"
                                       for f in FIELDS])
            self.console.print(table)
            changes = '\n'.join(exp3['changes']) if exp3['changes'] else 'нет'
            self.console.print(Text(f'Смена: «{exp3["switch_message"]}»\n'
                                    f'Изменения\n{changes}', style='dim'))
            if exp3.get('persisted') is not None:
                result = 'профиль загружен с диска' if exp3['persisted'] else 'профиль отличается от сохранённого'
                self.console.print(Text(f'Проверка сохранения · {result}', style=COLORS['tool']))
            self.console.print()

    def event(self, event):
        kind = event['kind']
        if kind == 'seed':
            self.panel('ДОЛГОВРЕМЕННАЯ ПАМЯТЬ',
                       f'Факты Дня 10 в учебном снимке · {event["count"]} записей.', 'tool')
            return
        if kind == 'profiles':
            items = '\n'.join(f'· {item}' for item in event['items'])
            self.panel('ПРОФИЛИ ПОЛЬЗОВАТЕЛЕЙ',
                       f'Учебные профили · {event["count"]}\n{items}', 'tool')
            return
        if kind == 'measure':
            self.panel('ЗАМЕР ТОКЕНОВ',
                       f'Репрезентативный запрос с блоком профиля и без него отправлен.\n'
                       f'Дельта prompt_tokens = {event["block_tokens"]} токенов — цена '
                       'подключения профиля к каждому запросу.', 'tool')
            self.usage(event['usage'])
            return
        if kind == 'case_start':
            self.current_case = event['name']
            self.console.rule(Text('ТЕСТ · ' + LABELS[event['name']].upper(),
                                   style='bold ' + COLORS['user']))
            self.panel('КАК ПРОХОДИТ ПРОВЕРКА', CONFIG_HELP[event['name']], 'tool')
            return
        if kind == 'prompt':
            phase = PHASES[event['phase']]
            case = event.get('profile') or 'Диалог'
            title = f'{case} · {phase} {event["index"]:02}/{event["total"]} · ВЫ'
            self.panel(title, event['prompt'], 'user')
            return
        self.progress(event['usage'], event['side_tokens'], event['side_calls'],
                      event['running'], event['profile_side_tokens'],
                      event['profile_side_calls'], event['profile_changes'],
                      checks=event['checks'],
                      answer=event['answer'])


def show_layer(terminal, name, title, value, window=4):
    if name == 'short':
        tail = value[-window:]
        text = (f'{len(value)} сообщений текущего диалога (в окно уходит {len(tail)}):\n\n'
                + (dump_json([{'role': m["role"], 'content': m["content"]} for m in tail])
                   if tail else '(пусто)'))
    else:
        text = dump_json(value) if value else '(пусто)'
    terminal.panel(title, text, 'tool')


def load_profile_by_name(name, profiles_dir):
    """Профиль по имени или имени файла: «Кратко», «кратко», «short»."""
    wanted = name.strip().lower()
    for path in sorted(profiles_dir.glob('*.json')):
        if path.stem.lower() == wanted:
            return UserProfile.load(path)
    for path in sorted(profiles_dir.glob('*.json')):
        profile = UserProfile.load(path)
        if profile.name.lower() == wanted:
            return profile
    raise ValueError(f'Профиля «{name}» нет. Доступны: '
                     f'{", ".join(p.stem for p in sorted(profiles_dir.glob("*.json"))) or "—"}')


def chat(client, terminal, window=4, profiles_dir=PROFILES_DIR, memory_file=DATA_PATH):
    memory = MemoryStore(memory_file, window)
    profiles = seed_profiles(profiles_dir, PROFILE_SEEDS, overwrite=False)
    profile = profiles['Кратко']
    agent = Agent(client, memory, profile)
    terminal.panel('ДЕНЬ 12 · ПЛАНИРУЕМ ЯПОНИЮ',
                   f'{client.model_name} · окно {window} сообщений · '
                   f'long-term: {len(memory.long_term)} записей\n'
                   f'Профиль: {profile.name} — {profile.describe()}\n'
                   'Синий — вы · зелёный — агент · жёлтый — память и профиль · красный — ошибка\n\n' + HELP)
    while True:
        try:
            text = terminal.console.input(
                f'[bold {COLORS["user"]}]окно {memory.window} · профиль {agent.profile.name} · Вы › [/]').strip()
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
            elif cmd == '/profile':
                if arg:
                    selected = load_profile_by_name(arg, profiles_dir)
                    if selected.path != agent.profile.path:
                        agent.reset()
                    agent.profile = selected
                    terminal.panel('ПРОФИЛЬ ПЕРЕКЛЮЧЁН',
                                   f'{agent.profile.name} — {agent.profile.describe()}', 'tool')
                else:
                    terminal.panel('АКТИВНЫЙ ПРОФИЛЬ',
                                   f'{agent.profile.name} — {agent.profile.describe()}\n'
                                   'Доступны: '
                                   + ', '.join(p.stem for p in sorted(profiles_dir.glob('*.json')))
                                   + '\nПереключение: /profile имя', 'tool')
            elif cmd == '/prefer':
                if agent.profile is None:
                    raise ValueError('Сначала выберите профиль: /profile имя')
                field, _, value = arg.partition(':')
                if not value.strip():
                    raise ValueError('Формат: /prefer поле: значение, например /prefer стиль: подробно')
                change = agent.profile.set(field, value)
                terminal.panel('ПРОФИЛЬ ИЗМЕНЁН', f'{change}\n{agent.profile.describe()}', 'tool')
            elif cmd == '/layers':
                terminal.panel('СЛОИ ПАМЯТИ И ПРОФИЛЬ',
                               f'short-term: {len(memory.short_term)} сообщений (окно {memory.window})\n'
                               f'working: {len(memory.working)} ключей данных задачи\n'
                               f'long-term: {len(memory.long_term)} записей · файл '
                               f'{memory_file}\n'
                               f'профиль: {agent.profile.name if agent.profile else "—"} · '
                               f'{profiles_dir}', 'tool')
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
                               dump_json(memory.context(agent.system_prompt,
                                                        profile_block=agent.profile.block()
                                                        if agent.profile else None)), 'tool')
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
                               f'({len(memory.long_term)} записей) и профиль '
                               f'«{agent.profile.name}» пережили сброс.', 'tool')
            elif cmd.startswith('/'):
                raise ValueError('Неизвестная команда. Список: /help')
            else:
                terminal.panel('ВЫ', text, 'user')
                with terminal.waiting({'offline': 'Агент работает', 'live': 'Агент работает · запрос к API',
                                       'codex': 'Агент работает · Codex CLI'}[client.mode]):
                    reply = agent.ask(text)
                terminal.panel('АГЕНТ', reply['answer'])
                if reply['profile_changes']:
                    terminal.panel('ПРОФИЛЬ ОБНОВЛЁН ИЗ ДИАЛОГА',
                                   '\n'.join(reply['profile_changes'])
                                   + f'\nСохранено в файл профиля: {agent.profile.path}',
                                   'tool')
                terminal.usage(reply['usage'], reply['side_tokens'], reply['side_calls'],
                               reply['profile_side_tokens'], reply['profile_side_calls'],
                               reply['profile_changes'])
        except (ClientError, ValueError) as error:
            terminal.panel('ОШИБКА', str(error), 'error')


def main():
    parser = argparse.ArgumentParser(description='День 12 · персонализация ассистента')
    providers = parser.add_mutually_exclusive_group()
    providers.add_argument('--live', action='store_true', help='Использовать API')
    providers.add_argument('--codex', action='store_true', help='Использовать вход ChatGPT в Codex CLI')
    parser.add_argument('--model', help='Модель для --live или --codex')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--demo', action='store_true', help='Автоматический прогон эксперимента')
    modes.add_argument('--chat', action='store_true', help='Ручной диалог с профилями')
    modes.add_argument('--show-results', type=Path, help='Показать сохранённый отчёт без API')
    parser.add_argument('--window', type=int, default=4,
                        help='Число последних сообщений short-term в промпте')
    parser.add_argument('--output-dir', type=Path, help='Новая папка результатов эксперимента')
    parser.add_argument('--profiles-dir', type=Path, help='Каталог профилей интерактивного чата')
    parser.add_argument('--memory-file', type=Path, help='Файл долговременной памяти чата')
    parser.add_argument('--verbose', action='store_true',
                        help='Показать реплики эксперимента (в режиме --codex включено по умолчанию)')
    args = parser.parse_args()
    if args.window < 1:
        parser.error('--window должен быть ≥ 1')
    if args.output_dir and (args.chat or args.show_results):
        parser.error('--output-dir используется только для эксперимента')
    if (args.profiles_dir or args.memory_file) and not args.chat:
        parser.error('--profiles-dir и --memory-file используются с --chat')
    if args.model and not (args.live or args.codex):
        parser.error('--model используется с --live или --codex')
    if (args.live or args.codex) and args.show_results:
        parser.error('--show-results читает готовый отчёт')
    provider_label = 'Codex CLI' if args.codex else ('API' if args.live else 'офлайн')
    terminal = Terminal()
    try:
        if args.show_results:
            report = load_report(args.show_results)
            terminal.panel('СОХРАНЁННЫЙ ПРОГОН · БЕЗ API', str(args.show_results), 'user')
            terminal.comparison(report)
            if report.get('error'):
                terminal.panel('ПРОГОН НЕ ЗАВЕРШЁН', report['error'], 'error')
            return 0
        if not args.demo and not args.chat:
            terminal.panel('ДЕНЬ 12 · ПЕРСОНАЛИЗАЦИЯ',
                           f'Япония · 3 профиля · {provider_label}\n\n'
                           '1 · Эксперимент: профили, соблюдение, стресс-тест\n'
                           '2 · Ручной чат с профилями и памятью\n'
                           '3 · Сохранённые результаты, без API\n0 · Выход')
            while True:
                choice = terminal.console.input('[bold #92b7ff]Режим › [/]').strip()
                if choice in ('0', '1', '2', '3'):
                    break
                terminal.console.print('Введите число от 0 до 3.', style=COLORS['error'])
            if choice == '0':
                return 0
            if choice == '3':
                report = load_report(DEFAULT_REPORT)
                terminal.panel('СОХРАНЁННЫЙ ПРОГОН', 'Модель ' + report['model'] +
                               ' · профили и соблюдение предпочтений, без новых запросов API.', 'user')
                terminal.comparison(report)
                return 0
            args.chat = choice == '2'
            args.demo = not args.chat
        if args.live:
            settings = load_settings(ENV_PATH)
            if args.model:
                settings['LLM_MODEL'] = args.model
            client = ChatClient(settings)
        elif args.codex:
            client = CodexClient(args.model)
            terminal.call_label = 'ход Codex'
        else:
            client = OfflineClient()
        if args.chat:
            return chat(client, terminal, args.window, args.profiles_dir or PROFILES_DIR,
                        args.memory_file or DATA_PATH)
        directory = args.output_dir or ROOT / 'results' / ('local-day12-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
        terminal.panel('ДЕНЬ 12 · ТЕСТ ПЕРСОНАЛИЗАЦИИ',
                       f'{client.model_name} · окно {args.window} сообщений\n'
                       '3 профиля × 3 общих вопроса · 10 вопросов подряд · обновление профиля из диалога',
                       'user')
        with terminal.waiting(f'Эксперимент · {provider_label}'):
            on_event = terminal.event if args.codex or args.verbose else None
            report = run_experiment(client, directory, args.window, on_event)
        terminal.panel('ОТЧЁТЫ', f'{directory / "report.md"}\n{directory / "report.json"}')
        terminal.comparison(report)
        if report.get('error'):
            terminal.panel('ПРОГОН ОСТАНОВЛЕН', report['error'] + '\nНезавершённые эксперименты не включены в итог.', 'error')
        return 0 if report['status'] == 'completed' else 1
    except (ClientError, OSError, ValueError, KeyError, TypeError) as error:
        terminal.panel('ОШИБКА', str(error), 'error')
        return 1
    except (KeyboardInterrupt, EOFError):
        terminal.console.print('\nРабота остановлена.', style='dim')
        return 130
