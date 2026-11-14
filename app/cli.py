"""Консоль task-6: диалог, инструменты и наглядное восстановление контекста."""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import termios
from datetime import datetime

from rich import box
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, ProgressColumn, SpinnerColumn, TextColumn
from rich.text import Text
from rich.table import Table

from .agent import Agent, AgentError, SYSTEM_PROMPT, OPENING_QUESTION
from .client import ChatClient, ClientError, load_settings
from .history import HistoryStore, HistoryError
from .tools import TravelTools
from .response import answer_text


ROOT = Path(__file__).resolve().parent.parent
COLORS = {'user': '#92b7ff', 'answer': '#76ddbd', 'tool': '#e7ba76', 'error': '#ff8a96'}


class WaitColumn(ProgressColumn):
    def render(self, task):
        elapsed = task.elapsed or 0
        fraction = min(elapsed / task.total, 1)
        filled = int(fraction * 16)
        return Text.assemble(('━' * filled, '#76ddbd'), ('─' * (16 - filled), '#344258'),
                             (f' {fraction:.0%} от {task.total:g}с · прошло {elapsed:.0f}с', 'dim'))


class Terminal:
    def __init__(self):
        self.console = Console(force_terminal=True, color_system='truecolor',
                               force_interactive=sys.stdout.isatty())
        self.console.width = min(self.console.width, 100)

    def panel(self, title, text, kind='user', markdown=False, sections=()):
        def clean(value):
            return ''.join(c if c in '\n\t' or c.isprintable() else '�' for c in str(value))

        content = [Markdown(clean(text)) if markdown else Text(clean(text))] if text else []
        for heading, body in sections:
            content.append(Panel(Text(clean(body)), title=Text(clean(heading), style='bold'),
                                 title_align='left', box=box.ROUNDED, padding=(0, 1),
                                 border_style=COLORS[kind], style='#e7edf6 on #172438'))
        self.console.print(Panel(Group(*content), title=Text(clean(title), style='bold'), title_align='left',
                                 box=box.ROUNDED, padding=(0, 2), border_style=COLORS[kind],
                                 style='#e7edf6 on #121c2b'))

    def event(self, event, historical=False):
        result = event['result']
        source = 'ИЗ ИСТОРИИ — не новый вызов' if historical else f"Модель вызвала {event['name']}"
        parameters = format_parameters(event['arguments'])
        if 'error' in result:
            self.panel('ОШИБКА ИНСТРУМЕНТА', source, 'error',
                       sections=[('ПАРАМЕТРЫ', parameters), ('РЕЗУЛЬТАТ', str(result['error']))])
            return
        if event['name'] == 'respond_to_user':
            return  # Ответ показывается после успешного сохранения хода, а не как выдача поиска.
        tours = result.get('tours')
        if isinstance(tours, list) and all(isinstance(t, dict) for t in tours):
            details = '\n\n'.join(format_tour(t) for t in tours) if tours else 'По заданным условиям путёвок нет.'
            details += '\n\nДанные и цены учебные, не предложение к покупке.'
        else:
            details = json.dumps(result, ensure_ascii=False, indent=2)
        self.panel('ИНСТРУМЕНТ · ПОИСК ПУТЁВОК', source, 'tool',
                   sections=[('ПАРАМЕТРЫ', parameters),
                             ('РЕЗУЛЬТАТ', details)])

    def context(self, agent, restored=False):
        fingerprint = hashlib.sha256(json.dumps(agent.messages, ensure_ascii=False, sort_keys=True,
                                                separators=(',', ':')).encode('utf-8')).hexdigest()[:12]
        calls = sum(call['id'] != 'local-opening-question'
                    for m in agent.messages for call in m.get('tool_calls') or [])
        users = sum(m['role'] == 'user' for m in agent.messages)
        accepted = {m['tool_call_id'] for m in agent.messages if m['role'] == 'tool'
                    and json.loads(m['content']).get('status') == 'accepted'}
        answers = sum(call['id'] in accepted and call['function']['name'] == 'respond_to_user'
                       for m in agent.messages for call in m.get('tool_calls', []))
        answers += sum(m['role'] == 'assistant' and not m.get('tool_calls') for m in agent.messages)
        systems = sum(m['role'] == 'system' for m in agent.messages)
        assistants = sum(m['role'] == 'assistant' for m in agent.messages)
        results = sum(m['role'] == 'tool' for m in agent.messages)
        self.panel('КОНТЕКСТ',
                   f"{'Восстановлено' if restored else 'Сохранено'}: {users + answers} реплик диалога "
                   f'(ваших: {users}, агента: {answers})\n'
                   f'В JSON: {len(agent.messages)} технических записей — '
                   f'system: {systems}, user: {users}, assistant: {assistants}, tool: {results}\n'
                   f'Вызовов инструментов в истории: {calls}\n'
                   f'Отпечаток истории: {fingerprint}\n'
                   f'Файл: {agent.history.path if agent.history else "демо — история только в памяти"}', 'answer')
        if agent.history:
            self.console.print(Text('  /history — все записи: системная инструкция, сообщения, вызовы и результаты',
                                    style='dim'))

    def history(self, messages):
        labels = {'system': 'СИСТЕМНАЯ ИНСТРУКЦИЯ', 'user': 'ВАШЕ СООБЩЕНИЕ',
                  'assistant': 'СООБЩЕНИЕ МОДЕЛИ / ВЫЗОВ ИНСТРУМЕНТА', 'tool': 'РЕЗУЛЬТАТ ИНСТРУМЕНТА'}
        self.panel('ПОЛНАЯ ИСТОРИЯ JSON',
                   f'{len(messages)} сохранённых записей. Показ без изменений и без запросов к API.\n'
                   'assistant с tool_calls — вызов; tool со status=accepted — подтверждение ответа.', 'answer')
        for index, message in enumerate(messages, 1):
            role = message['role']
            kind = 'tool' if role == 'tool' else 'user' if role == 'user' else 'answer'
            self.panel(f"#{index} · {role} · {labels.get(role, role)}",
                        json.dumps(message, ensure_ascii=False, indent=2), kind)

    def tokens(self, agent):
        stats = agent.token_stats()
        table = Table.grid(expand=True, padding=(0, 2))
        table.add_column(); table.add_column(justify='right')

        def total(usage):
            return (number(usage['known_total_tokens']) + ' + ?'
                    if usage['unknown_calls'] else number(usage['total_tokens']))

        first = agent.trace[0].get('usage') if agent.trace else None
        incoming = first['prompt_tokens'] if first else None
        growth = stats.get('input_growth')
        delta = f' ({signed(growth)} за ход)' if growth is not None else ''
        table.add_row('Контекст на входе¹', number(incoming) + delta)
        turn = stats['turn']
        table.add_row('Сгенерировано за ход²', number(turn['completion_tokens']) + (' + ?' if turn['unknown_calls'] else ''))
        table.add_row(f"Расход за ход · запросов: {turn['calls']}", total(turn))
        table.add_row('Расход с начала сценария³', total(stats['session']))
        table.add_row('Размер нового сообщения · локально', number(stats['message_estimate']))
        table.add_row('Размер сохранённой истории · локально', number(stats['history_estimate']))
        notes = ['¹ Первый запрос хода: история + сообщение + инструкции и инструменты.',
                 '² Включая рассуждения и вызовы инструментов.',
                 '³ Сумма входа и выхода всех запросов, не размер контекста.']
        if stats['count_method'].startswith('оценка'):
            notes.append('Локальные размеры приблизительные.')
        if stats['session']['unknown_calls']:
            notes.append('? — API не сообщил расход неудачного запроса.')
        self.console.print(Panel(Group(table, Text('\n'.join(notes), style='dim')),
                                 title='ТОКЕНЫ', title_align='left', border_style=COLORS['tool'], padding=(0, 1)))

    def metrics_table(self, headings, title=None):
        table = Table(title=title, box=box.SIMPLE, header_style='bold #92b7ff',
                      border_style='#344258', expand=True, padding=(0, 1))
        for index, heading in enumerate(headings):
            table.add_column(heading, justify='left' if index == 0 else 'right', overflow='fold')
        return table

    def memory(self, result):
        if result.get('correct') and len(result['correct']) == len(result['expected']):
            known = sum(value is not None for value in result['expected'].values())
            unknown = len(result['expected']) - known
            self.panel('ПАМЯТЬ', f'{known} фактов верны · {unknown} неизвестных — корректно null. Ошибок нет.', 'answer')
            return
        table = self.metrics_table(('Факт', 'Ожидаемое', 'Ответ модели', 'Проверка'))
        labels = {'code': 'Код заявки', 'budget': 'Текущий бюджет', 'country': 'Страна',
                  'initial_budget': 'Первоначальный бюджет', 'meeting_point': 'Место встречи',
                  'departure_city': 'Город вылета', 'booking_pin': 'PIN · не сообщался'}
        recalled = result['recalled']
        for key, expected in result['expected'].items():
            actual = recalled.get(key) if isinstance(recalled, dict) else None
            valid = isinstance(recalled, dict) and key in recalled
            status = ('ФОРМАТ' if not valid else
                      'ВЕРНО' if type(actual) is type(expected) and actual == expected else
                      'НЕ ВСПОМНИЛ' if actual is None else 'ВЫДУМАНО' if expected is None else 'ОШИБКА')
            table.add_row(labels.get(key, key), Text(str(expected) if expected is not None else 'Не сообщалось'),
                          Text(str(actual) if actual is not None else 'null'),
                          Text(status, style=COLORS['answer'] if status == 'ВЕРНО' else COLORS['error']))
        self.console.print(Panel(Group(table, Text(result['status'], style='bold')),
                                 title='ПАМЯТЬ · ФАКТЫ, ИЗМЕНЕНИЯ И НЕИЗВЕСТНЫЕ ДАННЫЕ', title_align='left',
                                 border_style=COLORS['error'] if result.get('wrong') or result.get('missing')
                                 or not isinstance(recalled, dict) else COLORS['tool']))

    def comparison(self, report):
        self.console.rule('ИТОГ · СРАВНЕНИЕ ТРЁХ СЛУЧАЕВ', style=COLORS['answer'])
        summary = self.metrics_table(('Случай', 'API', 'Вход Σ', 'Выход Σ', 'Всего Σ'))
        for case in report['cases']:
            usage = case['usage']
            partial = ' + ?' if usage['unknown_calls'] else ''
            summary.add_row(case['title'], str(usage['calls']), number(usage['prompt_tokens']) + partial,
                            number(usage['completion_tokens']) + partial, number(usage['total_tokens']))
        self.console.print(summary)
        for case in report['cases']:
            table = self.metrics_table(('Ход', 'Контекст¹', 'Прирост', 'Выход за ход', 'Расход с начала'), case['title'])
            previous = None
            outcomes = []
            for row in case['turns']:
                usage = row['stats']['turn']
                known = not usage['unknown_calls']
                first = row['trace'][0].get('usage') if row['trace'] else None
                incoming = first['prompt_tokens'] if first else None
                probe = row.get('kind') == 'memory_probe'
                delta = '—' if probe or incoming is None or previous is None else signed(incoming - previous)
                total = row['stats']['session']['total_tokens']
                status = row.get('memory', {}).get('status', 'ответ получен')
                if row['error']:
                    status = 'лимит API' if row.get('error_kind') == 'context_length' else 'ошибка'
                label = f"П{row['dialogue_turn']}" if probe else str(row.get('dialogue_turn', row['number']))
                table.add_row(label, number(incoming), delta,
                              number(usage['completion_tokens']) if known else '?', number(total))
                if probe or row['error']:
                    outcomes.append(f"{'Проверка' if probe else 'Ход'} {row.get('dialogue_turn', row['number'])}: {status}")
                if not probe:
                    previous = incoming
            self.console.print(table)
            self.console.print(Text('  ' + ' · '.join(outcomes), style='dim'))
        self.console.print(Text('\n¹ Вход первого API-вызова хода. Δ — прирост к предыдущему ходу.\n'
                                'Σ — сумма всех вызовов, включая поиск и повторную отправку истории.\n'
                                 'П — отдельная проверка памяти; Δ сравнивает только ходы основного диалога.\n'
                                '? — неизвестный расход, не ноль. Отказ по лимиту ≠ галлюцинация.', style='dim'))

    def replay(self, messages, last_turns=5):
        starts = [index for index, message in enumerate(messages) if message['role'] == 'user']
        if not starts:
            for message in messages:
                if message['role'] == 'assistant':
                    if message.get('content') and not message.get('tool_calls'):
                        self.panel('ВОПРОС · СТАРТ ПРИЛОЖЕНИЯ', message['content'], 'answer')
                    for call in message.get('tool_calls') or []:
                        if call['id'] == 'local-opening-question':
                            self.panel('ВОПРОС · ИЗ ИСТОРИИ',
                                       answer_text(call['function']['arguments']), 'answer')
            return
        start = starts[-last_turns] if len(starts) > last_turns else 1
        self.panel('ПОСЛЕДНИЕ СООБЩЕНИЯ',
                   f'Показаны последние {min(len(starts), last_turns)} ходов диалога.\n'
                   'Вся история загружена; ограничен только показ в консоли.', 'answer')
        calls = {}
        for message in messages[start:]:
            role = message['role']
            if role == 'user':
                self.panel('ВЫ · ИЗ ИСТОРИИ', message['content'])
            elif role == 'assistant':
                # Сопутствующий content остаётся в JSON, но не является принятым ответом.
                for call in message.get('tool_calls') or []:
                    calls[call['id']] = call['function']
            elif role == 'tool':
                function = calls[message['tool_call_id']]
                result = json.loads(message['content'])
                if function['name'] == 'respond_to_user' and result.get('status') == 'accepted':
                    self.panel('ОТВЕТ · ИЗ ИСТОРИИ', answer_text(function['arguments']),
                               'answer', markdown=True)
                else:
                    self.event({'name': function['name'], 'arguments': function['arguments'],
                                'result': result}, historical=True)

    @contextmanager
    def pause_input(self):
        """Не показывать и не ставить в очередь ввод, набранный во время ответа."""
        if not sys.stdin.isatty():
            yield
            return
        fd = sys.stdin.fileno()
        original = termios.tcgetattr(fd)
        muted = original.copy()
        muted[3] &= ~(termios.ECHO | termios.ECHONL)
        try:
            termios.tcsetattr(fd, termios.TCSANOW, muted)
            yield
        finally:
            termios.tcsetattr(fd, termios.TCSAFLUSH, original)

    def ask(self, agent, prompt, show_prompt=True, compact=True):
        if show_prompt:
            self.panel('ВЫ', prompt)
        with self.pause_input():
            if not compact:
                self.console.print(Text('  Агент обрабатывает запрос — дождитесь ответа.', style='bold #e7ba76'))
            with Progress(SpinnerColumn(), TextColumn('{task.description}'), WaitColumn(),
                          console=self.console, transient=True) as progress:
                task_id = progress.add_task('Подготовка', total=agent.client.timeout, start=False)
                previous = agent.on_progress

                def network_event(event):
                    step = event['step']
                    if event['kind'] == 'request':
                        context = f'всю историю: {len(agent.messages)} сообщений'
                        self.console.print(Text(f'  → LLM: запрос №{step} · передаю {context}',
                                                style='#92b7ff'))
                        progress.reset(task_id, description=f'Запрос №{step} · ждём ответ LLM')
                    else:
                        progress.stop_task(task_id)
                        if event['kind'] == 'response':
                            self.console.print(Text(f"  ← LLM: ответ на запрос №{step} · {event['seconds']:.2f} с",
                                                    style='#76ddbd'))
                        elif event['kind'] == 'continue':
                            self.console.print(Text('  Агент уточняет ответ: ' + event['message'],
                                                    style='#e7ba76'))
                        progress.update(task_id, description='Обработка ответа')
                    if previous:
                        previous(event)

                agent.on_progress = network_event
                try:
                    answer = agent.ask(prompt)
                except (AgentError, ClientError, HistoryError):
                    self.tokens(agent)
                    raise
                finally:
                    agent.on_progress = previous
            seconds = sum(row['seconds'] for row in agent.trace)
            summary = [f'Запросов к модели: {len(agent.trace)} · {seconds:.2f} с']
            if agent.events:
                summary.append(f'Вызванные инструменты ({len(agent.events)}):')
                names = {'search_tours': 'Поиск путёвок', 'respond_to_user': 'Ответ пользователю (JSON)'}
                for number, event in enumerate(agent.events, 1):
                    name = event['name']
                    status = ' · ошибка' if 'error' in event['result'] else ''
                    summary.append(f"{number}. {names.get(name, name)} ({name}) "
                                   f"— после ответа модели №{event['step']}{status}")
                if not any(event['name'] == 'search_tours' for event in agent.events):
                    summary.append('Поиск путёвок не вызывался.')
            else:
                summary.append('Инструменты не вызывались — модель ответила сама.')
            self.panel(f'ОТВЕТ · {agent.client.model_name}', answer, 'answer', markdown=True,
                       sections=[] if compact else [('ВЫЗОВЫ ЗА ЭТОТ ЗАПРОС', '\n'.join(summary))])
            self.tokens(agent)
            if not compact:
                self.context(agent)
            self.console.print()
        return answer


def number(value):
    return f'{value:,}'.replace(',', ' ') if value is not None else '?'


def signed(value):
    return ('+' if value >= 0 else '−') + number(abs(value))


def money(value):
    return f'{value:,} ₽'.replace(',', ' ') if type(value) is int else str(value)


def format_parameters(arguments):
    labels = {'departure_city': 'Вылет из', 'departure_date': 'Дата вылета',
              'flex_days': 'Гибкость, ± дней', 'nights_min': 'Ночей от', 'nights_max': 'Ночей до',
              'adults': 'Взрослых', 'children_ages': 'Возраст детей', 'budget_total': 'Бюджет на всех',
              'country': 'Страна', 'resort': 'Курорт', 'stars_min': 'Звёзд от',
              'meal': 'Питание', 'features': 'Пожелания'}
    try:
        args = json.loads(arguments)
        if not isinstance(args, dict):
            return arguments
        lines = []
        for key, value in args.items():
            if value is None:
                value = 'не задано'
            elif key == 'budget_total':
                value = money(value)
            elif isinstance(value, list):
                value = ', '.join(map(str, value)) or 'нет'
            lines.append(f'{labels.get(key, key)}: {value}')
        return '\n'.join(lines)
    except ValueError:
        return arguments


def format_tour(tour):
    meals = {'AI': 'всё включено', 'HB': 'завтрак и ужин', 'BB': 'завтрак'}
    text = (f"{tour.get('id')} · {tour.get('hotel')}\n"
            f"{tour.get('country')} · {tour.get('resort')}\n"
            f"Вылет: {tour.get('departure_city')} · {tour.get('departure_date')} · {tour.get('nights')} ночей\n"
            f"Взрослых: {tour.get('adults')} · Детей: {tour.get('children')}\n"
            f"Отель: {tour.get('stars')}★ · {meals.get(tour.get('meal'), tour.get('meal'))}\n"
            f"Цена на всех: {money(tour.get('price_total'))}\n"
            f"Особенности: {', '.join(map(str, tour.get('features', [])))}")
    if tour.get('children'):
        text += f"\nВозраст детей: {tour.get('child_age_min')}–{tour.get('child_age_max')} лет"
    return text


def main():
    parser = argparse.ArgumentParser(description='День 8 · сравнение трёх сценариев расхода токенов')
    parser.add_argument('--env', type=Path, default=ROOT / '.env', help='Файл настроек API')
    parser.add_argument('--model', help='Модель, иначе LLM_MODEL или coder')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--demo', action='store_true', help='Все три сценария автоматически')
    modes.add_argument('--chat', action='store_true', help='Обычный чат с сохранением истории')
    parser.add_argument('--output-dir', type=Path, help='Новая папка отчёта --demo')
    modes.add_argument('--case', type=int, choices=(1, 2, 3), help='Один сценарий демо; 3 — реальное переполнение')
    args = parser.parse_args()
    terminal = Terminal()
    try:
        if not (args.demo or args.case or args.chat):
            terminal.panel('ДЕНЬ 8 · ВЫБЕРИ РЕЖИМ',
                           '1. Все три кейса по порядку — автоматический прогон\n'
                           '2. Только короткий диалог\n'
                           '3. Только длинный диалог\n'
                           '4. Только предел контекста и переполнение API\n'
                           '5. Обычный чат — отвечать самому\n\n'
                           'В кейсах сообщения уже подготовлены; вводить ответы не нужно.\n'
                           'Кейс 3: около 7000 новых токенов за ход, 7 проверок после каждого хода.\n'
                           'Enter — все три кейса. 0 — выход.', 'answer')
            while True:
                try:
                    selection = terminal.console.input('[bold #92b7ff]  Режим › [/]').strip() or '1'
                except EOFError:
                    return 0
                if selection == '0':
                    return 0
                if selection in ('1', '2', '3', '4', '5'):
                    args.demo = selection == '1'
                    args.chat = selection == '5'
                    args.case = int(selection) - 1 if selection in ('2', '3', '4') else None
                    break
                terminal.console.print('Введите 1–5 или 0 для выхода.', style=COLORS['error'])
        if args.output_dir and args.chat:
            parser.error('--output-dir используется для кейсов, а не для чата')
        settings = load_settings(args.env)
        if args.model:
            settings['LLM_MODEL'] = args.model
        catalog = json.loads((ROOT / 'data' / 'tours.json').read_text(encoding='utf-8'))
        store = HistoryStore(ROOT / 'data' / 'session.json')
        client = ChatClient(settings)
        client.load_model_catalog()
        if args.demo or args.case:
            from .demo import run_demo
            directory = args.output_dir or ROOT / 'results' / ('tokens-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
            return 0 if run_demo(client, terminal, catalog, directory, args.case) else 1
        agent = Agent(client, TravelTools(catalog), on_event=terminal.event, history=store)
        terminal.panel('К О М П А С  /  ДЕНЬ 8',
                       'Подбор путёвки · полная история между запусками\n'
                       'Данные и цены вымышленные. Работайте только в одном экземпляре.\n'
                       f'{len(catalog)} учебных путёвок · июнь–сентябрь 2027 · 5 городов вылета.\n\n'
                       '/new — новый подбор   /history — вся история   /exit — выход', 'answer')
        if agent.start_dialog():
            terminal.panel('НОВЫЙ ДИАЛОГ', 'Начинаем подбор путёвки.')
            terminal.panel('ВОПРОС · СТАРТ ПРИЛОЖЕНИЯ', OPENING_QUESTION, 'answer')
            terminal.console.print(Text('  Это готовая реплика: запросов API — 0, вызовов инструментов — 0.', style='dim'))
        elif agent.restored:
            terminal.panel('ИСТОРИЯ ВОССТАНОВЛЕНА', 'Продолжаем сохранённый диалог. /new — начать с нуля.', 'answer')
            terminal.replay(agent.messages)
        else:
            store.save(agent.messages)
            terminal.panel('НОВЫЙ ДИАЛОГ', 'Предыдущей истории нет. Расскажите, какой отдых хотите.')
            terminal.context(agent)
        while True:
            try:
                text = terminal.console.input('[bold #92b7ff]  Вы › [/]').strip()
            except EOFError:
                break
            if text.lower() in ('/exit', '/quit'):
                break
            if text.lower() == '/history':
                terminal.history(agent.messages)
                continue
            if text.lower() == '/new':
                try:
                    store.save([{'role': 'system', 'content': SYSTEM_PROMPT}])
                    agent = Agent(client, TravelTools(catalog), on_event=terminal.event, history=store)
                    agent.start_dialog()
                    terminal.panel('НОВЫЙ ПОДБОР', 'История очищена и сохранена. Начинаем заново.')
                    terminal.panel('ВОПРОС · СТАРТ ПРИЛОЖЕНИЯ', OPENING_QUESTION, 'answer')
                    terminal.console.print(Text('  Запросов API — 0, вызовов инструментов — 0.', style='dim'))
                except HistoryError as error:
                    terminal.panel('ОШИБКА', str(error), 'error')
            elif text:
                try:
                    terminal.ask(agent, text)
                except (AgentError, ClientError, HistoryError) as error:
                    terminal.panel('ОШИБКА', str(error), 'error')
                    terminal.panel('КОНТЕКСТ · ХОД НЕ СОХРАНЁН',
                                   'Незавершённый ход отменён. Продолжаем с последнего сохранённого ответа.', 'error')
        return 0
    except (ClientError, HistoryError, OSError, ValueError) as error:
        message = str(error) if isinstance(error, (ClientError, HistoryError)) else 'Не удалось прочитать данные.'
        terminal.panel('ОШИБКА ЗАПУСКА', message, 'error')
        return 1
    except KeyboardInterrupt:
        terminal.console.print('\n  Работа остановлена.\n', style='dim')
        return 130
