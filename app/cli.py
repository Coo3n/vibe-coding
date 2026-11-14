"""Цветной чат и четыре демонстрационных сюжета."""

import argparse
from contextlib import contextmanager
from datetime import datetime
import json
from pathlib import Path
import sys
import termios

from rich import box
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, ProgressColumn, SpinnerColumn, TextColumn
from rich.text import Text

from .agent import Agent, AgentError, matching_apartment
from .client import ChatClient, ClientError, load_settings
from .tools import HousingTools


ROOT = Path(__file__).resolve().parent.parent
COLORS = {'user': '#92b7ff', 'answer': '#76ddbd', 'tool': '#e7ba76', 'error': '#ff8a96'}
CASES = [
    ('Без инструментов', 'Что такое первоначальный взнос по ипотеке?'),
    ('Поиск до подходящей цены', 'Найди случайную квартиру в Санкт-Петербурге.'),
    ('Поиск до одобрения банка', 'Найди квартиру в Санкт-Петербурге, которую полностью оплатит банк.'),
]
GOALS = (None, 'price', 'mortgage')


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

    def event(self, event):
        result = event['result']
        if 'error' in result:
            self.panel('ОШИБКА ИНСТРУМЕНТА', f"{event['name']}: {result['error']}", 'error')
            return
        if event['name'] == 'choose_apartment':
            limit = json.loads(event['arguments']).get('max_price')
            condition = f'Цена до {money(limit)}' if limit is not None else 'Без ограничения цены'
            apartment = result.get('apartment')
            if result.get('status') == 'price_above_limit':
                rejected = result['rejected_apartment']
                details = (f"Квартира {rejected['id']} отсеяна по цене\n"
                           f"Цена: {money(rejected['price'])}\n"
                           f"Допустимый максимум: {money(result['price_limit'])}\n"
                           'Для этой квартиры проверка банка не выполняется.')
            elif apartment:
                rooms = 'студия' if apartment['rooms'] == 0 else str(apartment['rooms'])
                details = (f"Найдена квартира {apartment['id']}\n"
                           f"Район: {apartment['district']}\n"
                           f"Комнат: {rooms} · Площадь: {apartment['area_m2']:g} м²\n"
                           f"Цена: {money(apartment['price'])}")
            else:
                details = 'Новых квартир под эти условия не осталось.'
            self.panel('ИНСТРУМЕНТ · ПОИСК КВАРТИРЫ',
                       'Модель вызвала choose_apartment', 'tool',
                       sections=[('ПАРАМЕТРЫ', condition), ('РЕЗУЛЬТАТ', details)])
        elif event['name'] == 'check_mortgage':
            decision = 'Денег хватает на всю стоимость квартиры' if result['affordable'] else 'Денег не хватает'
            text = (f"Банк: {result['bank']}\nЛимит: {money(result['limit'])}\n"
                    f"Решение: {decision}")
            if not result['affordable']:
                text += f"\nНедостаёт: {money(result['shortfall'])}"
            self.panel('ИНСТРУМЕНТ · ПРОВЕРКА БАНКА', 'Модель вызвала check_mortgage', 'tool',
                       sections=[('КВАРТИРА', f"{result['apartment_id']} · {money(result['price'])}"),
                                 ('РЕШЕНИЕ БАНКА', text)])

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

    def ask(self, agent, prompt, show_prompt=True, goal=None):
        if show_prompt:
            self.panel('ВЫ', prompt)
        with self.pause_input():
            self.console.print(Text('  Агент обрабатывает запрос — дождитесь ответа.\n'
                                    '  Ввод временно отключён, новые сообщения не сохраняются.',
                                    style='bold #e7ba76'))
            with Progress(SpinnerColumn(), TextColumn('{task.description}'), WaitColumn(),
                          console=self.console, transient=True) as progress:
                task_id = progress.add_task('Подготовка', total=agent.client.timeout, start=False)
                previous = agent.on_progress

                def network_event(event):
                    step = event['step']
                    if event['kind'] == 'request':
                        context = ('ваше сообщение' if step == 1 else
                                   'результаты и текущее состояние подбора')
                        self.console.print(Text(f'  → LLM: запрос №{step} · передаю {context}',
                                                style='#92b7ff'))
                        progress.reset(task_id, description=f'Запрос №{step} · ждём ответ LLM')
                    else:
                        progress.stop_task(task_id)
                        if event['kind'] == 'response':
                            self.console.print(Text(f"  ← LLM: ответ на запрос №{step} · {event['seconds']:.2f} с",
                                                    style='#76ddbd'))
                        elif event['kind'] == 'continue':
                            self.console.print(Text('  Агент продолжает поиск: ' + event['message'],
                                                    style='#e7ba76'))
                        progress.update(task_id, description='Обработка ответа')
                    if previous:
                        previous(event)

                agent.on_progress = network_event
                try:
                    answer = agent.ask(prompt, goal=goal)
                finally:
                    agent.on_progress = previous
            seconds = sum(row['seconds'] for row in agent.trace)
            summary = [f'Запросов к модели: {len(agent.trace)} · {seconds:.2f} с']
            if agent.events:
                summary.append(f'Вызванные инструменты ({len(agent.events)}):')
                names = {'choose_apartment': 'Поиск квартиры', 'check_mortgage': 'Проверка банка'}
                for number, event in enumerate(agent.events, 1):
                    name = event['name']
                    status = ' · ошибка' if 'error' in event['result'] else ''
                    if event['result'].get('status') == 'price_above_limit':
                        status = ' · квартира отсеяна: дороже 15 млн ₽'
                    summary.append(f"{number}. {names.get(name, name)} ({name}) "
                                   f"— после ответа модели №{event['step']}{status}")
            else:
                summary.append('Инструменты не вызывались — модель ответила сама.')
            self.panel(f'ОТВЕТ · {agent.client.model_name}', answer, 'answer', markdown=True,
                       sections=[('ВЫЗОВЫ ЗА ЭТОТ ЗАПРОС', '\n'.join(summary))])
            self.console.print()
        return answer


def money(value):
    return f'{value:,} ₽'.replace(',', ' ')


def evaluate(index, events, answer):
    if not answer:
        return False
    names = [e['name'] for e in events]
    if index == 0:
        return not events
    if index == 1:
        return (set(names) == {'choose_apartment'}
                and matching_apartment(events, 'price') is not None)
    bank_limits = {e['result']['limit'] for e in events
                   if e['name'] == 'check_mortgage' and 'error' not in e['result']}
    return len(bank_limits) == 1 and matching_apartment(events, 'mortgage') is not None


def run_demo(client, terminal, catalog, directory, case_number=None):
    directory.mkdir(parents=True, exist_ok=False)
    report = {'started': datetime.now().astimezone().isoformat(),
               'route': client.settings['LLM_MODEL'],
               'note': 'Реальный API; выбор квартиры и лимит банка случайны, seed не фиксируется.',
               'cases': []}
    indices = [case_number - 1] if case_number else range(3)
    for index in indices:
        title, prompt = CASES[index]
        terminal.panel(f'КЛЮЧ · ДЕНЬ 6 · СЮЖЕТ {index + 1}',
                       f'{title} · {client.model_name}\n'
                       '300 учебных квартир в СПб · вымышленный банк «Нева»',
                       sections=[('ВАШ ЗАПРОС', prompt)])
        agent = Agent(client, HousingTools(catalog), on_event=terminal.event)
        answer, error = '', None
        try:
            answer = terminal.ask(agent, prompt, show_prompt=False, goal=GOALS[index])
        except (AgentError, ClientError) as caught:
            error = str(caught)
            terminal.panel('ОШИБКА', error, 'error')
        passed = not error and evaluate(index, agent.events, answer)
        report['cases'].append({'title': title, 'prompt': prompt, 'answer': answer, 'error': error,
                                'model': client.actual_model,
                                'goal': GOALS[index],
                                'matched_apartment': matching_apartment(agent.events, GOALS[index]),
                                'passed': passed, 'events': agent.events, 'trace': agent.trace})
        report['passed'] = len(report['cases']) == len(indices) and all(r['passed'] for r in report['cases'])
        (directory / 'demo.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                                            encoding='utf-8')
    return report['passed']


def main():
    parser = argparse.ArgumentParser(description='Ключ · квартирный агент с LLM tool calling')
    parser.add_argument('--env', type=Path, default=ROOT / '.env', help='Файл настроек API')
    parser.add_argument('--model', help='Модель, иначе LLM_MODEL или coder')
    parser.add_argument('--demo', action='store_true', help='Три обязательных сюжета ТЗ через API')
    parser.add_argument('--case', type=int, choices=(1, 2, 3), help='Один из трёх сюжетов ТЗ')
    parser.add_argument('--output-dir', type=Path, help='Новая папка для отчёта --demo')
    args = parser.parse_args()
    if args.output_dir and not (args.demo or args.case):
        parser.error('--output-dir используется вместе с --demo или --case')
    terminal = Terminal()
    try:
        settings = load_settings(args.env)
        if args.model:
            settings['LLM_MODEL'] = args.model
        catalog = json.loads((ROOT / 'data' / 'apartments.json').read_text(encoding='utf-8'))
        client = ChatClient(settings)
        client.load_model_catalog()
        if args.demo or args.case:
            directory = args.output_dir or ROOT / 'results' / ('local-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
            return 0 if run_demo(client, terminal, catalog, directory, args.case) else 1
        terminal.panel('К Л Ю Ч  /  ДЕНЬ 6',
                       f'300 квартир в Петербурге · банк «Нева» · {client.model_name}\n'
                       'Данные и банк вымышленные.\n\n/new — новый подбор   /exit — выход', 'answer')
        agent = Agent(client, HousingTools(catalog), on_event=terminal.event)
        while True:
            try:
                text = terminal.console.input('[bold #92b7ff]  Вы › [/]').strip()
            except EOFError:
                break
            if text.lower() in ('/exit', '/quit'):
                break
            if text.lower() == '/new':
                agent = Agent(client, HousingTools(catalog), on_event=terminal.event)
                terminal.panel('НОВЫЙ ПОДБОР', 'История очищена. Банк определит новый лимит при проверке.')
            elif text:
                try:
                    terminal.ask(agent, text)
                except (AgentError, ClientError) as error:
                    terminal.panel('ОШИБКА', str(error), 'error')
        return 0
    except (ClientError, OSError, ValueError) as error:
        message = str(error) if isinstance(error, ClientError) else 'Не удалось прочитать данные или создать папку отчёта.'
        terminal.panel('ОШИБКА ЗАПУСКА', message, 'error')
        return 1
    except KeyboardInterrupt:
        terminal.console.print('\n  Работа остановлена.\n', style='dim')
        return 130
