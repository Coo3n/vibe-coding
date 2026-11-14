"""Короткий, длинный диалог и настоящее переполнение API без локальной блокировки."""

from copy import deepcopy
from datetime import datetime
import json
import re

from .agent import Agent, AgentError
from .client import ClientError
from .tools import TravelTools


SHORT = 'Вылет из Москвы. Хочу пляжный отдых в Турции. Пока не ищи, уточни условия.'
LONG = [SHORT, '10 июня 2027 года, 7 ночей, даты строгие.',
        'Двое взрослых, детей нет.', 'Бюджет 200000 рублей на всех, 4 звезды, всё включено. Ищи.',
        'Сравни найденные варианты по цене. Не выполняй новый поиск.',
        'Напомни мои условия и найденные ID, не выполняй новый поиск.']
FACTS = {'code': 'KMP-739182', 'budget': 187400, 'country': 'Шри-Ланка'}
ANCHOR = ('Запомни мою контрольную карточку поездки: код заявки KMP-739182, '
          'бюджет 187400 рублей на всех, страна Шри-Ланка. '
          'Пока не ищи туры и не задавай вопросов, просто подтверди получение.')
PROBE = ('Без нового поиска проверь память всей переписки. '
          'Вызови respond_to_user с action=answer, а в строке answer верни только JSON '
          'с ключами code (код заявки), initial_budget (первоначальный бюджет), '
          'budget (последний установленный бюджет), country (страна), '
          'meeting_point (согласованное место встречи), departure_city (город вылета), '
          'booking_pin (секретный PIN бронирования, не код заявки). Бюджеты — числа. '
          'Верни точные значения из переписки. Если значение не сообщалось или '
          'не помнишь его, поставь null. Не выводи значения из учебных ситуаций и не угадывай.')

# Новые факты сообщаются ровно один раз в основной истории.
UPDATES = {
    5: ('Изменение моей контрольной карточки: теперь бюджет 213600 рублей на всех. '
        'Это заменяет прежний бюджет; остальные условия не меняются.', {'budget': 213600}),
    8: ('Личная договорённость по моей поездке: место встречи — «Северный вход, стойка 17». '
        'Запомни точную формулировку.', {'meeting_point': 'Северный вход, стойка 17'}),
    12: ('Дополнение к моей контрольной карточке: город вылета — Казань.',
         {'departure_city': 'Казань'}),
}


def expected_memory(turn):
    expected = {**FACTS, 'initial_budget': FACTS['budget'], 'meeting_point': None,
                'departure_city': None, 'booking_pin': None}
    for number, (_, values) in UPDATES.items():
        if number <= turn:
            expected.update(values)
    return expected


TOPICS = ('трансфер из аэропорта', 'страховка', 'багаж', 'питание в дороге',
          'связь за рубежом', 'прогулки', 'экскурсии', 'выбор номера',
          'документы', 'аптечка', 'погода', 'отмена поездки')
MAX_DIALOGUE_TURNS = 80
MATERIAL_TOKENS = 7000
PROBE_INTERVAL = 1


def conversation_message(number, counter):
    header = (f'Новые материалы к обсуждению поездки, выпуск {number}. '
              'Это вымышленные учебные ситуации, не реальные предложения и не изменения '
              'моей контрольной карточки. Сохрани предыдущую переписку. Не ищи туры '
              'и не повторяй код, бюджет или страну из карточки.\n')
    footer = ('\nВыдели три вопроса, которые стоит уточнить по новым материалам. '
              'Ответь кратко, до 60 слов. Не пересказывай все материалы.')
    sections = []
    while True:
        item = len(sections) + 1
        topic = TOPICS[(number + item) % len(TOPICS)]
        sections.append(
            f'\nСитуация {number}.{item}: {topic}. Учебная группа готовит этап '
            f'маршрута {number * 100 + item}. До выезда осталось {2 + item % 19} дней. '
            f'В обсуждении участвуют {2 + (number + item) % 7} человек. '
            'Один участник предлагает подготовить всё самостоятельно, другой просит '
            'письменное подтверждение от организатора. Первый вариант даёт больше свободы '
            'при изменении расписания, но требует самостоятельно проверять документы, '
            'контакты и условия. Второй сокращает число самостоятельных действий, '
            'однако состав услуг ещё не подтверждён.\n'
            'В рабочей заметке нужно разделить уже согласованные действия и открытые '
            'вопросы. Согласовано: составить список необходимых вещей, сохранить копии '
            'документов отдельно от оригиналов и записать контакты ответственных. '
            'Не согласовано: кто оплачивает дополнительные услуги, в какие сроки '
            'можно менять договорённости и куда обращаться при задержке. Нельзя считать '
            'страховку, питание, регистрацию или трансфер включёнными без подтверждения. '
            'Если основной вариант окажется недоступен, группа сначала уточнит доступные '
            'альтернативы и условия отмены, а затем выберет новый вариант. '
            'Эта ситуация нужна для обсуждения подготовки и не задаёт пожелания клиента.\n')
        update = UPDATES.get(number, ('', {}))[0]
        message = header + ''.join(sections) + ('\n' + update if update else '') + footer
        if counter.text(message) >= MATERIAL_TOKENS:
            return message


def dialogue_prompts(counter):
    yield ANCHOR, False, 1
    yield PROBE, True, 1
    for turn in range(2, MAX_DIALOGUE_TURNS + 1):
        yield conversation_message(turn, counter), False, turn
        if turn % PROBE_INTERVAL == 0:
            yield PROBE, True, turn


def memory_check(answer, expected=None):
    expected = FACTS if expected is None else expected
    try:
        text = answer.strip()
        if text.startswith('```'):
            text = text.split('\n', 1)[1].rsplit('```', 1)[0]
        recalled = json.loads(text)
        if not isinstance(recalled, dict) or set(recalled) != set(expected):
            raise ValueError
    except (AttributeError, ValueError, IndexError):
        return {'status': 'не удалось проверить формат', 'expected': expected, 'recalled': answer}
    correct = [key for key, value in recalled.items()
               if type(value) is type(expected[key]) and value == expected[key]]
    wrong = {key: value for key, value in recalled.items() if value is not None and key not in correct}
    missing = [key for key, value in recalled.items() if value is None and expected[key] is not None]
    invented = [key for key in wrong if expected[key] is None]
    status = 'ошибка фактов' if wrong else 'не вспомнил' if missing else 'всё верно'
    return {'status': f'{len(correct)}/{len(expected)} проверок верны · {status}',
            'expected': expected, 'recalled': recalled, 'wrong': wrong, 'missing': missing,
            'invented': invented, 'correct': correct}


def run_demo(client, terminal, catalog, directory, case_number=None):
    window = client.context_window
    indices = [case_number - 1] if case_number else range(3)
    directory.mkdir(parents=True, exist_ok=False)
    report = {'started': datetime.now().astimezone().isoformat(), 'route': client.settings['LLM_MODEL'],
              'note': 'Все запросы отправляются в API. Локальной блокировки нет. '
                       'Один накопительный диалог из подготовленных сообщений и реальных ответов. '
                       'Проверки памяти идут в отдельных ветках текущей истории.',
              'context_window': window, 'cases': []}
    terminal.panel('ДЕНЬ 8 · КАК ЧИТАТЬ ЭКСПЕРИМЕНТ',
                   f'Модель: {client.model_name}\nОкно из каталога: {window or "неизвестно"} токенов\n\n'
                   '1 · Короткий — базовый расход одного хода.\n'
                   '2 · Длинный — как история увеличивает каждый следующий запрос.\n'
                   '3 · Предел — накапливаем сообщения и ответы до отказа API.\n\n'
                   'Синие числа — токены, жёлтый блок — расход, красный — проблема.\n'
                   f'Около {MATERIAL_TOKENS} новых токенов за ход; короткие ответы.\n'
                   f'До {MAX_DIALOGUE_TURNS} ходов; после каждого — 7 проверок памяти одним запросом.\n'
                   'Ответы проверок не добавляются в основной диалог, чтобы не освежать факты.', 'answer')

    def save():
        report['model'] = client.model_name
        report['passed'] = len(report['cases']) == len(indices) and all(c.get('passed') for c in report['cases'])
        (directory / 'demo.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    for index in indices:
        title = ('Короткий диалог', 'Длинный диалог', 'Предел контекста API')[index]
        agent = Agent(client, TravelTools(catalog))
        agent.start_dialog()
        case = {'title': title, 'turns': [], 'passed': False}
        if index == 2:
            case.update(overflow_confirmed=False, max_dialogue_turns=MAX_DIALOGUE_TURNS,
                        material_tokens=MATERIAL_TOKENS, probe_interval=PROBE_INTERVAL)
        report['cases'].append(case)
        terminal.panel(f'СЛУЧАЙ {index + 1}/3 · {title}',
                       f'{client.model_name} · окно API: {window or "неизвестно"}\n'
                       'Локального лимита нет. История сценария изолирована от session.json.', 'answer')
        prompts = (dialogue_prompts(agent.counter) if index == 2 else
                   ((text, False, n) for n, text in enumerate([SHORT] if index == 0 else LONG, 1)))
        dialogue = agent
        for number, (prompt, is_probe, dialogue_turn) in enumerate(prompts, 1):
            agent = dialogue
            if is_probe:
                agent = Agent(client, TravelTools(catalog))
                agent.messages = deepcopy(dialogue.messages)
                # Общий расход сценария, но история и прирост основного диалога независимы.
                agent.meter = dialogue.meter
            before = deepcopy(agent.messages)
            answer, error, error_kind = None, None, None
            phase = 'ПРОВЕРКА ПАМЯТИ' if is_probe else 'ХОД ДИАЛОГА'
            terminal.panel(f'{phase} {dialogue_turn} · запросный шаг {number}',
                           f'В истории до запроса: {len(agent.messages)} записей.\n' +
                           (prompt if len(prompt) < 2000 else
                            f'Новые учебные материалы: {agent.counter.text(prompt):,} токенов '
                            '(локальный подсчёт).\nПредыдущая история сохранена. '
                            'Полный текст будет в demo.json.'))
            try:
                answer = terminal.ask(agent, prompt, show_prompt=False, compact=True)
            except (AgentError, ClientError) as caught:
                error, error_kind = str(caught), getattr(caught, 'kind', 'agent')
                terminal.panel('ОТВЕТ API · ПЕРЕПОЛНЕНИЕ' if error_kind == 'context_length' else 'ОШИБКА', error, 'error')
                terminal.panel('ЧТО ПРОИЗОШЛО',
                               ('Сервер отклонил вход: история не помещается в окно.\n'
                                'Ответа модели нет — оценивать галлюцинации на этом шаге нельзя.\n'
                                'Последний ход отменён; предыдущая история сохранена.')
                               if error_kind == 'context_length' else
                               'Ход не завершён. Эта ошибка НЕ доказывает переполнение контекста.\n'
                               'Причина указана выше; расход без usage неизвестен.', 'error')
            stats = agent.token_stats()
            row = {'number': number, 'answer': answer, 'error': error, 'error_kind': error_kind,
                   'kind': 'memory_probe' if is_probe else 'dialogue', 'dialogue_turn': dialogue_turn,
                   'prompt': prompt, 'history_records_before': len(before),
                   'history_records_after': len(agent.messages),
                   'stats': stats, 'trace': deepcopy(agent.trace), 'history_unchanged': agent.messages == before}
            if is_probe and answer:
                row['memory'] = memory_check(answer, expected_memory(dialogue_turn))
                terminal.memory(row['memory'])
            case['turns'].append(row)
            case['usage'] = stats['session']
            if index == 2:
                case['overflow_confirmed'] = error_kind == 'context_length' and agent.messages == before
                case['passed'] = case['overflow_confirmed']
                if case['overflow_confirmed']:
                    match = re.search(r'maximum context length is\s+(\d+)', error)
                    case['context_window_from_error'] = int(match[1]) if match else None
                    if match:
                        terminal.panel('ЛИМИТ ИЗ ОШИБКИ СЕРВЕРА',
                                       f'{match[1]} токенов; каталог сообщил {window}.\n'
                                       'Лимит из каталога не совпадает с ограничением сервера.')
            else:
                case['passed'] = error is None and number == (1 if index == 0 else len(LONG))
            save()
            if error:
                break
        if index == 2:
            terminal.panel('ВЫВОД ПО ЛИМИТУ',
                           'Сервер подтвердил переполнение. Ответ не сгенерирован, незавершённый ход отменён.'
                           if case['overflow_confirmed'] else 'Переполнение НЕ подтверждено. Смотрите ответ сервера и фактический usage.\n'
                           'Успешный ответ, таймаут и другие HTTP-ошибки не считаются доказательством переполнения.',
                            'answer' if case['passed'] else 'error')
            terminal.panel('ПРОВЕРКИ ПАМЯТИ',
                           'Смотрите сверки фактов по ходу растущего диалога.\n'
                           f'Остановка: ошибка API либо {MAX_DIALOGUE_TURNS} ходов. '
                           'Достижение числа ходов не считается переполнением.')
    terminal.comparison(report)
    terminal.panel('РЕЗУЛЬТАТЫ',
                   'Ошибки памяти показаны только если реально обнаружены; приближение к лимиту не гарантирует галлюцинаций.\n'
                   f'Трассировка: {directory / "demo.json"}', 'answer')
    return report['passed']
