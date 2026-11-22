"""Локальный клиент для проверки потока данных и учебных ответов без сети."""

import json
import re

from .memory import LONG_TERM_HEADER, WORKING_HEADER, WORKING_SYSTEM_PROMPT
from .profile import PROFILE_HEADER, PROFILE_SYSTEM_PROMPT


FUTURE = re.compile(
    r'^(?:пожалуйста[, ]+)?(?:теперь|отныне|впредь|в дальнейшем|начиная с этого момента|'
    r'всегда|предпочитаю|не используй больше|отвечай впредь|перейди)(?=[\s,:])',
    re.I,
)
PREFERENCES = {
    'стиль': (('кратко', r'кратк|коротк|лаконич'), ('подробно', r'подроб|разв[её]рнут')),
    'формат': (('без кода', r'без\s+код|не\s+(?:показывай|добавляй|используй)\s+(?:больше\s+)?код'),
               ('с кодом', r'с\s+кодом|пример(?:ы|ов)?\s+код|(?:показывай|добавляй)\s+код')),
    'тон': (('неформально', r'неформальн|на\s+[«"„]?ты\b|по-дружески'),
            ('формально', r'(?<!не)формальн|на\s+[«"„]?вы\b|официально')),
}

CONVERSION = ('def to_rubles(yen, rubles_per_yen):\n'
              '    return round(yen * rubles_per_yen, 2)')
BUDGET = ('def daily_limit(total, fixed, reserve, days):\n'
          '    return (total - fixed - reserve) / days')
FOOD = ('def food_budget(days, daily_limit):\n'
        '    return days * daily_limit')
PASS = ('def pass_is_cheaper(pass_cost, covered_tickets):\n'
        '    return pass_cost < sum(covered_tickets)')
TABLE = ('def daily_rows(totals):\n'
         '    return list(enumerate(totals, start=1))')

# Порядок правил выбирает сначала узкую тему, затем общую.
RULES = (
    (r'конвертац|конвертир|курс иен|иен.*рубл', None,
     'Умножь сумму в иенах на курс в рублях за одну иену.', CONVERSION),
    (r'таблиц.*расход', 'бюджет',
     'Заведи по строке на день и колонки для еды, поездок и других расходов.', TABLE),
    (r'расход.*дневн|ежедневн.*расход|уложиться в бюджет|какой.*бюджет', 'бюджет',
     'Вычти фиксированные траты и резерв, остаток раздели на число дней.', BUDGET),
    (r'сколько.*еду|на еду.*посчит', 'бюджет',
     'Умножь дневной лимит на еду на число дней и держи отдельный резерв.', FOOD),
    (r'что.*поесть|ограничени.*еде|аллерг|вегетариан', 'ограничения по еде',
     'Запиши ограничения на карточке и сверяй состав блюда с персоналом.', None),
    (r'из токио в киото|по маршруту', 'маршрут',
     'Привяжи переезд ко времени выезда из отеля и оставь запас на дорогу до станции.', None),
    (r'гион|жить в киото', 'маршрут',
     'Сгруппируй места по району и оставь время на прогулку по Гиону.', None),
    (r'виз[уыае]|20 марта', 'виза',
     'Вынеси проверку документов и подачу в отдельные пункты плана поездки.', None),
    (r'нар[уеы]|олен', 'экскурсия',
     'Запланируй отдельную поездку с возвращением в Киото в тот же день.', None),
    (r'jr\s*pass|окупится', 'транспорт',
     'Сравни цену пропуска с суммой билетов на покрываемые им сегменты.', PASS),
    (r'взять с собой', 'даты',
     'Собери документы, зарядку и удобную обувь. Одежду подбери ближе к выезду.', None),
    (r'обменять деньги', 'бюджет',
     'Сравни полную стоимость обмена и заложи небольшую сумму наличными.', None),
)

FORMAL_VERBS = {
    'Умножь': 'Умножьте', 'Заведи': 'Заведите', 'Вычти': 'Вычтите', 'Запиши': 'Запишите',
    'Привяжи': 'Привяжите', 'Сгруппируй': 'Сгруппируйте', 'Вынеси': 'Вынесите',
    'Запланируй': 'Запланируйте', 'Сравни': 'Сравните', 'Собери': 'Соберите',
    'раздели': 'разделите', 'держи': 'держите', 'сверяй': 'сверяйте', 'оставь': 'оставьте',
    'подбери': 'подберите', 'заложи': 'заложите',
}
FORMAL_VERB = re.compile(r'\b(?:' + '|'.join(FORMAL_VERBS) + r')\b')


class OfflineClient:
    mode = 'offline'
    model_name = 'Локальные шаблоны'

    def complete(self, messages):
        instruction = messages[0]['content']
        text = messages[-1]['content']
        if instruction == WORKING_SYSTEM_PROMPT:
            return self._working(text), None
        if instruction == PROFILE_SYSTEM_PROMPT:
            return self._preferences(text), None
        return self._answer(messages, text), None

    @staticmethod
    def _preferences(text):
        prefix, _, utterance = text.partition('\n\nНовое сообщение пользователя:\n')
        profile = json.loads(prefix.split('\n', 1)[1])
        if FUTURE.search(utterance.strip()):
            for field, choices in PREFERENCES.items():
                alternatives = '|'.join(f'(?P<v{index}>{pattern})'
                                        for index, (_, pattern) in enumerate(choices))
                matches = list(re.finditer(alternatives, utterance, re.I))
                if matches:
                    profile[field] = choices[int(matches[-1].lastgroup[1:])][0]
            for pattern, value in ((r'без\s+эмодзи|не\s+используй\s+(?:больше\s+)?эмодзи', 'без эмодзи'),
                                   (r'пиши\s+списк|отвечай\s+списк', 'пиши списком'),
                                   (r'без\s+вступлен', 'без вступлений'),
                                   (r'без\s+извинен', 'без извинений')):
                if re.search(pattern, utterance, re.I) and value not in profile['ограничения']:
                    profile['ограничения'].append(value)
        return json.dumps(profile, ensure_ascii=False)

    @staticmethod
    def _working(text):
        prefix, _, utterance = text.partition('\n\nНовое сообщение пользователя:\n')
        working = json.loads(prefix.split('\n', 1)[1])
        if not utterance.rstrip().endswith('?'):
            for key in ('даты', 'бюджет', 'маршрут', 'цель'):
                match = re.match(rf'^{key}\s*(?:[=:—]|определились\s*:?)\s*(.+)', utterance, re.I)
                if match:
                    working[key] = match[1].rstrip('.')
        return json.dumps(working, ensure_ascii=False)

    @classmethod
    def _answer(cls, messages, question):
        facts = {}
        profile = {}
        restrictions = []
        for message in messages:
            if message['role'] != 'system':
                continue
            content = message['content']
            title, _, block = content.partition('\n')
            if title in (LONG_TERM_HEADER, WORKING_HEADER):
                facts.update(json.loads(block))
            elif title.startswith(PROFILE_HEADER):
                for line in block.splitlines():
                    field, _, value = line.partition(': ')
                    if field == '- ограничение':
                        restrictions.append(value)
                    elif field.startswith('- ') and value:
                        profile[field[2:]] = value
        formal = profile.get('тон') == 'формально'
        detailed = profile.get('стиль ответов') == 'подробно'
        with_code = profile.get('формат') == 'с кодом'
        normalized = question.casefold()
        key, tip, example = next(
            ((key, tip, example) for pattern, key, tip, example in RULES
             if re.search(pattern, normalized)),
            (None, 'Уточни, какая часть поездки тебя интересует.', None),
        )
        if FUTURE.search(question.strip()):
            tip, key = 'Учту новые предпочтения в следующих ответах.', None
            example = None
        fact = facts.get(key)
        if formal:
            tip = FORMAL_VERB.sub(lambda match: FORMAL_VERBS[match[0]], tip).replace(' тебя ', ' Вас ')
        heading = f'{key.capitalize()} — {fact}.' if isinstance(fact, str) else tip
        if heading != tip:
            tip = heading + '\n' + tip
        if detailed:
            tip += cls._details(heading, formal)
        elif not any('без вступлен' in item for item in restrictions):
            tip = ('Для Вас. ' if formal else 'Для тебя. ') + tip
        else:
            tip += '\n' + ('Вы можете сверить это с планом.' if formal else 'Ты можешь сверить это с планом.')
        if any('списк' in item for item in restrictions):
            tip = '\n'.join(f'• {line}' if line else '' for line in tip.splitlines())
        if with_code and example and (key not in ('бюджет', 'транспорт')
                                      or re.search(r'код|скрипт|расход|посчит|окупится|таблиц',
                                                   normalized)):
            tip += '\n\n```python\n' + example + '\n```'
        return tip

    @staticmethod
    def _details(heading, formal):
        if formal:
            return (
                '\n\n1. Что уже известно. ' + heading + ' Вы можете использовать это '
                'как исходную запись для общей таблицы поездки. Сохраняйте единицы измерения '
                'рядом с числами и помечайте, какие данные уже подтверждены.\n'
                '2. Как проверить расчёт или решение. Выпишите исходные условия отдельно от '
                'производных значений. Если сравниваете варианты, применяйте к каждому одни и те же '
                'условия, иначе сравнение смешает отличия маршрута и способа расчёта. '
                'Проверьте один небольшой пример вручную перед тем, как распространять его на всю поездку.\n'
                '3. Как использовать результат. Добавьте результат к плану по дням и обсудите '
                'его со спутником. Вы сможете обновить одну исходную цифру или условие и '
                'повторить расчёт. Отдельно запишите действия, которые зависят от даты, и сведения, '
                'которые нужно уточнить перед покупкой. Для неизвестных величин оставьте пустое '
                'поле с пояснением, чтобы их случайно не приняли за подтверждённые.\n'
                '4. Контроль. Сверьте итог с целью вопроса, а затем с общей суммой и сроками поездки.'
            )
        return (
            '\n\n1. Собери исходные условия. ' + heading + ' Ты можешь перенести это в план '
            'и указать рядом, когда последний раз проверялись данные. Для денежных величин '
            'выбери одну валюту расчёта и сохрани курс отдельно от сумм расходов. Если речь '
            'о маршруте, вынеси время переезда и время на пересадку в разные строки.\n'
            '2. Разбей расчёт или решение на шаги. Сначала посчитай один день или один '
            'сегмент. Проверь промежуточный результат вручную. Затем повтори порядок для '
            'всех оставшихся дней или сегментов. Не подставляй неизвестные величины как нули: '
            'запиши, что нужно выяснить, чтобы итог оставался понятным.\n'
            '3. Сравни варианты по общей цене и срокам. Обсуди их со спутником, отметь '
            'фиксированные условия и оставь запас. Так тебе будет проще понять, какие данные '
            'можно заменить, если решение изменится.\n'
            '4. Заверши сверкой. Проверь один пограничный пример и сохрани промежуточные '
            'значения. Позже ты сможешь повторить тот же расчёт с обновлёнными данными.'
        )
