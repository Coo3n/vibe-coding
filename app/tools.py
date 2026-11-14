"""Детерминированный поиск в фиксированном учебном каталоге."""

from datetime import date
from copy import deepcopy
import json

from .response import RESPONSE_FORMAT, reply_payload


PROPERTIES = {
    'departure_city': {'type': 'string', 'description': 'Город вылета по-русски'},
    'departure_date': {'type': 'string', 'description': 'Дата вылета YYYY-MM-DD'},
    'flex_days': {'type': 'integer', 'minimum': 0, 'maximum': 30},
    'nights_min': {'type': 'integer', 'minimum': 1, 'maximum': 30},
    'nights_max': {'type': 'integer', 'minimum': 1, 'maximum': 30},
    'adults': {'type': 'integer', 'minimum': 1, 'maximum': 6},
    'children_ages': {'type': 'array', 'items': {'type': 'integer', 'minimum': 0, 'maximum': 17}},
    'budget_total': {'type': 'integer', 'minimum': 1, 'description': 'Бюджет на всех в рублях'},
    'country': {'type': 'string', 'description': 'Не передавать, если направление не важно'},
    'resort': {'type': 'string'},
    'stars_min': {'type': 'integer', 'minimum': 1, 'maximum': 5},
    'meal': {'type': 'string', 'enum': ['AI', 'HB', 'BB'],
             'description': 'AI — всё включено, HB — завтрак и ужин, BB — завтрак'},
    'features': {'type': 'array', 'items': {'type': 'string', 'enum': [
        'пляжный', 'экскурсионный', 'активный', 'спокойный', 'первая линия', 'для детей', 'без визы']}},
}
REQUIRED = ['departure_city', 'departure_date', 'nights_min', 'nights_max',
            'adults', 'children_ages', 'budget_total']
SCHEMA_PROPERTIES = deepcopy(PROPERTIES)
for key, schema in SCHEMA_PROPERTIES.items():
    schema['type'] = [schema['type'], 'null']
    if 'enum' in schema:
        schema['enum'].append(None)

TOOL_SCHEMAS = [{'type': 'function', 'function': {
    'name': 'search_tours',
    'description': 'Поиск путёвок в фиксированном вымышленном каталоге. '
                   'Цена за всю группу включает перелёт и проживание. '
                   'Все features обязательны одновременно. Не меняй условия без согласия пользователя. '
                   'Неизвестные параметры передавай null: поиск без этих ограничений, не выдумывай их. '
                   'Пустая выдача означает отсутствие вариантов, а не разрешение их придумать.',
    'strict': True,
    'parameters': {'type': 'object', 'properties': SCHEMA_PROPERTIES, 'required': list(PROPERTIES),
                   'additionalProperties': False}}},
    {'type': 'function', 'function': {
        'name': 'respond_to_user', 'strict': True,
        'description': 'Передать пользователю один уточняющий вопрос (action=ask) или ответ '
                       '(action=answer). Не выполняет поиск. Если нужны варианты — сначала '
                       'вызови search_tours, затем respond_to_user по результату. '
                       'Нельзя обещать поиск вместо его выполнения.',
        'parameters': deepcopy(RESPONSE_FORMAT['json_schema']['schema']),
    }}]


class TravelTools:
    def __init__(self, catalog):
        self.catalog = catalog

    def execute(self, name, arguments):
        try:
            if name == 'respond_to_user':
                reply_payload(arguments)
                return {'status': 'accepted'}
            if name != 'search_tours':
                raise ValueError('Неизвестный инструмент')
            args = json.loads(arguments)
            if not isinstance(args, dict) or set(args) - PROPERTIES.keys():
                raise ValueError('Нужны обязательные параметры поиска без неизвестных полей')
            args = {key: value for key, value in args.items() if value is not None}
            for key, value in args.items():
                schema = PROPERTIES[key]
                if schema['type'] == 'integer':
                    if type(value) is not int or not schema.get('minimum', 0) <= value <= schema.get('maximum', 10**12):
                        raise ValueError(f'Некорректное число: {key}')
                elif schema['type'] == 'string':
                    if not isinstance(value, str) or not value.strip() or ('enum' in schema and value not in schema['enum']):
                        raise ValueError(f'Некорректная строка: {key}')
                elif not isinstance(value, list):
                    raise ValueError(f'Нужен список: {key}')
            ages = args.get('children_ages')
            if ages is not None and (len(ages) > 4 or any(type(age) is not int or not 0 <= age <= 17 for age in ages)):
                raise ValueError('Возраст детей — целые числа от 0 до 17, не более четырёх детей')
            features = args.get('features', [])
            if any(not isinstance(f, str) or f not in PROPERTIES['features']['items']['enum'] for f in features):
                raise ValueError('Неизвестное пожелание к отдыху')
            departure = date.fromisoformat(args['departure_date']) if 'departure_date' in args else None
            if departure is not None and departure.isoformat() != args['departure_date']:
                raise ValueError('Дата должна быть YYYY-MM-DD')
            if args.get('nights_min', 1) > args.get('nights_max', 30):
                raise ValueError('Минимум ночей больше максимума')
            found = []
            for tour in self.catalog:
                if any(tour[key].casefold() != args[key].strip().casefold()
                       for key in ('departure_city', 'country', 'resort') if key in args):
                    continue
                if departure is not None and abs((date.fromisoformat(tour['departure_date']) - departure).days) > args.get('flex_days', 0):
                    continue
                if not args.get('nights_min', 1) <= tour['nights'] <= args.get('nights_max', 30):
                    continue
                if 'adults' in args and tour['adults'] != args['adults']:
                    continue
                if ages is not None and (len(ages) != tour['children'] or any(not tour['child_age_min'] <= age <= tour['child_age_max'] for age in ages)):
                    continue
                if tour['price_total'] > args.get('budget_total', 10**12) or tour['stars'] < args.get('stars_min', 1):
                    continue
                if 'meal' in args and tour['meal'] != args['meal']:
                    continue
                if not set(features) <= set(tour['features']):
                    continue
                found.append(tour)
            ordered = sorted(found, key=lambda t: (t['price_total'], t['id']))
            return {'mock': True, 'currency': 'RUB', 'tours': ordered[:5],
                    'count': len(found), 'shown': min(len(found), 5),
                    'unspecified': [key for key in REQUIRED if key not in args],
                    'note': 'Показаны до 5 самых дешёвых вариантов. Неизвестные параметры не ограничивали поиск; '
                            'состав группы каждого варианта и общая цена указаны в выдаче. Данные учебные.'}
        except (ValueError, TypeError, OverflowError) as error:
            return {'error': str(error)}
