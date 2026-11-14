"""Каталог и два инструмента вымышленного рынка квартир."""

import json
import math
import random


MAX_PRICE = 15_000_000


def generate_catalog():
    rng = random.Random(606)
    districts = ['Адмиралтейский', 'Василеостровский', 'Выборгский', 'Калининский',
                 'Кировский', 'Колпинский', 'Красногвардейский', 'Красносельский',
                 'Московский', 'Невский', 'Петроградский', 'Приморский',
                 'Пушкинский', 'Фрунзенский', 'Центральный']
    rows = []
    for index in range(300):
        rooms = rng.randrange(5)
        area = round(rng.uniform(22 + rooms * 12, 35 + rooms * 22), 1)
        price = round(area * rng.randint(180_000, 390_000) / 50_000) * 50_000
        rows.append({'id': f'SPB-{index + 1:03}', 'city': 'Санкт-Петербург',
                     'district': districts[index % len(districts)], 'rooms': rooms,
                     'area_m2': area, 'price': price, 'mock': True})
    return rows


TOOL_SCHEMAS = [
    {'type': 'function', 'function': {
        'name': 'choose_apartment',
        'description': 'Случайная непросмотренная квартира из учебного каталога СПб. '
                       'Возвращает ID, район, комнаты, площадь и цену в рублях; '
                       'если квартира не проходит проверку допустимой цены, возвращает '
                       'apartment=null, status=price_above_limit и фактический price_limit. '
                       'Если вариантов нет, apartment=null и status=not_found.',
        'parameters': {'type': 'object', 'properties': {
            'max_price': {'type': 'integer', 'minimum': 1,
                          'description': 'Максимальная цена в рублях, необязательно'}},
            'additionalProperties': False}}},
    {'type': 'function', 'function': {
        'name': 'check_mortgage',
        'description': 'Вымышленный банк «Нева»: определяет лимит финансирования без взноса '
                       'и проверяет, хватит ли его на квартиру по ID. '
                       'Фактический лимит возвращается в результате и сохраняется на весь подбор. '
                       'Квартира должна пройти проверку допустимой цены.',
        'parameters': {'type': 'object', 'properties': {
            'apartment_id': {'type': 'string', 'description': 'ID квартиры, например SPB-001'}},
            'required': ['apartment_id'], 'additionalProperties': False}}},
]


class HousingTools:
    def __init__(self, catalog, rng=None):
        if not catalog or not 0 < min(r['price'] for r in catalog) <= MAX_PRICE:
            raise ValueError('Каталог должен содержать квартиру не дороже 15 млн ₽')
        self.catalog = catalog
        self.rng = rng if rng is not None else random.Random()
        self.bank_limit = None
        self.seen = set()

    def choose_apartment(self, max_price=None):
        candidates = [r for r in self.catalog if r['id'] not in self.seen
                      and (max_price is None or r['price'] <= max_price)]
        if not candidates:
            return {'apartment': None, 'status': 'not_found',
                    'message': 'Новых квартир под это ограничение нет.'}
        apartment = self.rng.choice(candidates)
        self.seen.add(apartment['id'])
        if apartment['price'] > MAX_PRICE:
            return {'apartment': None, 'status': 'price_above_limit',
                    'rejected_apartment': apartment, 'price_limit': MAX_PRICE,
                    'message': 'Квартира не подходит: цена выше допустимого лимита покупки.'}
        return {'apartment': apartment, 'status': 'found', 'price_limit': MAX_PRICE, 'mock': True}

    def check_mortgage(self, apartment_id):
        apartment = next((r for r in self.catalog if r['id'] == apartment_id), None)
        if apartment is None:
            raise ValueError('Неизвестный ID квартиры')
        if apartment['price'] > MAX_PRICE:
            raise ValueError('Квартира дороже 15 млн ₽: не подходит по цене, проверка банка не выполняется')
        if self.bank_limit is None:
            self.bank_limit = self.rng.randint(min(r['price'] for r in self.catalog), MAX_PRICE)
        return {'bank': 'Нева (вымышленный)', 'apartment_id': apartment_id,
                'price': apartment['price'], 'limit': self.bank_limit,
                'affordable': apartment['price'] <= self.bank_limit,
                'shortfall': max(0, apartment['price'] - self.bank_limit), 'mock': True}

    def execute(self, name, arguments):
        """Только разрешённые функции; ошибки аргументов возвращаются модели."""
        try:
            args = json.loads(arguments)
            if not isinstance(args, dict):
                raise ValueError('Аргументы должны быть JSON-объектом')
            if name == 'choose_apartment':
                if set(args) - {'max_price'}:
                    raise ValueError('Неизвестные аргументы')
                price = args.get('max_price')
                if 'max_price' in args and (type(price) is not int or price <= 0
                                           or not math.isfinite(price)):
                    raise ValueError('max_price должен быть положительным целым числом')
                return self.choose_apartment(**args)
            if name == 'check_mortgage':
                if set(args) != {'apartment_id'} or not isinstance(args['apartment_id'], str):
                    raise ValueError('Нужен строковый apartment_id')
                return self.check_mortgage(**args)
            raise ValueError('Неизвестный инструмент')
        except (ValueError, TypeError, OverflowError) as error:
            return {'error': str(error)}
