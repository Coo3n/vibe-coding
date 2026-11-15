"""Read-only teaching catalog and deterministic procurement arithmetic."""

from decimal import Decimal

from jsonschema import Draft202012Validator

from .protocol import SCHEMAS, object_schema


PRODUCTS = [
    {'id': 'potato', 'name': 'Картофель', 'purchase_rub_per_kg': 38, 'stock_kg': 300},
    {'id': 'carrot', 'name': 'Морковь', 'purchase_rub_per_kg': 46, 'stock_kg': 180},
    {'id': 'onion', 'name': 'Лук репчатый', 'purchase_rub_per_kg': 42, 'stock_kg': 200},
    {'id': 'cabbage', 'name': 'Капуста', 'purchase_rub_per_kg': 35, 'stock_kg': 150},
    {'id': 'tomato', 'name': 'Помидоры', 'purchase_rub_per_kg': 140, 'stock_kg': 80},
    {'id': 'cucumber', 'name': 'Огурцы', 'purchase_rub_per_kg': 110, 'stock_kg': 90},
]

ARGUMENTS = {
    'get_products': object_schema({}),
    'calculate_batch': object_schema({
        'items': {'type': 'array', 'minItems': 1, 'maxItems': 6,
                  'items': object_schema({'product_id': {'type': 'string', 'enum': [p['id'] for p in PRODUCTS]},
                                          'kg': {'type': 'number', 'exclusiveMinimum': 0, 'maximum': 10000}})},
        'packaging_rub': {'type': 'number', 'minimum': 0, 'maximum': 1000000},
        'delivery_rub': {'type': 'number', 'minimum': 0, 'maximum': 1000000},
    }),
}

DESCRIPTIONS = {
    'get_products': 'Получить учебный каталог овощей: ID, закупочные цены за кг и остатки. Ничего не заказывает.',
    'calculate_batch': 'Точно посчитать закупку по ID и кг, плюс упаковка и доставка всей партии. '
                       'Проверяет остатки. Не оформляет заказ. Все расходы бери из условий пользователя.',
}

TOOL_SCHEMAS = [{'type': 'function', 'function': {'name': name, 'description': DESCRIPTIONS[name],
                                               'strict': True, 'parameters': schema}}
                for name, schema in ARGUMENTS.items()]
TOOL_SCHEMAS.append({'type': 'function', 'function': {
    'name': 'respond_to_user', 'strict': True, 'parameters': SCHEMAS['chat'],
    'description': 'Завершить ход ответом пользователю в JSON. Сначала выполни нужные рабочие инструменты, '
                   'затем передай ответ, вопросы и следующий шаг.'}})


def execute(name, arguments):
    if name not in ARGUMENTS:
        return {'error': 'unknown_tool', 'message': 'Такого инструмента нет.'}
    if not Draft202012Validator(ARGUMENTS[name]).is_valid(arguments):
        return {'error': 'invalid_arguments', 'message': 'Аргументы не соответствуют JSON Schema инструмента.'}
    if name == 'get_products':
        return {'source': 'Учебный каталог, не реальные предложения', 'currency': 'RUB', 'products': PRODUCTS}
    catalog = {p['id']: p for p in PRODUCTS}
    seen, lines = set(), []
    subtotal = Decimal(0)
    for item in arguments['items']:
        identifier = item['product_id']
        if identifier in seen:
            return {'error': 'duplicate_product', 'message': 'Объедините количество одного товара в одну строку.'}
        seen.add(identifier)
        product = catalog[identifier]
        kg = Decimal(str(item['kg']))
        if kg > product['stock_kg']:
            return {'error': 'insufficient_stock', 'product_id': identifier, 'available_kg': product['stock_kg']}
        amount = (kg * product['purchase_rub_per_kg']).quantize(Decimal('0.01'))
        subtotal += amount
        lines.append({'product_id': identifier, 'kg': item['kg'], 'amount_rub': float(amount)})
    packaging = Decimal(str(arguments['packaging_rub'])).quantize(Decimal('0.01'))
    delivery = Decimal(str(arguments['delivery_rub'])).quantize(Decimal('0.01'))
    return {'source': 'Учебный расчёт; заказ не размещён', 'currency': 'RUB', 'items': lines,
            'goods_rub': float(subtotal), 'packaging_rub': float(packaging), 'delivery_rub': float(delivery),
            'total_rub': float(subtotal + packaging + delivery)}
