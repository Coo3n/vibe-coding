"""Воспроизводимый учебный каталог: 6 контрольных + 600 сгенерированных путёвок."""

from datetime import date, timedelta
import json
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parent.parent
CITIES = ['Москва', 'Санкт-Петербург', 'Казань', 'Екатеринбург', 'Новосибирск']
DESTINATIONS = [
    ('Турция', ['Анталья', 'Сиде', 'Кемер', 'Аланья'], 'AI', 0),
    ('Египет', ['Хургада', 'Шарм-эль-Шейх'], 'AI', 10000),
    ('Россия', ['Сочи', 'Анапа', 'Геленджик'], 'BB', -20000),
    ('Таиланд', ['Пхукет', 'Паттайя'], 'BB', 35000),
    ('ОАЭ', ['Дубай', 'Шарджа', 'Рас-эль-Хайма'], 'HB', 20000),
    ('Шри-Ланка', ['Хиккадува', 'Бентота'], 'HB', 25000),
]


def generate_catalog():
    rng = random.Random(707)
    tours = json.loads((ROOT / 'data' / 'tours.base.json').read_text(encoding='utf-8'))
    # По шесть отелей на один набор условий: выдачи содержат сравнимые варианты.
    for group in range(100):
        city = CITIES[group % len(CITIES)]
        country, resorts, default_meal, premium = DESTINATIONS[group % len(DESTINATIONS)]
        resort = rng.choice(resorts)
        departure = date(2027, 6, 1) + timedelta(days=(group * 7) % 120)
        nights = rng.choice([5, 7, 10, 14])
        adults = rng.choice([1, 2, 2, 2, 3, 4])
        children = rng.choice([0, 0, 1, 2])
        for hotel in range(6):
            stars = 3 + hotel // 2
            meal = default_meal if hotel % 2 == 0 else rng.choice(['BB', 'HB', 'AI'])
            features = ['пляжный', 'спокойный']
            if hotel % 2 == 0:
                features.append('первая линия')
            if hotel in (2, 3, 5):
                features.append('активный')
            if hotel in (1, 3, 5):
                features.append('экскурсионный')
            if children:
                features.append('для детей')
            if country in ('Турция', 'Россия', 'Таиланд', 'ОАЭ'):
                features.append('без визы')
            price = max(35000, adults * 30000 + children * 15000 + nights * 4500
                        + premium + (stars - 3) * 25000 + hotel * 3000
                        + {'BB': 0, 'HB': 10000, 'AI': 25000}[meal]
                        + CITIES.index(city) * 5000)
            tours.append({
                'id': f'DEMO-{group * 6 + hotel + 1:03}',
                'hotel': f'Учебный отель «{resort} {group + 1}-{hotel + 1}»',
                'departure_city': city, 'country': country, 'resort': resort,
                'departure_date': departure.isoformat(), 'nights': nights,
                'adults': adults, 'children': children, 'child_age_min': 0, 'child_age_max': 17,
                'price_total': price, 'stars': stars, 'meal': meal,
                'features': features, 'mock': True,
            })
    return tours


if __name__ == '__main__':
    catalog = generate_catalog()
    output = ROOT / 'data' / 'tours.json'
    output.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'{len(catalog)} учебных путёвок → {output}')
