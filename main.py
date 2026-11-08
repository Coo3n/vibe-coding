"""День 5: одинаковые задачи на трёх моделях."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from http.client import HTTPException
import json
from pathlib import Path
from queue import Empty, Queue
import statistics
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from grading import score_answer
from pricing import PRICING, estimate_cost


MODELS = [
    {'route': 'llm-small', 'expected_model': 'Qwen/Qwen3.8-27B-FP8',
     'display_name': 'Qwen 3.8 27B'},
    {'route': 'llm-medium', 'expected_model': 'deepseek-ai/DeepSeek-V4-Flash-0731',
     'display_name': 'DeepSeek V4 Flash'},
    {'route': 'coder', 'expected_model': 'glm-53-flash',
     'display_name': 'GLM 5.3 Flash'},
]
REASONING_EFFORT = 'medium'
REASONING_NOTE = ('Поддержка запрошенного уровня зависит от модели и конфигурации API. '
                  'Параметр отправляется буквально, без автоматической замены. '
                  'Приём поля API и расход reasoning-токенов не доказывают одинаковую '
                  'интерпретацию запрошенного уровня всеми моделями.')
LIMITATION = ('Стоимость — сценарная оценка по публичным тарифам, не фактическое списание. '
              'Qwen: тариф размещённой 27B как ориентир для FP8; DeepSeek: исторический off-peak; '
              'GLM: предполагается соответствие ID официальной Flash. '
              'GLM Flash не является флагманом; сравнение с флагманом остаётся за рамками прогона.')


def token_value(usage, field):
    if not isinstance(usage, dict):
        return None
    value = usage.get(field)
    if field == 'reasoning_tokens' and value is None:
        details = usage.get('completion_tokens_details')
        value = details.get(field) if isinstance(details, dict) else None
    return value if type(value) is int and value >= 0 else None


def summarize(records, models, task_count):
    summaries = []
    for model in models:
        rows = [r for r in records if r['route'] == model['route']]
        accepted = [r for r in rows if r.get('eligible') and not r.get('error')]
        times = [r['seconds'] for r in rows if isinstance(r.get('seconds'), (int, float))]
        tokens = {}
        for field in ('prompt_tokens', 'completion_tokens', 'reasoning_tokens', 'total_tokens'):
            values = [token_value(r.get('usage'), field) for r in rows]
            tokens[field] = sum(values) if values and None not in values else None
        known = [token_value(r.get('usage'), 'total_tokens') for r in rows]
        correct = sum(r.get('correct') is True for r in accepted)
        summaries.append({
            **model, 'attempts': len(rows), 'task_count': task_count,
            'responses': sum(r.get('response_received', not r.get('error')) for r in rows),
            'api_errors': sum(bool(r.get('error')) for r in rows),
            'excluded': sum(not r.get('eligible') and not r.get('error') for r in rows),
            'format_errors': sum(not r.get('format_valid') for r in accepted),
            'compared': len(accepted), 'correct': correct,
            'accuracy': correct / task_count if task_count else None,
            'mean_seconds': statistics.mean(times) if times else None,
            'median_seconds': statistics.median(times) if times else None,
            'sum_seconds': sum(times), **tokens,
            'known_total_tokens': sum(v for v in known if v is not None),
            'usage_missing': sum(v is None for v in known),
            'cost': estimate_cost(model['route'], tokens['prompt_tokens'], tokens['completion_tokens']),
            'cost_basis': 'public_price_estimate',
        })
    return summaries


def load_settings(path):
    config = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            key, value = line.split('=', 1)
            config[key.strip()] = value.strip().strip('\"\'')
    for name in ('LLM_API_URL', 'LLM_API_KEY'):
        if not config.get(name):
            raise ValueError(f'Заполните {name} в .env')
    return config


def api_request(config, payload=None):
    url = config['LLM_API_URL']
    if payload is None:
        url = url.removesuffix('/chat/completions') + '/models'
    request = Request(url, data=None if payload is None else json.dumps(payload).encode(),
                      headers={'Authorization': 'Bearer ' + config['LLM_API_KEY'],
                               'Content-Type': 'application/json'})
    with urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode('utf-8'))


def build_payload(route, task, effort=REASONING_EFFORT):
    return {'model': route, 'temperature': 0, 'reasoning_effort': effort,
            'messages': [{'role': 'user', 'content':
        task['prompt'] + '\n\nВерни только JSON-объект с единственным ключом "answer" '
        'и буквой выбранного варианта "A", "B", "C" или "D". '
        'Без Markdown, пояснений и текста вне JSON.'}]}


def run_case(config, model, task, request_fn=api_request, effort=REASONING_EFFORT):
    payload = build_payload(model['route'], task, effort)
    record = {'route': model['route'], 'expected_model': model['expected_model'],
              'task_id': task['id'], 'expected_answer': task['answer'], 'request': payload,
              'response_received': False, 'eligible': False, 'correct': False}
    started = time.perf_counter()
    try:
        data = request_fn(config, payload)
        record.update(response_received=True, raw_response=data)
        if isinstance(data, dict):
            record.update(usage=data.get('usage'), response_model=data.get('model'))
        choice = data['choices'][0]
        content = choice['message'].get('content')
        content = '' if content is None else content
        if not isinstance(content, str):
            raise ValueError('Нет текстового content')
        record.update(content=content, finish_reason=choice.get('finish_reason'))
        if record['response_model'] != model['expected_model']:
            record['exclusion_reason'] = 'Фактический ID модели не совпадает с каталогом'
        else:
            record['eligible'] = True
            record.update(score_answer(content, task['answer'], choice.get('finish_reason')))
    except HTTPError as error:
        record['error'] = f'API HTTP {error.code}'
        error.close()
    except (URLError, OSError, TimeoutError, HTTPException):
        record['error'] = 'Ошибка соединения или таймаут API'
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        record['error'] = 'Повреждённая структура ответа API'
        record.update(eligible=False, correct=False)
    record['seconds'] = time.perf_counter() - started
    return record


def render_report(report):
    def number(value):
        return 'нет данных' if value is None else f'{value:.2f}'

    def tokens(value):
        return 'нет данных' if value is None else str(value)

    def money(value):
        return 'нет данных' if value is None else f'${value:.8f}'

    rows = report['summary']
    tasks = report['dataset']['tasks']
    effort = report.get('method', {}).get('reasoning_setting')
    effort_description = (f'Запрошен reasoning_effort={effort} у всех моделей. '
                          + REASONING_NOTE if effort else 'Уровень reasoning_effort клиентом не задан.')
    source_count = report['dataset'].get('source_task_count', len(tasks))
    task_description = (f'{len(tasks)} задач: {source_count} неизменённых из `task-3`'
                        + (f' и {len(tasks) - source_count} добавленных для Дня 5'
                           if len(tasks) > source_count else '') + '.')
    lines = ['# День 5 · Версии моделей', '']
    if report.get('video'):
        lines += [f"[Видео запуска]({report['video']})", '']
    lines += ['> ' + LIMITATION, '',
             f"Запуск: {report['started']} · завершён: {'да' if report['completed'] else 'нет'} · "
             f"ответов/попыток: {len(report['records'])}/{len(tasks) * len(report['models'])}", '',
             effort_description, '',
             '| Модель | Верно | Среднее / медиана, с | Сумма времени, с | Вход | Выход | Всего токенов | Расчётная стоимость, USD |',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    for row in rows:
        lines.append(f"| {model_name(row['route'], row['expected_model'])} | {score_display(row)} | "
                     f"{number(row['mean_seconds'])} / {number(row['median_seconds'])} | "
                     f"{number(row['sum_seconds'])} | {tokens(row['prompt_tokens'])} | {tokens(row['completion_tokens'])} | "
                     f"{tokens(row['total_tokens'])} | {money(row['cost'])} |")
    lines += ['', '## Публичные тарифы для расчёта', '',
              'Проверены 08.11.2026. USD за 1 млн токенов; вход без кэша. '
              'Это сценарный расчёт, не подтверждённые списания.', '',
              '| Модель | Вход / 1M | Выход / 1M | Источник и допущение |',
              '|---|---:|---:|---|']
    for model in report['models']:
        rate = PRICING['rates'].get(model['route'])
        if rate:
            lines.append(f"| {rate['public_model']} | ${rate['input_per_million']:.2f} | "
                         f"${rate['output_per_million']:.2f} | [Прайс]({rate['source']}). {rate['note']} |")
    lines += ['', '`Стоимость = (вход × тариф входа + выход × тариф выхода) / 1 000 000`.',
              'Reasoning уже включён в выход и повторно не оплачивается в этом расчёте. '
              'Для DeepSeek в peak-сценарии оценка вдвое выше.', '']
    lines += ['', '## Надёжность и скрытое рассуждение', '',
              '| Модель | Сбои API/конверта | Исключено: другая модель | Ошибки формата | Reasoning-токены | Вызовы без total_tokens |',
              '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {model_name(row['route'])} | {row['api_errors']} | {row['excluded']} | {row['format_errors']} | "
                     f"{tokens(row['reasoning_tokens'])} | {row['usage_missing']} |")
    lines += ['', '## По задачам', '', '| Задача | Эталон | ' + ' | '.join(model_name(m['route']) for m in report['models']) + ' |',
              '|---|---|' + '---|' * len(rows)]
    for task in tasks:
        cells = []
        for model in report['models']:
            record = next((r for r in report['records'] if r['task_id'] == task['id']
                           and r['route'] == model['route']), None)
            cells.append('ожидание' if record is None else '✓' if record.get('correct')
                         and record.get('eligible') else 'ошибка API' if record.get('error')
                         else 'исключено' if not record.get('eligible')
                         else 'сбой формата' if not record.get('format_valid') else 'неверно')
        lines.append(f"| {task['id']} | {task['answer']} | " + ' | '.join(cells) + ' |')
    lines += ['', '## Выводы', '']
    if report['completed']:
        small = rows[0]
        passed = [r['task_id'] for r in report['records'] if r['route'] == small['route']
                  and r.get('eligible') and r.get('correct')]
        if small['compared']:
            lines.append(f"- Малая модель решила {small['correct']}/{len(tasks)} задач. "
                         'На этом наборе её хватило для: ' + (', '.join(passed) or 'ни одной задачи') + '.')
        else:
            lines.append('- От малой модели нет допущенных ответов; ошибки API не являются неверными решениями.')
        best = max(rows, key=lambda r: r['correct'])
        lines.append(f"- Максимальный результат: {best['correct']}/{len(tasks)}; "
                      'его показали ' + ', '.join(model_name(r['route']) for r in rows if r['correct'] == best['correct']) + '.')
        third = rows[-1]
        comparison_ready = all(row['compared'] == len(tasks) and not row['api_errors']
                               and not row['excluded'] for row in (small, third))
        if comparison_ready:
            delta = (third['accuracy'] - small['accuracy']) * 100
            lines.append(f'- Третья модель против малой: разница точности {delta:+.1f} п.п.; '
                         'это не сравнение подтверждённого флагмана с малой моделью.')
        else:
            lines.append('- Полное сравнение точности с малой моделью недоступно: есть сбои API или исключённые ответы.')
        if comparison_ready and small['cost'] and third['cost'] is not None:
            lines.append(f"- По выбранным публичным тарифам третья модель стоила бы в "
                         f"{third['cost'] / small['cost']:.2f} раза от стоимости малой за весь набор "
                         f"({money(third['cost'])} против {money(small['cost'])}). "
                         'Это сценарная оценка Flash, не цена флагмана.')
    else:
        lines.append('Прогон не завершён: итоговый рейтинг пока не формируется.')
    lines += ['', '## Методика и ограничения', '',
              '- ' + task_description + ' Один ответ каждой модели на задачу; эталоны не передаются.',
              '- Одинаковый прямой промпт, temperature=0, без истории, seed и max_tokens клиента.',
              '- JSON должен содержать только answer: A–D. Пустые, обрезанные и неформатные ответы не засчитываются.',
              '- При смене ID модели ответ исключается из оценки; затраченные токены сохраняются.',
              '- Три параллельные очереди, внутри каждой — последовательный запуск. Конкуренция за ресурсы API может влиять на время.',
              '- Время — до полного ответа, включая сеть, не TTFT. Сумма времени не равна длительности параллельного прогона.',
              '- Выходные токены могут включать reasoning; отдельно к total они не прибавляются. Отсутствующий usage — не ноль.',
              '- Реальные настройки reasoning могут различаться: результат отражает конфигурации API, а не только размер модели.',
              f'- Без повторов и автоматических ретраев; {len(tasks)} задач — малая выборка. Равная точность не доказывает равенство моделей.',
              '- Размер GLM/DeepSeek и положение в линейках не подтверждены каталогом: сравниваем доступные маршруты, не число параметров.',
              '', '## Исходные ответы', '']
    for task in tasks:
        lines += [f"### {task['id']}", '', task['prompt'], '']
        for record in report['records']:
            if record['task_id'] == task['id']:
                lines += [f"**{model_name(record['route'])}** · {record['seconds']:.2f} с · "
                          f"finish_reason={record.get('finish_reason', 'нет')} · "
                          f"модель={record.get('response_model', 'нет')}", '',
                          '```text', record.get('content') or record.get('error') or '(пустой ответ)', '```', '']
    return '\n'.join(lines)


def save_report(directory, report):
    report.update(pricing=PRICING, limitation=LIMITATION)
    report['summary'] = summarize(report['records'], report['models'], len(report['dataset']['tasks']))
    for name, text in [('comparison.json', json.dumps(report, ensure_ascii=False, indent=2)),
                       ('comparison.md', render_report(report))]:
        target = directory / name
        staging = directory / (name + '.tmp')
        staging.write_text(text, encoding='utf-8')
        staging.replace(target)


def model_name(route, fallback=None):
    return next((model['display_name'] for model in MODELS if model['route'] == route),
                fallback or route)


def score_display(row):
    return f"{row['correct']}/{row['task_count']}" if row['compared'] else 'нет ответов'


def format_progress(record, index, total):
    status = ('✓' if record.get('correct') and record.get('eligible') else
              record.get('error') or record.get('exclusion_reason') or
              ('неверно' if record.get('format_valid') else 'сбой формата'))
    usage = record.get('usage')
    counts = [token_value(usage, field) for field in
              ('prompt_tokens', 'completion_tokens', 'total_tokens')]
    tokens = '/'.join('?' if value is None else str(value) for value in counts)
    cost = estimate_cost(record['route'], counts[0], counts[1])
    price = 'нет данных' if cost is None else f'${cost:.8f}'
    return (f"{index:02}/{total} · {model_name(record['route']):17} · {record['task_id']} · {status} · "
            f"{record['seconds']:.2f} с · токены: {tokens} · стоимость: {price}")


def render_terminal_summary(report):
    def number(value):
        return 'нет данных' if value is None else f'{value:.2f}'

    header = ['Модель', 'Верно', 'Сбои API', 'Среднее, с', 'Всего, с', 'Токены', 'Оценка, USD']
    table = [header]
    for row in report['summary']:
        table.append([model_name(row['route'], row['expected_model']), score_display(row), str(row['api_errors']),
                      number(row['mean_seconds']), number(row['sum_seconds']),
                      'нет данных' if row['total_tokens'] is None else str(row['total_tokens']),
                      'нет данных' if row['cost'] is None else f"${row['cost']:.8f}"])
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    lines = ['ИТОГОВАЯ ТАБЛИЦА']
    for i, row in enumerate(table):
        lines.append(' | '.join(cell.ljust(width) for cell, width in zip(row, widths)))
        if i == 0:
            lines.append('-+-'.join('-' * width for width in widths))
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument('--env', type=Path, default=root / '.env')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--effort', choices=('medium', 'high'), default=REASONING_EFFORT,
                        help='Уровень рассуждений для всех моделей (по умолчанию medium)')
    args = parser.parse_args()
    config = load_settings(args.env)
    dataset = json.loads((root / 'tasks.json').read_text(encoding='utf-8'))
    catalog = api_request(config)
    available = {m['id']: m.get('real_model') for m in catalog['data']}
    for model in MODELS:
        if available.get(model['route']) != model['expected_model']:
            parser.error(f"Каталог моделей изменился для {model['route']}; обновите конфигурацию явно")
    directory = args.output_dir or root / 'results' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    directory.mkdir(parents=True, exist_ok=False)
    report = {'started': datetime.now().astimezone().isoformat(), 'completed': False,
              'models': MODELS, 'catalog': catalog, 'dataset': dataset, 'records': [],
              'method': {'temperature': 0, 'parallel_model_queues': 3, 'repeats': 1,
                          'client_token_limit': None, 'reasoning_setting': args.effort,
                          'reasoning_support': 'unverified', 'reasoning_note': REASONING_NOTE},
              'pricing': PRICING, 'limitation': LIMITATION}
    save_report(directory, report)
    events = Queue()

    def run_model(model):
        for task in dataset['tasks']:
            events.put(run_case(config, model, task, effort=args.effort))

    started = time.perf_counter()
    print(f'ДЕНЬ 5 · 3 модели × {len(dataset["tasks"])} задач · temperature=0 · '
          f'reasoning_effort={args.effort}', flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run_model, model) for model in MODELS]
        expected = len(MODELS) * len(dataset['tasks'])
        for _ in range(expected):
            # Обнаруживаем сбой worker, не зависая в ожидании потерянной записи.
            while True:
                try:
                    record = events.get(timeout=1)
                    break
                except Empty:
                    if all(f.done() for f in futures):
                        for future in futures:
                            future.result()
                        raise RuntimeError('Очередь завершилась без всех результатов')
            report['records'].append(record)
            report['wall_seconds'] = time.perf_counter() - started
            save_report(directory, report)
            print(format_progress(record, len(report['records']), expected), flush=True)
        for future in futures:
            future.result()
    report.update(completed=True, finished=datetime.now().astimezone().isoformat(),
                  wall_seconds=time.perf_counter() - started)
    save_report(directory, report)
    print('\n' + render_terminal_summary(report), flush=True)
    print(f'Отчёт: {directory / "comparison.md"}', flush=True)
    return 1 if any(r.get('error') or not r.get('eligible') for r in report['records']) else 0


if __name__ == '__main__':
    raise SystemExit(main())
