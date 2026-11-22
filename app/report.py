"""JSON и Markdown с наблюдаемыми результатами прогона."""

import json


def initial_memory_sections(memory):
    return (
        ('Актуальные долговременные факты',
         [f'{key} — {value}' for key, value in memory['active'].items()]),
        ('Накопительные записи',
         [f'{key} — {value}' for key, values in memory['history'].items() for value in values]),
        ('Рабочая память',
         [f'{key} — {value}' for key, value in memory['working'].items()]),
        ('Краткосрочная память',
         [f"Сообщений в диалоге — {memory['short_term_count']}"]),
    )


def load_report(path):
    if path.is_dir():
        path = path / 'report.json'
    report = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(report, dict) or not isinstance(report.get('steps'), list):
        raise ValueError('Некорректный файл отчёта Дня 15.')
    return report


def write_report(report, directory):
    (directory / 'report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    checks = report['checks']
    total = len(report['steps'])
    lines = [
        '# День 15 · устаревание и конфликты памяти',
        '',
        f"Клиент — {report['model']} ({report['mode']}). Окно — {report['window']} сообщений.",
        '',
        '## Исходные данные в памяти',
        '',
    ]
    for title, entries in initial_memory_sections(report['initial_memory']):
        lines.extend([f'**{title}**', '', *(f'- {entry}' for entry in entries), ''])
    lines += [
        '## Результаты',
        '',
        '| Факт | Реплика изменения | Контрольный вопрос | Ответ |',
        '|---|---|---|---|',
    ]
    for step in report['steps']:
        lines.append(f"| {step['key']} | {step['statement']} | {step['question']} | "
                     f"{step['answer'].replace(chr(10), '<br>')} |")
    usage = report['usage']
    lines += [
        '',
        f"- Актуальное значение в ответе — {checks['correct_answers']}/{total}.",
        f"- Устаревшее значение в ответе — {checks['stale_answers']}/{total}.",
        f"- Устаревшее значение в запросе к клиенту — {checks['stale_prompts']}/{total}.",
        f"- Замен с записью старого и нового значений — {checks['replacements']}.",
        f"- Накопительных решений сохранено — {checks['decisions']}.",
        f"- Запоздалая запись отклонена — {'да' if checks['late_write_ignored'] else 'нет'}.",
        f"- Запись с равной отметкой отклонена — {'да' if checks['tied_write_ignored'] else 'нет'}.",
        f"- Рабочих записей до и после завершения — {checks['working_before_finish']} "
        f"→ {checks['working_after_finish']}.",
        f"- Долговременное состояние после завершения и повторного чтения совпало — "
        f"{'да' if checks['durable_unchanged'] else 'нет'}.",
        f"- Обращений к модели — {usage['model_calls']}. "
        f"Ответов локального клиента — {usage['offline_answers']}. "
        f"Операций с памятью — {usage['memory_actions']}.",
        f"- Токены модели — {usage['prompt_tokens']} входных, {usage['completion_tokens']} выходных, "
        f"всего {usage['total_tokens']}.",
        '',
        'Полные запросы и журнал замен доступны в `report.json`. '
        'Состояние на диске доступно в `memory.json`.',
        '',
        'Проверка ищет буквальные совпадения со значениями из сценария. '
        'Отдельно учитываются ответы клиента и операции записи, отказа и очистки, '
        'которые выполняются приложением без вызовов модели.',
        '',
    ]
    (directory / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
