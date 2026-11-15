"""Readable comparison, backed by report.json including all actual requests."""

from pathlib import Path

from .context import atomic_json


def number(value):
    return '?' if value is None else f'{value:,}'.replace(',', ' ')


def cell(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def write_report(report, directory):
    directory = Path(directory)
    turns = report['settings'].get('turns', 15)
    atomic_json(directory / 'report.json', report)
    rows = ['# День 9 · Овощной онлайн-прилавок', '',
            f"Статус: **{report['status']}**. Модель API: `{report['model']}`.", '',
            '## Методика', '', report['method'], '',
            f'План: {turns} пользовательских сообщений + {turns} финальных ответов = {turns * 2} видимых сообщений. '
            'Tool calls и результаты — дополнительные служебные записи. '
            'Числа N и порции относятся к видимым сообщениям; служебные записи сохраняются вместе с ходом.', '',
            f"Фактически завершено ходов: {sum(row.get('turn') is not None for row in report['dialogue'])}/{turns}.", '',
            'Ответы на вопросы не добавляются в контекст других проверок. Эталонные значения '
            'не передаются модели. Каталог учебный, факты о договорённостях инструменты не хранят.', '']
    if report.get('error'):
        rows += ['**Ошибка / неполный прогон:** ' + report['error'], '']
    if 'counts' in report:
        counts = report['counts']
        rows += [f"Сжатий: {counts['compressions']}; записей полной истории: {counts['full_records']}; "
                 f"в хвосте: {counts['compressed_tail_records']} записей / "
                 f"{counts['compressed_tail_visible_messages']} видимых сообщений.", '']
    rows += ['## Память', '', '| Деталь | Эталон | Без сжатия | Со сжатием |',
             '|---|---|---|---|']
    probes = {mode: {row['fact_id']: row for row in report['probes'][mode]} for mode in ('full', 'compressed')}
    for fact in report['facts']:
        values = []
        for mode in ('full', 'compressed'):
            evaluation = probes[mode].get(fact['id'], {}).get('evaluation')
            if evaluation is None:
                values.append('— не проверено')
            else:
                mark = {'correct': '✓', 'missing': 'не вспомнил', 'mismatch': '≠ требует сверки'}[evaluation['status']]
                values.append(f"{cell(evaluation['value'])} · {mark}")
        rows.append(f"| {fact['label']} | {cell(fact['expected'])} | {values[0]} | {values[1]} |")
    rows += ['', 'Оценка: нормализованное точное совпадение с заранее заданными допустимыми значениями. '
             'Другие формулировки помечаются «требует сверки», а не автоматически объявляются забытыми.', '']
    for mode, label in (('full', 'Без сжатия'), ('compressed', 'Со сжатием')):
        score = report['score'][mode]
        rows.append(f"- {label}: **{score['correct']}/{score['total']}**; выполнено {score['checked']} проверок.")
    lost = report['lost_after_compression']
    rows += ['', 'Детали, прошедшие без сжатия, но не прошедшие со сжатием: ' + (', '.join(lost) if lost else 'не обнаружены в выполненных проверках') + '.',
             '', '## Фактические токены API', '', '| Этап | Запросов | Вход | Выход | Всего |', '|---|---:|---:|---:|---:|']
    labels = [('preparation', 'Общая подготовка диалога, включая tools'), ('summary', 'Создание summary'),
              ('full_probes', '20 проверок без сжатия'), ('compressed_probes', '20 проверок со сжатием'),
              ('full_comparison', 'Сравнение: полная история'),
              ('compressed_comparison', 'Сравнение: сжатая история + summary'),
              ('full_total', 'Полный режим с общей подготовкой'),
              ('compressed_total', 'Сжатый режим с общей подготовкой'), ('actual_run', 'Реально потрачено на весь эксперимент')]
    for key, label in labels:
        usage = report['usage'][key]
        rows.append(f"| {label} | {usage['requests']} | {number(usage['prompt_tokens'])} | "
                    f"{number(usage['completion_tokens'])} | {number(usage['total_tokens'])} |")
    saving = report['savings']
    rows += ['', f"Экономия на проверках **с учётом summary**: {number(saving['tokens'])} токенов "
             f"({saving['percent'] if saving['percent'] is not None else '?'}%).", '',
             'Это измерение повторных запросов к одному готовому диалогу, а не экономии '
             'при генерации второго независимого диалога. Общая подготовка одинакова в обоих итогах. '
             'Два итоговых режима перекрываются по подготовке; их нельзя складывать. '
             'Отдельная строка показывает реально потраченные токены.', '',
             '`?` означает неизвестный usage, не ноль. Кэшированные токены уже входят во вход. '
             'Стоимость в деньгах не рассчитывается. Один прогон не доказывает сохранение любых фактов.', '',
             '## Сжатия', '']
    for index, event in enumerate(report['compressions'], 1):
        rows += [f"### Summary {index} · после {event['after_visible_messages']} сообщений", '']
        for key, title in (('facts', 'Факты'), ('decisions', 'Решения'), ('open_questions', 'Открытые вопросы')):
            rows.append(f'**{title}:**')
            rows.extend('- ' + item for item in (event.get('summary') or {}).get(key, []))
            rows.append('')
    rows += ['Полные JSON-запросы, ответы, usage и состояния: `report.json`.', '']
    (directory / 'report.md').write_text('\n'.join(rows), encoding='utf-8')
