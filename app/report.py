"""Readable comparison, backed by report.json including all actual requests."""

from pathlib import Path

from .context import atomic_json
from .tokens import PHASES, progress_rows


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
             '', '## Два случая: без сжатия и со сжатием', '',
             '| Показатель | 1. Без сжатия | 2. Со сжатием |', '|---|---:|---:|',
             '| Контекст каждого вопроса | Вся переписка | Summary + хвост |']
    for key, label in (('prompt_tokens', 'Входные токены'), ('completion_tokens', 'Выходные токены'),
                       ('total_tokens', 'Всего токенов')):
        rows.append(f"| {label} | {number(report['usage']['full_comparison'][key])} | "
                    f"{number(report['usage']['compressed_comparison'][key])} |")
    rows += [f"| Из них создание summary | 0 | {number(report['usage']['summary']['total_tokens'])} |", '',
             '**Summary уже включено в итог второго случая.** Это не отдельный режим и не дополнительная сумма к итогу.']
    saving = report['savings']
    rows += ['', f"Экономия на проверках **с учётом summary**: {number(saving['tokens'])} токенов "
             f"({saving['percent'] if saving['percent'] is not None else '?'}%).", '',
             'Это измерение повторных запросов к одному готовому диалогу, а не экономии '
             'при генерации второго независимого диалога. '
             f"Общая подготовка переписки: {number(report['usage']['preparation']['total_tokens'])} токенов; "
             'она выполнена один раз до сравнения и не включена в таблицу. '
             f"Всего реально потрачено на запуск: {number(report['usage']['actual_run']['total_tokens'])} токенов. "
             'Подробный учёт этапов находится в report.json.', '',
             '`?` означает неизвестный usage, не ноль. Кэшированные токены уже входят во вход. '
             'Стоимость в деньгах не рассчитывается. Один прогон не доказывает сохранение любых фактов.', '',
             '## Динамика токенов по каждому API-запросу', '',
             'Δ входа — изменение размера входа относительно предыдущего запроса **того же этапа**. '
             'Расход запроса — вход + выход. Накопленный расход не является размером контекста.', '',
             '| API № | Этап | Вход | Δ входа | Выход | Расход запроса | Накоплено на этапе | Накоплено за запуск |',
             '|---:|---|---:|---:|---:|---:|---:|---:|']
    for progress in progress_rows(report):
        delta = progress['input_delta']
        change = '—' if delta is None else ('+' if delta >= 0 else '−') + number(abs(delta))
        rows.append(f"| {progress['request_number']} | {PHASES[progress['phase']]} | "
                    f"{number(progress['prompt_tokens'])} | {change} | {number(progress['completion_tokens'])} | "
                    f"{number(progress['request_tokens'])} | {number(progress['phase_total'])} | "
                    f"{number(progress['run_total'])} |")
    rows += ['', '## Сжатия', '']
    for index, event in enumerate(report['compressions'], 1):
        rows += [f"### Summary {index} · после {event['after_visible_messages']} сообщений", '']
        for key, title in (('facts', 'Факты'), ('decisions', 'Решения'), ('open_questions', 'Открытые вопросы')):
            rows.append(f'**{title}:**')
            rows.extend('- ' + item for item in (event.get('summary') or {}).get(key, []))
            rows.append('')
    rows += ['Полные JSON-запросы, ответы, usage и состояния: `report.json`.', '']
    (directory / 'report.md').write_text('\n'.join(rows), encoding='utf-8')
