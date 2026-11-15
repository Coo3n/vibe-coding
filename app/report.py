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
    case = report.get('case', 'both')
    modes = ('full', 'compressed') if case == 'both' else (case,)
    labels = {'full': 'Без сжатия', 'compressed': 'Со сжатием'}
    atomic_json(directory / 'report.json', report)
    rows = ['# День 9 · Овощной онлайн-прилавок', '',
            f"Статус: **{report['status']}**. Модель API: `{report['model']}`.", '',
            f"## Всего за запуск: {number(report['usage']['actual_run']['total_tokens'])} токенов", '',
            'Полный фактический расход: создание диалога + все выполненные сжатия + все выполненные проверки памяти. '
            'Каждый запрос учтён один раз. Таблица сравнения ниже относится только к проверкам и summary, не ко всему запуску.', '',
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
        rows += [f"Сжатий: {counts['compressions']}; записей полной истории: {counts['full_records']}.", '']
        if counts['compressed_tail_records'] is not None:
            rows += [f"В хвосте: {counts['compressed_tail_records']} записей / "
                     f"{counts['compressed_tail_visible_messages']} видимых сообщений.", '']
    rows += ['## Память', '', '| Деталь | Эталон | ' + ' | '.join(labels[mode] for mode in modes) + ' |',
             '|---|---|' + '---|' * len(modes)]
    probes = {mode: {row['fact_id']: row for row in report['probes'][mode]} for mode in ('full', 'compressed')}
    for fact in report['facts']:
        values = []
        for mode in modes:
            evaluation = probes[mode].get(fact['id'], {}).get('evaluation')
            if evaluation is None:
                values.append('— не проверено')
            else:
                mark = {'correct': '✓', 'missing': 'не вспомнил', 'mismatch': '≠ требует сверки'}[evaluation['status']]
                values.append(f"{cell(evaluation['value'])} · {mark}")
        rows.append(f"| {fact['label']} | {cell(fact['expected'])} | " + ' | '.join(values) + ' |')
    rows += ['', 'Оценка: нормализованное точное совпадение с заранее заданными допустимыми значениями. '
             'Другие формулировки помечаются «требует сверки», а не автоматически объявляются забытыми.', '']
    for mode in modes:
        score = report['score'][mode]
        rows.append(f"- {labels[mode]}: **{score['correct']}/{score['total']}**; выполнено {score['checked']} проверок.")
    lost = report['lost_after_compression']
    if case == 'both':
        rows += ['', 'Детали, прошедшие без сжатия, но не прошедшие со сжатием: ' +
                 (', '.join(lost) if lost else 'не обнаружены в выполненных проверках') + '.']
    rows += ['', '## Расход проверок памяти (без создания диалога)', '', '| Показатель | ' + ' | '.join(labels[m] for m in modes) + ' |',
             '|---|' + '---:|' * len(modes), '| Контекст каждого вопроса | ' +
             ' | '.join('Вся переписка' if m == 'full' else 'Summary + хвост' for m in modes) + ' |']
    for key, label in (('prompt_tokens', 'Входные токены'), ('completion_tokens', 'Выходные токены'),
                       ('total_tokens', 'Всего токенов')):
        rows.append(f"| {label} | " + ' | '.join(number(report['usage'][m + '_comparison'][key]) for m in modes) + ' |')
    if 'compressed' in modes:
        rows += ['| Из них создание summary | ' + ' | '.join(
            '0' if m == 'full' else number(report['usage']['summary']['total_tokens']) for m in modes) + ' |', '',
                 '**Summary уже включено в итог теста со сжатием.** Повторно прибавлять его не нужно.']
    saving = report['savings']
    if case == 'both':
        rows += ['', f"Экономия на проверках **с учётом summary**: {number(saving['tokens'])} токенов "
                 f"({saving['percent'] if saving['percent'] is not None else '?'}%)."]
    else:
        rows += ['', 'Выполнен один выбранный тест. Для расчёта экономии запустите оба теста на одной переписке (--case both).']
    rows += ['',
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
