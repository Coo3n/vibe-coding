"""JSON и Markdown-отчёт Дня 11: слои, токены, ablation."""

import json


LABELS = {
    'full': 'Полный контекст',
    'no-short': 'Без short-term',
    'no-working': 'Без working',
    'no-long': 'Без long-term',
    'new-dialog': 'Новый диалог (после /reset)',
    'after-remember': 'После /remember и /reset',
    'new-dialog-no-long': 'Новый диалог без long-term',
}

PHASE_LABELS = {'same-dialogue': 'тот же диалог', 'after-reset': 'новый диалог'}


def initial_memory_sections(memory):
    return (
        ('Долговременная память · факты Дня 10',
         [f'{key} — {value}' for key, value in memory['long_term'].items()]),
        ('Рабочая память',
         [f'{key} — {value}' for key, value in memory['working'].items()] or ['Пуста']),
        ('Краткосрочная память', [f"Сообщений в диалоге — {memory['short_messages']}"]),
    )


def _initial_memory(report):
    if 'initial_memory' not in report:
        return []
    lines = ['## Исходные данные в памяти', '']
    for title, entries in initial_memory_sections(report['initial_memory']):
        lines.extend([f'**{title}**', '', *(f'- {entry}' for entry in entries), ''])
    return lines


def load_report(path):
    report = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(report, dict) or not isinstance(report.get('runs'), list):
        raise ValueError('Некорректный формат отчёта: нужен список runs.')
    return report


def _tokens_table(report):
    measure = report.get('token_measure')
    if not measure:
        return ['## Токены слоёв', '', 'Замер не выполнен: прогон остановлен раньше.', '']
    main_calls = report.get('main_calls')
    lines = ['## Токены каждого слоя', '',
             f"Замер на репрезентативном запросе «{measure['question']}» из одинакового "
             'состояния слоёв: дельта `prompt_tokens` при отключении слоя по одному.', '',
             '| Слой | На запрос | За прогон (оценка¹) | prompt_tokens полных |',
             '|---|---:|---:|---:|']
    full = measure['prompt_tokens'].get('full')
    for layer, title in (('long', 'long-term'), ('working', 'working'),
                         ('short', 'short-term (окно)')):
        value = measure['layer_tokens'][layer]
        estimate = f'{value * main_calls:,}'.replace(',', ' ') if main_calls else '—'
        lines.append(f'| {title} | {value} | {estimate} | {full} |')
    side_tokens = f"{measure.get('working_side_tokens', 0):,}".replace(',', ' ')
    lines += [
        '',
        'Сопровождение working-памяти — цена служебных вызовов после реплик '
        + 'диалога, измерено точно: '
        + side_tokens + ' токенов в '
        + f"{measure.get('working_side_calls', 0)} вызовах (на момент замера, "
        'после 10 реплик диалога). Эта цена поддержки слоя не входит в колонку «На запрос».',
        '',
        '¹ Оценка: токены слоя на репрезентативном запросе × число основных вызовов '
        'за прогон. Реальный вклад меняется по мере роста слоёв, поэтому оценка нижняя '
        'граница не гарантирована — способ сравнения, а не тарификация.',
        '']
    return lines


def _ablation_table(report):
    lines = ['## Ablation: очистка слоёв по очереди', '',
             'После диалога состояние слоёв восстанавливается из снимка, один слой '
             'отключается из промпта, пробные вопросы задаются заново.', '',
             '| Конфигурация | Детали | Сломавшиеся вопросы |', '|---|---:|---|']
    for run in report['runs']:
        failed = [r['question'] for r in run['answers'] if not r['ok']]
        label = LABELS.get(run['name'], run['name'])
        if run['phase'] != 'after-reset':
            label += ' · ' + PHASE_LABELS.get(run['phase'], run['phase'])
        lines.append(f"| {label} | "
                     f"{run['correct']}/{run['total']} | "
                     f"{'; '.join(failed) if failed else '—'} |")
    lines.append('')
    return lines


def _layers_dump(report):
    dump = report.get('layers_dump')
    if not dump:
        return []
    lines = ['## Что попало в каждый слой (после диалога)', '']
    lines.append(f"- short-term: {dump['short_messages']} сообщений диалога, "
                 f"в окно последних {report['window']} входят:")
    lines += [f'  - {text}' for text in dump['short_window_messages']]
    lines.append('- working (обновляется автоматически после каждой реплики):')
    lines.append('```json')
    lines.append(json.dumps(dump['working'], ensure_ascii=False, indent=2))
    lines.append('```')
    lines.append('- long-term (только явные записи, файл на диске):')
    lines.append('```json')
    lines.append(json.dumps(dump['long_term'], ensure_ascii=False, indent=2))
    lines.append('```')
    lines.append('')
    return lines


def write_report(report, directory):
    (directory / 'report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# День 11 · модель памяти ассистента', '',
             f"Модель: {report['model']} · окно: {report['window']} сообщений.",
             f"Статус: {report.get('status', 'archived')}.", '']
    lines += _initial_memory(report)
    if report.get('mode') == 'codex':
        lines += ['Токены получены из событий Codex CLI. В общий расход входит служебный '
                  'контекст Codex. Дельты входных токенов сравниваются внутри этого прогона.', '']
    if report.get('tokens') is not None:
        lines.append(f"Расход прогона: {report['tokens']:,} токенов · ".replace(',', ' ')
                     + f"{report.get('main_calls', '?')} основных и "
                     + f"{report.get('side_calls', 0)} служебных вызовов (служебные — "
                     + f"{report.get('side_tokens', 0):,} токенов).".replace(',', ' '))
        lines.append('')
    lines += _tokens_table(report)
    lines += _ablation_table(report)
    lines += _layers_dump(report)
    if report.get('error'):
        lines += ['## Причина остановки', '', report['error'], '',
                  'Незавершённые конфигурации не включены: их расход неизвестен, а не равен нулю.', '']
    lines += ['## Методика', '',
              'Три слоя хранятся отдельно: short-term (окно последних сообщений), '
              'working (словарь задачи, служебный LLM-вызов после реплики), '
              'long-term (файл на диске, пополняется только явно).',
              'Оценка ответов по ключевым словам приблизительная: denial-маркеры '
              'и совпадение слова не гарантируют понимания, полные ответы — в report.json.',
              'Счётчик вызовов — логические запросы, не сетевые попытки.']
    (directory / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
