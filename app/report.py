"""JSON и Markdown-отчёт с разделением общих и служебных расходов."""

import json


CONCLUSIONS = ('- Sliding Window — короткие независимые вопросы, минимум кода и расходов.\n'
               '- Sticky Facts — рабочие диалоги с бюджетом, ограничениями и решениями.\n'
               '- Summary — связный рассказ и нюансы, которые сложно свести к ключам.\n')


def load_report(path):
    report = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(report, dict) or not isinstance(report.get('runs'), list):
        raise ValueError('Некорректный формат отчёта: нужен список runs.')
    return report


def write_report(report, directory):
    (directory / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# День 10 · сравнение стратегий', '', f"Модель: {report['model']} · окно: {report['window']} сообщений.",
             f"Статус: {report.get('status', 'archived')}.", '',
             '| Стратегия | Детали | Токены всего | Из них память | API всего | Из них память | Строк класса |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for run in report['runs']:
        lines.append(f"| {run['name']} | {run['correct']}/{run['questions']} | {run['tokens']} | "
                     f"{run['side_tokens']} | {run['calls']} | {run['side_calls']} | {run['loc']} |")
    lines += ['', 'Служебные токены и вызовы уже включены в итог. Строки включают комментарии.',
              'Счётчик API — логические вызовы, не сетевые попытки. Оценка по ключевым словам приблизительная.',
              'Включены диалог, проверка памяти и поддержка стратегии. Проверки идут последовательно.', '']
    if report.get('error'):
        lines += ['## Причина остановки', '', report['error'], '',
                  'Незавершённая стратегия не включена: её расход неизвестен, а не равен нулю.', '']
    lines += ['## Когда применять', '', CONCLUSIONS]
    (directory / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
