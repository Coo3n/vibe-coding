"""Сохранение проверок жизненного цикла в JSON и Markdown."""

import json


def load_report(path):
    report = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(report, dict) or not isinstance(report.get('checks'), list):
        raise ValueError('В отчёте нужен список checks.')
    return report


def write_report(report, directory):
    (directory / 'report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    mode = 'офлайн' if report.get('mode') != 'codex' else 'Codex CLI'
    call_line = (f"Вызовов Codex CLI — {report['codex_calls']}."
                 if mode == 'Codex CLI' else 'Вызовов API — 0.')
    lines = ['# День 13 · состояние задачи', '',
             f"Режим — {mode}, модель — {report['model']}. {call_line}",
             f"Пройдено проверок — {report['passed']}/{len(report['checks'])}.", '',
             '| Проверка | Итог | Наблюдение |', '|---|---|---|']
    for check in report['checks']:
        detail = check['detail'].replace('|', '&#124;').replace('\n', '<br>')
        outcome = 'пройдена' if check['ok'] else 'ошибка'
        lines.append(f"| {check['name']} | {outcome} | {detail} |")
    lines += ['', 'Автомат фиксирует переходы из кода. Сценарий подставляет ответ модели '
                  'с запрещённым действием и проверяет, что состояние осталось прежним. '
                  'Итог проверки артефакта вводится пользователем.', '']
    if mode == 'Codex CLI':
        usage = report['usage']
        lines += [f"Токены Codex CLI — {usage['prompt_tokens']} входящих, "
                  f"{usage['completion_tokens']} исходящих, всего {usage['total_tokens']}.", '',
                  'Проверки диалога оценивают изменение сохранённого состояния задачи. '
                  'Реакции Terra, исходные отвергнутые ответы и снимки автомата '
                  'сохранены в report.json.', '']
    if report.get('error'):
        lines += [f"Прогон остановлен. {report['error']}", '']
    (directory / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
