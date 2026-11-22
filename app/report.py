"""Сохранение полного прогона и сводки для просмотра без API."""

import json


GROUPS = {'control': 'Контроль', 'bypass': 'Попытка обхода', 'malformed': 'Ответ вне формата'}


def load_report(path):
    report = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(report, dict) or not isinstance(report.get('cases'), list):
        raise ValueError('Некорректный отчёт эксперимента.')
    return report


def write_report(report, directory):
    (directory / 'report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    codex = report['mode'] == 'codex'
    lines = ['# День 14 · Инварианты и ограничения состояния', '']
    if codex:
        lines += [f"Codex CLI, модель `{report['model']}`, размер окна {report['window']}.",
                  f"Завершено {report['codex_calls']} из {report['codex_attempts']} вызовов Codex CLI.",
                  f"Токены завершённых ходов с контекстом Codex · {report['tokens'] or 'нет данных'}.",
                  f"Модель выдала ожидаемый по сценарию результат в {report['model_matched']} из "
                  f"{report['model_total']} завершённых запросов."]
    else:
        lines += [f"Офлайн-режим `{report['model']}`, размер окна {report['window']}, вызовов API {report['api_calls']}."]
    if report['error']:
        lines += ['', 'Прогон прерван. ' + report['error'], '']
    lines += [
        f"Проверки приложения прошли в {report['application_passed']} из {report['total']} сценариев.",
        f"Конфликтных вариантов выдано {report['bypasses_accepted']} из {report['bypasses']} попыток обхода.",
        f"Отказ с указанием нужного инварианта получен в {report['bypasses_blocked']} из {report['bypasses']} случаев.",
        f"Локальных фикстур с ответами вне формата заблокировано {report['malformed_blocked']} из {report['malformed']}.",
        '', '## Сценарии', '',
        '| Группа | Проверка | Источник | Результат | Проверка приложения |', '|---|---|---|---|---|',
    ]
    source_labels = {'codex': 'Codex CLI', 'offline': 'Офлайн', 'local': 'Приложение', 'fixture': 'Фикстура'}
    for case in report['cases']:
        status = case['reply']['status'] if case['reply'] else '—'
        expected = case['expected_status']
        if expected and not case['passed']:
            status += f' (ожидалось {expected})'
        result = 'Прошла' if case['application_ok'] else 'Ошибка'
        lines.append(f"| {GROUPS[case['group']]} | {case['name']} | {source_labels[case['source']]} | {status} | {result} |")
    lines.extend(['', '## Журнал явных изменений', ''])
    for entry in report['audit']:
        lines.append(f"- `{entry['field']}` · `{entry['old_value']}` → `{entry['new_value']}` · {entry['changed_at']}")
    lines += ['', 'Запросы, исходные кандидаты и ответы приложения доступны в [report.json](report.json).']
    if codex:
        lines += ['Счётчик вызовов включает ходы Codex CLI. Команды приложения и ответы фикстур обрабатываются локально. '
                  'Конфликт в структурированном кандидате фиксируется независимо от того, что приложение вывело пользователю.']
    else:
        lines += ['Прогон проверяет кодовый фильтр с локальными ответами. Поведение реальной модели не измерялось.']
    (directory / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
