"""JSON и Markdown-отчёт Дня 12: профили, соблюдение предпочтений, стресс-тест."""

import json

from .profile import DETAILED_MIN_WORDS, SHORT_MAX_WORDS


LABELS = {
    'exp1-opposite': 'Эксперимент 1 · противоположные профили',
    'exp2-stability': 'Эксперимент 2 · устойчивость (10 вопросов)',
    'exp3-replenish': 'Эксперимент 3 · пополнение из диалога',
}

PHASE_LABELS = {'base': 'базовый диалог', 'question': 'вопрос профилю',
                'stress': 'стресс-тест', 'before': 'до смены стиля',
                'switch': 'смена предпочтений', 'after': 'после смены'}

FIELDS = ('стиль', 'формат', 'тон')


def load_report(path):
    report = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(report, dict) or not isinstance(report.get('runs'), list):
        raise ValueError('Некорректный формат отчёта: нужен список runs.')
    return report


def _pref_summary(run):
    return ' · '.join(f"{run['prefs'][f]}" for f in FIELDS)


def _spending(report):
    if report.get('mode') == 'offline':
        return ['Локальные шаблоны · вызовов API 0 · токены не измерялись.', '']
    lines = []
    if report.get('mode') == 'codex':
        lines.extend(['Codex CLI · токены включают контекст агента Codex.', ''])
    if report.get('tokens') is not None:
        working = f"{report.get('side_tokens', 0):,}".replace(',', ' ')
        profile = f"{report.get('profile_side_tokens', 0):,}".replace(',', ' ')
        lines.append(f"Расход прогона: {report['tokens']:,} токенов · ".replace(',', ' ')
                     + f"{report.get('main_calls', '?')} основных и "
                     + f"{report.get('side_calls', 0) + report.get('profile_side_calls', 0)} "
                       'служебных вызовов (working-память — '
                     + working + ' токенов, извлечение предпочтений — '
                     + profile + ' токенов).')
        lines.append('')
    return lines


def _measure_table(report):
    measure = report.get('profile_measure')
    if not measure:
        return ([] if report.get('mode') in ('offline', 'codex')
                else ['## Токены блока профиля', '', 'Замер недоступен.', ''])
    return ['## Цена профиля в промпте', '',
            f"Замер на репрезентативном запросе «{measure['question']}» из одинакового "
            'состояния памяти: `prompt_tokens` с блоком профиля «'
            + measure['profile'] + '» и без него, дельта = цена блока.', '',
            '| Блок | Токенов на запрос | prompt_tokens с блоком | без блока |',
            '|---|---:|---:|---:|',
            f"| Профиль пользователя | {measure['block_tokens']} | "
            f"{measure['prompt_tokens_with']} | {measure['prompt_tokens_without']} |",
            '',
            'Извлечение предпочтений — отдельная цена: служебный вызов после каждой '
            'реплики (учтён в расходе прогона выше).', '']


def _exp1_table(report):
    run = next((r for r in report['runs'] if r['name'] == 'exp1-opposite'), None)
    if not run:
        return []
    lines = ['## Эксперимент 1 · противоположные профили', '',
             'Три профиля отвечают на одни и те же вопросы. Перед каждым ответом '
             'восстанавливаются данные и история базового диалога. Предпочтение '
             '«с кодом» проверяется на двух вопросах, где код уместен. '
             'В столбце «Ответы» учитываются ответы, прошедшие все применимые проверки '
             'с сохранением исходного профиля.', '',
             '| Профиль | Предпочтения | Стиль | Формат | Тон | Ответы |',
             '|---|---|---:|---:|---:|---:|']
    for name, data in run['profiles'].items():
        compliance = data['compliance']
        cells = [f"{fraction(*compliance.get(f, (0, 0)))}" for f in FIELDS]
        answers = data['compliant_answers']
        lines.append(f"| {name} | {_pref_summary(data)} | " + ' | '.join(cells)
                     + f' | {answers}/{len(data["questions"])} |')
    lines.append('')
    details = []
    for name, data in run['profiles'].items():
        for q in data['questions']:
            failed = [f"{c['field']} ({c['detail']})" for c in q['checks'] if not c['ok']]
            failed.extend(q.get('profile_changes_during', []))
            if failed:
                details.append(f"- {name} · «{q['question']}» — "
                               + ', '.join(failed))
    if details:
        lines += ['Нарушения по ответам:', ''] + details + ['']
    return lines


def _exp2_table(report):
    run = next((r for r in report['runs'] if r['name'] == 'exp2-stability'), None)
    if not run:
        return []
    first = run.get('first_violation')
    lines = ['## Эксперимент 2 · устойчивость: 10 вопросов подряд', '',
             f"Профиль «{run['profile']}». Вопросы разные, часть соблазняет нарушить "
             'формат («покажи кодом», «расскажи подробно»).', '',
             '| № | Вопрос | Стиль | Формат | Тон |', '|---:|---|---:|---:|---:|']
    for index, q in enumerate(run['questions'], 1):
        marks = {c['field']: ('✓' if c['ok'] else '✗') for c in q['checks']}
        lines.append(f"| {index} | {q['question']} | "
                     + ' | '.join(marks.get(f, '—') for f in FIELDS) + ' |')
    lines.append('')
    if first:
        lines.append(f"**Первое нарушение — вопрос №{first['index']}** "
                     f"(«{first['question']}»): {', '.join(first['fields'])}.")
    else:
        lines.append('10 из 10 ответов прошли проверку. Первого нарушения нет.')
    changed = [f"№{i + 1}: {q['profile_changes_during']}"
               for i, q in enumerate(run['questions']) if q.get('profile_changes_during')]
    if changed:
        lines += ['', 'Изменения профиля на разовых просьбах', ''] + [f'- {line}' for line in changed]
    lines.append('')
    return lines


def _exp3_table(report):
    run = next((r for r in report['runs'] if r['name'] == 'exp3-replenish'), None)
    if not run:
        return []
    lines = ['## Эксперимент 3 · пополнение профиля из диалога', '',
             f"Диалог в профиле «{run['profile']}»: 2 вопроса → явная смена "
             'предпочтений → ещё 2 вопроса. Смена:', '', f"> {run['switch_message']}",
             '']
    lines.append('Изменения, извлечённые из реплики')
    lines += ['', *([f'- {item}' for item in run['changes']] or ['Изменений нет.'])]
    lines += ['', '| Этап | Стиль | Формат | Тон |',
              '|---|---:|---:|---:|']
    for stage, prefs, compliance in (('до смены', run['before_prefs'], run['before_compliance']),
                                     ('после смены', run['after_prefs'], run['after_compliance'])):
        cells = [f"{prefs[f]} · {fraction(*compliance.get(f, (0, 0)))}" for f in FIELDS]
        lines.append(f'| {stage} | ' + ' | '.join(cells) + ' |')
    result = 'совпадает с активным' if run['persisted'] else 'отличается от активного'
    lines += ['', f'Профиль, загруженный с диска после смены, {result}.', '',
              '```json',
              json.dumps(run['saved_profile'], ensure_ascii=False, indent=2),
              '```', '']
    return lines


def fraction(ok, total):
    return '?' if not total else f'{ok}/{total}'


def write_report(report, directory):
    (directory / 'report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# День 12 · персонализация ассистента', '',
             f"{report['model']} · окно {report['window']} сообщений.",
             f"Статус прогона — {report.get('status', 'archived')}.", '']
    lines += _spending(report)
    lines += _measure_table(report)
    lines += _exp1_table(report)
    lines += _exp2_table(report)
    lines += _exp3_table(report)
    if report.get('error'):
        lines += ['## Причина остановки', '', report['error'], '',
                  'В расходе учтены вызовы с полученным usage.', '']
    lines += ['## Методика', '',
              'Профиль добавляется после долговременной памяти отдельным сообщением system. '
              'Команда /prefer и обработка явных предпочтений в реплике обновляют JSON профиля. '
              'У каждого прогона временное хранилище.',
              '',
              f'Пороги длины. «Кратко» — до {SHORT_MAX_WORDS} слов, «подробно» — от {DETAILED_MIN_WORDS}. '
              '«С кодом» — оформленный блок кода '
              '(только на вопросах, где код уместен), «без кода» — отсутствие блока; '
              'Для тона проверяется форма обращения по местоимениям и списку '
              'распространённых повелительных форм. Например, «проверь» засчитывается '
              'для неформального профиля, а «проверьте» — для формального. '
              'Формальный профиль также требует отсутствия эмодзи. '
              'Нейтральная подача без обращения и смешанное обращение отмечаются отдельно '
              'и не засчитываются ни одному профилю.',
              '',
              'Разовые просьбы («покажи кодом» в одном вопросе) не должны менять '
              'профиль — извлекатель инструктирован отличать их от установок на '
              'будущее; срабатывания зафиксированы в отчёте.',
              '',
              'Полные ответы сохранены в report.json.']
    (directory / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
