"""Эксперимент Дня 12: персонализация поверх модели памяти.

Три эксперимента: противоположные профили на одинаковых вопросах,
стресс-тест из 10 вопросов и пополнение профиля из диалога.
Соблюдение предпочтений проверяется эвристиками (слова, блоки кода,
маркеры тона) — приблизительно, с честными оговорками в отчёте.
"""

import re
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from .agent import Agent
from .memory import MemoryStore
from .profile import DETAILED_MIN_WORDS, SHORT_MAX_WORDS, UserProfile, seed_profiles
from .report import write_report
from .scenario import (BASE_DIALOGUE, EXP1_QUESTIONS, EXP2_PROFILE, EXP2_QUESTIONS,
                       EXP3_AFTER, EXP3_BEFORE, EXP3_PROFILE, EXP3_SWITCH,
                       LONG_TERM_SEED, MEASURE_QUESTION, PROFILE_SEEDS, SYSTEM_PROMPT)


# ---- эвристики соблюдения предпочтений (приблизительные, честно оговорены) ----

CODE_BLOCK = re.compile(r'(?ms)^ *```[^\n]*\n[^\n]+.*?^ *```\s*$')
WORD_RE = re.compile(r'[0-9A-Za-zА-Яа-яЁё]+(?:[-\'][0-9A-Za-zА-Яа-яЁё]+)*')
EMOJI_RE = re.compile('[\U0001F000-\U0001FAFF☀-➿⬀-⯿️]')
FORMAL_YOU_RE = re.compile(r'\b(Вы|Вас|Вам|Вами|Ваш\w*)\b')
INFORMAL_YOU_RE = re.compile(
    r'\b(?:ты|тебя|тебе|тобой|тобою|тво(?:й|я|[её]|и|его|ей|ему|им|их|ими|ём|ем|ю))\b',
    re.IGNORECASE)
DIRECTIVES_RE = re.compile(
    r'\b(умножь|проверь|учти|заведи|вычти|запиши|привяжи|сгруппируй|вынеси|'
    r'собери|раздели|оставь|сравни|сверь|используй|посмотри)(те)?\b', re.IGNORECASE)


def word_count(text):
    return len(WORD_RE.findall(text))


def score_address(value, answer):
    prose = CODE_BLOCK.sub('', answer)
    directives = list(DIRECTIVES_RE.finditer(prose))
    formal = FORMAL_YOU_RE.search(prose) or next(
        (match for match in directives if match[2]), None)
    informal = INFORMAL_YOU_RE.search(prose) or next(
        (match for match in directives if not match[2]), None)
    if formal and informal:
        return False, f'смешанное обращение: «{formal[0]}» и «{informal[0]}»'
    if informal:
        return value == 'неформально', f'обращение на «ты»: «{informal[0]}»'
    if formal:
        emoji = bool(EMOJI_RE.search(prose))
        detail = f'обращение на «вы»: «{formal[0]}»'
        if emoji and value == 'формально':
            detail += ', есть эмодзи'
        return value == 'формально' and not emoji, detail
    return False, 'нейтральная подача, явных форм обращения нет'


def score_answer(profile, answer, code_appropriate):
    """Проверка каждого предпочтения профиля в одном ответе → ok/fail.

    «с кодом» проверяется только на вопросах, где код уместен;
    «без кода» — на всех: ассистент не должен вставлять код сам.
    """
    words = word_count(answer)
    code = bool(CODE_BLOCK.search(answer))
    checks = []
    for field, value in (('стиль', profile.style), ('формат', profile.code_format),
                         ('тон', profile.tone)):
        if field == 'стиль':
            if value == 'кратко':
                ok, detail = words <= SHORT_MAX_WORDS, f'{words} слов (порог ≤ {SHORT_MAX_WORDS})'
            else:
                ok, detail = words >= DETAILED_MIN_WORDS, f'{words} слов (порог ≥ {DETAILED_MIN_WORDS})'
        elif field == 'формат':
            if value == 'с кодом' and not code_appropriate:
                continue  # код в этом вопросе не уместен — не проверяем
            ok, detail = code if value == 'с кодом' else not code, \
                'блок кода ' + ('есть' if code else 'нет')
        else:
            ok, detail = score_address(value, answer)
        checks.append({'field': field, 'value': value, 'ok': ok, 'detail': detail})
    return checks


def compliance(check_lists):
    """Сводка по полю: сколько проверок пройдено из сделанных."""
    summary = {}
    for checks in check_lists:
        for check in checks:
            field = check['field']
            ok, total = summary.get(field, (0, 0))
            summary[field] = (ok + int(check['ok']), total + 1)
    return summary


def fraction(ok, total):
    return '?' if total == 0 else f'{ok}/{total}'


# ---- один шаг диалога с событием для консоли ----

def ask_step(agent, text, phase, index, total, on_event, running, calls, score=None):
    """Реплика пользователя → ответ; score(item) — функция проверок или None."""
    if on_event:
        on_event({'kind': 'prompt', 'phase': phase, 'index': index, 'total': total,
                  'prompt': text, 'profile': agent.profile.name if agent.profile else None})
    reply = agent.ask(text)
    if reply['usage'] is not None:
        calls[0] += 1
        running[0] += (reply['usage']['total_tokens'] + reply['side_tokens']
                       + reply['profile_side_tokens'])
    checks = score(text, reply['answer']) if score else None
    if on_event:
        on_event({'kind': 'step', **reply, 'checks': checks, 'running': running[0]})
    return reply, checks


def run_experiment(client, directory, window=4, on_event=None):
    directory.mkdir(parents=True, exist_ok=False)
    with TemporaryDirectory(prefix='lab12-') as temporary:
        scratch = Path(temporary)
        return _run_experiment(client, directory, scratch / 'long_term.json',
                               scratch / 'profiles', window, on_event)


def _run_experiment(client, directory, long_term_path, profiles_dir, window, on_event):
    memory = MemoryStore(long_term_path, window)
    report = {'model': client.model_name, 'mode': client.mode, 'window': window,
              'status': 'running', 'runs': []}
    running = [0]
    calls = [0]
    created = []  # все созданные объекты профилей — для честного подсчёта расхода

    def fresh(name):
        """Свежий профиль из сида на диске — изоляция экспериментов друг от друга."""
        seed = next(item for item in PROFILE_SEEDS if item['имя'] == name)
        path = profiles_dir / f'run-{len(created)}-{seed["файл"]}.json'
        profile = UserProfile.from_dict(seed, path)
        profile.save()
        created.append(profile)
        return profile

    try:
        # Сиды записываются во временное хранилище этого прогона.
        memory.long_term = dict(LONG_TERM_SEED)
        memory._save_long_term()
        if on_event:
            on_event({'kind': 'seed', 'count': len(memory.long_term),
                      'path': str(long_term_path)})
        profiles = seed_profiles(profiles_dir, PROFILE_SEEDS)
        created.extend(profiles.values())
        if on_event:
            on_event({'kind': 'profiles', 'count': len(profiles),
                      'path': str(profiles_dir),
                      'items': [p.describe() for p in profiles.values()]})

        # 1. Базовый диалог БЕЗ профиля: память Дня 11 наполняется как обычно.
        agent = Agent(client, memory)
        for index, prompt in enumerate(BASE_DIALOGUE, 1):
            ask_step(agent, prompt, 'base', index, len(BASE_DIALOGUE),
                     on_event, running, calls)
        state = memory.snapshot()
        report['base_dump'] = {'working': deepcopy(memory.working),
                               'short_messages': len(memory.short_term)}
        write_report(report, directory)

        # Парный замер доступен только с токенизатором провайдера.
        if client.mode == 'live':
            memory.restore(state)
            memory.short_term.append({'role': 'user', 'content': MEASURE_QUESTION})
            block = fresh('Подробно').block()
            _, usage_with = client.complete(memory.context(SYSTEM_PROMPT, profile_block=block))
            calls[0] += 1
            running[0] += usage_with['total_tokens']
            _, usage_without = client.complete(memory.context(SYSTEM_PROMPT))
            calls[0] += 1
            running[0] += usage_without['total_tokens']
            with_tokens, without_tokens = usage_with.get('prompt_tokens'), usage_without.get('prompt_tokens')
            if type(with_tokens) is int and type(without_tokens) is int:
                report['profile_measure'] = {
                    'question': MEASURE_QUESTION, 'profile': 'Подробно',
                    'prompt_tokens_with': with_tokens, 'prompt_tokens_without': without_tokens,
                    'block_tokens': with_tokens - without_tokens}
                if on_event:
                    on_event({'kind': 'measure', 'usage': usage_with,
                              'block_tokens': report['profile_measure']['block_tokens']})
            write_report(report, directory)

        # 3. Эксперимент 1: противоположные профили, одинаковые вопросы.
        if on_event:
            on_event({'kind': 'case_start', 'name': 'exp1-opposite'})
        exp1 = {'name': 'exp1-opposite', 'profiles': {}}
        for seed in PROFILE_SEEDS:
            name = seed['имя']
            agent.profile = fresh(name)
            expected = agent.profile
            questions = []
            for index, item in enumerate(EXP1_QUESTIONS, 1):
                memory.restore(state)
                agent.profile = fresh(name)
                reply, checks = ask_step(
                    agent, item['question'], 'question', index, len(EXP1_QUESTIONS),
                    on_event, running, calls,
                    score=lambda text, answer, item=item: score_answer(
                        expected, answer, item['code_appropriate']))
                questions.append({'question': item['question'],
                                  'code_appropriate': item['code_appropriate'],
                                  'answer': reply['answer'],
                                  'words': word_count(reply['answer']),
                                  'checks': checks,
                                  'profile_changes_during': reply['profile_changes']})
            exp1['profiles'][name] = {
                'prefs': expected.to_dict(),
                'compliance': compliance([q['checks'] for q in questions]),
                'compliant_answers': sum(all(check['ok'] for check in q['checks'])
                                         and not q['profile_changes_during'] for q in questions),
                'questions': questions}
        report['runs'].append(exp1)
        write_report(report, directory)

        # 4. Эксперимент 2: устойчивость — 10 вопросов подряд, профиль «Кратко».
        if on_event:
            on_event({'kind': 'case_start', 'name': 'exp2-stability'})
        memory.restore(state)
        agent.profile = fresh(EXP2_PROFILE)
        expected = UserProfile.from_dict(agent.profile.to_dict())
        questions, check_lists = [], []
        first_violation = None
        for index, item in enumerate(EXP2_QUESTIONS, 1):
            reply, checks = ask_step(agent, item['question'], 'stress', index,
                                     len(EXP2_QUESTIONS), on_event, running, calls,
                                     score=lambda text, answer, item=item: score_answer(
                                         expected, answer, item['code_appropriate']))
            failed = [c['field'] for c in checks if not c['ok']]
            if reply['profile_changes']:
                failed.append('профиль изменился на разовой просьбе')
            if failed and first_violation is None:
                first_violation = {'index': index, 'fields': failed,
                                   'question': item['question']}
            check_lists.append(checks)
            questions.append({'question': item['question'],
                              'code_appropriate': item['code_appropriate'],
                              'answer': agent.memory.short_term[-1]['content'],
                              'words': word_count(agent.memory.short_term[-1]['content']),
                              'checks': checks,
                              'profile_changes_during': reply['profile_changes']})
        report['runs'].append({'name': 'exp2-stability', 'profile': EXP2_PROFILE,
                               'questions': questions,
                               'compliant_answers': sum(all(check['ok'] for check in q['checks'])
                                                        and not q['profile_changes_during']
                                                        for q in questions),
                               'compliance': compliance(check_lists),
                               'first_violation': first_violation})
        write_report(report, directory)

        # 5. Эксперимент 3: пополнение из диалога — смена предпочтений посреди чата.
        if on_event:
            on_event({'kind': 'case_start', 'name': 'exp3-replenish'})
        memory.restore(state)
        agent.profile = fresh(EXP3_PROFILE)
        before_questions = []
        for index, item in enumerate(EXP3_BEFORE, 1):
            reply, checks = ask_step(agent, item['question'], 'before', index,
                                 len(EXP3_BEFORE), on_event, running, calls,
                                 score=lambda text, answer, item=item: score_answer(
                                     agent.profile, answer, item['code_appropriate']))
            before_questions.append({**item, 'answer': reply['answer'], 'checks': checks})
        before_prefs = {'стиль': agent.profile.style, 'формат': agent.profile.code_format,
                        'тон': agent.profile.tone}
        reply, _ = ask_step(agent, EXP3_SWITCH, 'switch', 1, 1,
                            on_event, running, calls)
        changes = reply['profile_changes']
        after_prefs = {'стиль': agent.profile.style, 'формат': agent.profile.code_format,
                       'тон': agent.profile.tone}
        saved_profile = UserProfile.load(agent.profile.path).to_dict()
        persisted = saved_profile == agent.profile.to_dict()
        agent.profile = UserProfile.load(agent.profile.path)
        created.append(agent.profile)
        after_questions = []
        for index, item in enumerate(EXP3_AFTER, 1):
            reply, checks = ask_step(agent, item['question'], 'after', index,
                                 len(EXP3_AFTER), on_event, running, calls,
                                 score=lambda text, answer, item=item: score_answer(
                                     agent.profile, answer, item['code_appropriate']))
            after_questions.append({**item, 'answer': reply['answer'], 'checks': checks})
        report['runs'].append({
            'name': 'exp3-replenish', 'profile': EXP3_PROFILE,
            'switch_message': EXP3_SWITCH, 'changes': changes,
            'before_prefs': before_prefs, 'after_prefs': after_prefs,
            'before_compliance': compliance([q['checks'] for q in before_questions]),
            'after_compliance': compliance([q['checks'] for q in after_questions]),
            'before_questions': before_questions,
            'after_questions': after_questions,
            'persisted': persisted,
            'saved_profile': saved_profile})
        write_report(report, directory)

        # 6. Итоговый расход.
        report['tokens'] = running[0]
        report['main_calls'] = calls[0]
        report['side_tokens'] = memory.side_tokens
        report['side_calls'] = memory.side_calls
        report['profile_side_tokens'] = sum(p.side_tokens for p in created)
        report['profile_side_calls'] = sum(p.side_calls for p in created)
        report['status'] = 'completed'
    except (OSError, ValueError, RuntimeError, KeyboardInterrupt) as error:
        report['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        report['error'] = ('Работа остановлена пользователем.' if isinstance(error, KeyboardInterrupt)
                           else str(error))
        report['tokens'] = running[0]
        report['main_calls'] = calls[0]
        report['side_tokens'] = memory.side_tokens
        report['side_calls'] = memory.side_calls
        report['profile_side_tokens'] = sum(p.side_tokens for p in created)
        report['profile_side_calls'] = sum(p.side_calls for p in created)
    if client.mode == 'offline':
        report.update(tokens=None, api_calls=0)
    else:
        calls_total = report['main_calls'] + report['side_calls'] + report['profile_side_calls']
        report['codex_calls' if client.mode == 'codex' else 'api_calls'] = calls_total
    write_report(report, directory)
    return report
