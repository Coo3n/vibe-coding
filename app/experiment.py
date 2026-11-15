"""Один сценарий и одинаковые контрольные вопросы для трёх стратегий."""

import inspect

from .context import STRATEGY_CLASSES
from .report import write_report
from .scenario import SYSTEM_PROMPT, DIALOGUE, QUESTIONS


DENIAL_MARKERS = (
    'не обсужда', 'не упомин', 'неизвестно', 'нет информации',
    'не зафиксирова', 'не указан', 'не помню', 'нет данных',
    'нет упоминаний', 'не удалось найти', 'не называл', 'не задан',
    'не знаю', 'не говорили', 'жду от вас', 'жду вашу', 'от вас жду',
    'эту информацию у вас', 'вы должны мне', 'выбираете вы',
)


def answer_ok(answer, keywords):
    low = answer.lower()
    keyword = min((low.find(word) for word in keywords if word in low), default=-1)
    denial = min((low.find(word) for word in DENIAL_MARKERS if word in low), default=-1)
    return keyword != -1 and (denial == -1 or keyword < denial)


def run_strategy(client, name, window=4, on_event=None):
    cls = STRATEGY_CLASSES[name]
    strategy = cls(window)
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    main_calls, main_tokens = 0, 0
    answers = []
    phases = (('dialogue', DIALOGUE), ('probe', QUESTIONS))
    for phase, items in phases:
        for index, item in enumerate(items, 1):
            prompt = item if phase == 'dialogue' else item['question']
            before_tokens, before_calls = strategy.side_tokens, strategy.side_calls
            messages.append({'role': 'user', 'content': prompt})
            strategy.after_user_message(messages, client.complete)
            answer, usage = client.complete(strategy.context(messages))
            messages.append({'role': 'assistant', 'content': answer})
            strategy.after_exchange(messages, client.complete)
            main_calls += 1
            main_tokens += usage['total_tokens']
            ok = None
            if phase == 'probe':
                ok = answer_ok(answer, item['keywords'])
                answers.append({'question': prompt, 'answer': answer, 'ok': ok})
            if on_event:
                on_event({'kind': 'step', 'phase': phase, 'index': index, 'total': len(items),
                          'prompt': prompt, 'answer': answer, 'ok': ok, 'usage': usage,
                          'side_tokens': strategy.side_tokens - before_tokens,
                          'side_calls': strategy.side_calls - before_calls,
                          'running': main_tokens + strategy.side_tokens})
    return {'name': name, 'correct': sum(answer['ok'] for answer in answers),
            'questions': len(QUESTIONS), 'tokens': main_tokens + strategy.side_tokens,
            'main_tokens': main_tokens, 'side_tokens': strategy.side_tokens,
            'calls': main_calls + strategy.side_calls, 'main_calls': main_calls,
            'side_calls': strategy.side_calls, 'loc': len(inspect.getsource(cls).splitlines()),
            'answers': answers}


def run_experiment(client, names, directory, window=4, on_event=None):
    directory.mkdir(parents=True, exist_ok=False)
    report = {'model': client.model_name, 'window': window, 'status': 'running', 'runs': []}
    try:
        for name in names:
            if on_event:
                on_event({'kind': 'case_start', 'name': name})
            report['runs'].append(run_strategy(client, name, window, on_event))
            write_report(report, directory)
        report['status'] = 'completed'
    except (ValueError, RuntimeError, KeyboardInterrupt) as error:
        report['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        report['error'] = 'Работа остановлена пользователем.' if isinstance(error, KeyboardInterrupt) else str(error)
    write_report(report, directory)
    return report
