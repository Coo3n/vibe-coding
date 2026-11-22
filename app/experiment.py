"""Эксперимент Дня 11: слои памяти, токены каждого слоя и ablation-очистка."""

import inspect
from copy import deepcopy

from .agent import Agent
from .memory import MemoryStore
from .report import write_report
from .scenario import (DIALOGUE, EXTRA_DIALOGUE, LONG_TERM_SEED, PROBES,
                       REMEMBER_DECISIONS, REMEMBER_PROBES, RESET_PROBES, SYSTEM_PROMPT)


DENIAL_MARKERS = (
    'не обсужда', 'не упомин', 'неизвестно', 'нет информации',
    'не зафиксирова', 'не указан', 'не помню', 'нет данных',
    'нет упоминаний', 'не удалось найти', 'не называл', 'не задан',
    'не знаю', 'не говорили', 'жду от вас', 'жду вашу', 'от вас жду',
    'эту информацию у вас', 'вы должны мне', 'выбираете вы',
)

# Конфигурации ablation: имя → какие слои НЕ попадают в промпт.
ABLATIONS = (
    ('full', ()),
    ('no-short', ('short',)),
    ('no-working', ('working',)),
    ('no-long', ('long',)),
)

MEASURE_QUESTION = 'Какой у нас бюджет на человека?'


def answer_ok(answer, keywords):
    low = answer.lower()
    keyword = min((low.find(word) for word in keywords if word in low), default=-1)
    denial = min((low.find(word) for word in DENIAL_MARKERS if word in low), default=-1)
    return keyword != -1 and (denial == -1 or keyword < denial)


def ask_probes(agent, probes, skip, phase, on_event, running, calls):
    results = []
    for index, probe in enumerate(probes, 1):
        if on_event:
            on_event({'kind': 'prompt', 'phase': phase, 'index': index, 'total': len(probes),
                      'prompt': probe['question']})
        before = agent.memory.side_tokens
        reply = agent.ask(probe['question'], skip)
        calls[0] += 1
        ok = answer_ok(reply['answer'], probe['keywords'])
        results.append({'question': probe['question'], 'layer': probe['layer'],
                        'answer': reply['answer'], 'ok': ok})
        running[0] += reply['usage']['total_tokens'] + reply['side_tokens']
        if on_event:
            on_event({'kind': 'step', 'phase': phase, 'answer': reply['answer'], 'ok': ok,
                      'usage': reply['usage'],
                      'side_tokens': agent.memory.side_tokens - before,
                      'side_calls': reply['side_calls'], 'running': running[0]})
    return results


def measure_layer_tokens(client, memory, state, on_event, running, calls):
    """Дельта prompt_tokens при отключении слоёв по одному = вклад слоя.

    Замер на одном репрезентативном запросе из одинакового состояния слоёв;
    текст ответа не используется и в историю не попадает.
    """
    prompt_tokens, answers = {}, {}
    for name, skip in (('full', ()) , ('no-long', ('long',)),
                       ('no-working', ('working',)), ('no-short', ('short',))):
        memory.restore(state)
        memory.short_term.append({'role': 'user', 'content': MEASURE_QUESTION})
        if on_event:
            on_event({'kind': 'measure_prompt', 'name': name, 'prompt': MEASURE_QUESTION})
        answer, usage = client.complete(memory.context(SYSTEM_PROMPT, skip))
        calls[0] += 1
        running[0] += usage['total_tokens']
        prompt_tokens[name] = usage['prompt_tokens']
        answers[name] = answer
        if on_event:
            on_event({'kind': 'measure', 'answer': answer, 'usage': usage, 'running': running[0]})
    memory.restore(state)
    return {'question': MEASURE_QUESTION, 'prompt_tokens': prompt_tokens,
            'layer_tokens': {
                'long': prompt_tokens['full'] - prompt_tokens['no-long'],
                'working': prompt_tokens['full'] - prompt_tokens['no-working'],
                'short': prompt_tokens['full'] - prompt_tokens['no-short'],
            },
            'answers': answers}


def run_experiment(client, directory, long_term_path, window=4, on_event=None):
    directory.mkdir(parents=True, exist_ok=False)
    memory = MemoryStore(long_term_path, window)
    report = {'model': client.model_name, 'mode': getattr(client, 'mode', 'live'),
              'window': window, 'status': 'running', 'runs': []}
    running = [0]
    calls = [0]
    try:
        # 0. Явный сид долговременной памяти в каталоге эксперимента.
        memory.long_term = dict(LONG_TERM_SEED)
        memory._save_long_term()
        initial_memory = {
            'long_term': deepcopy(memory.long_term),
            'working': deepcopy(memory.working),
            'short_messages': len(memory.short_term),
        }
        report['initial_memory'] = initial_memory
        if on_event:
            on_event({'kind': 'initial_memory', 'memory': initial_memory})

        # 1. Диалог + дополнительные ходы: working обновляется автоматически.
        agent = Agent(client, memory)
        for phase, items in (('dialogue', DIALOGUE), ('extra', EXTRA_DIALOGUE)):
            for index, prompt in enumerate(items, 1):
                if on_event:
                    on_event({'kind': 'prompt', 'phase': phase, 'index': index,
                              'total': len(items), 'prompt': prompt})
                before = memory.side_tokens
                reply = agent.ask(prompt)
                calls[0] += 1
                running[0] += reply['usage']['total_tokens'] + reply['side_tokens']
                if on_event:
                    on_event({'kind': 'step', 'phase': phase, 'answer': reply['answer'], 'ok': None,
                              'usage': reply['usage'],
                              'side_tokens': reply['side_tokens'],
                              'side_calls': reply['side_calls'], 'running': running[0]})
        state = memory.snapshot()

        # 2. Дамп слоёв после диалога: что попало в каждый слой.
        # Копируем, чтобы позднейшие явные записи в long-term не мутировали дамп.
        report['layers_dump'] = {
            'working': deepcopy(memory.working),
            'short_messages': len(memory.short_term),
            'short_window_messages': [m['content'] for m in memory.short_term[-window:]],
            'long_term': deepcopy(memory.long_term)}
        write_report(report, directory)

        # 3. Токены каждого слоя: дельты на репрезентативном запросе.
        measure = measure_layer_tokens(client, memory, state, on_event, running, calls)
        measure['working_side_tokens'] = memory.side_tokens
        measure['working_side_calls'] = memory.side_calls
        report['token_measure'] = measure
        write_report(report, directory)

        # 4. Ablation в том же диалоге: полный → без слоя, вопросы одинаковые.
        for name, skip in ABLATIONS:
            if on_event:
                on_event({'kind': 'case_start', 'name': name})
            memory.restore(state)
            results = ask_probes(agent, PROBES, skip, 'probe', on_event, running, calls)
            report['runs'].append({'name': name, 'skip': list(skip), 'phase': 'same-dialogue',
                                   'correct': sum(r['ok'] for r in results),
                                   'total': len(results), 'answers': results})
            write_report(report, directory)

        # 5. Новый диалог после /reset: отвечает только long-term.
        if on_event:
            on_event({'kind': 'case_start', 'name': 'new-dialog'})
        agent.reset()
        results = ask_probes(agent, RESET_PROBES, (), 'reset', on_event, running, calls)
        report['runs'].append({'name': 'new-dialog', 'skip': [], 'phase': 'after-reset',
                               'correct': sum(r['ok'] for r in results),
                               'total': len(results), 'answers': results})
        write_report(report, directory)

        # 6. Явное сохранение решений в long-term, затем новый диалог.
        if on_event:
            on_event({'kind': 'case_start', 'name': 'after-remember'})
        for text in REMEMBER_DECISIONS:
            memory.remember(text)
        agent.reset()
        results = ask_probes(agent, REMEMBER_PROBES, (), 'remember', on_event, running, calls)
        report['runs'].append({'name': 'after-remember', 'skip': [], 'phase': 'after-reset',
                               'correct': sum(r['ok'] for r in results),
                               'total': len(results), 'answers': results})
        write_report(report, directory)

        # 7. Финальная очистка: long-term сброшен, новый диалог — не отвечает ничего.
        if on_event:
            on_event({'kind': 'case_start', 'name': 'new-dialog-no-long'})
        memory.clear('long')
        agent.reset()
        results = ask_probes(agent, RESET_PROBES, (), 'reset', on_event, running, calls)
        report['runs'].append({'name': 'new-dialog-no-long', 'skip': [], 'phase': 'after-reset',
                               'correct': sum(r['ok'] for r in results),
                               'total': len(results), 'answers': results})
        report['tokens'] = running[0]
        report['main_calls'] = calls[0]
        report['side_tokens'] = memory.side_tokens
        report['side_calls'] = memory.side_calls
        report['loc'] = len(inspect.getsource(MemoryStore).splitlines())
        report['status'] = 'completed'
    except (ValueError, RuntimeError, KeyboardInterrupt) as error:
        report['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        report['error'] = ('Работа остановлена пользователем.' if isinstance(error, KeyboardInterrupt)
                           else str(error))
        report['tokens'] = running[0]
        report['main_calls'] = calls[0]
        report['side_tokens'] = memory.side_tokens
        report['side_calls'] = memory.side_calls
    write_report(report, directory)
    return report
