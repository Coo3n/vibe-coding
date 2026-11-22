"""Воспроизводимая проверка ответов, содержимого промпта и очистки."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from .agent import Agent, RECORD_LABELS
from .memory import ACTIVE_HEADER, WORKING_HEADER, MemoryStore
from .report import write_report
from .scenario import BASELINE, CHECKS


SENTINEL = 'буфер посадочных талонов'


def _check_step(ask, key, statement, expected, question, retired):
    change = ask(statement, 'ИЗМЕНЕНИЕ')
    reply = ask(question, 'ПРОВЕРКА')
    prompt = '\n'.join(message['content'] for message in reply['messages'])
    return {
        'key': key, 'statement': statement, 'question': question,
        'expected': expected, 'answer': reply['answer'],
        'expected_in_answer': expected.casefold() in reply['answer'].casefold(),
        'stale_in_answer': [value for value in retired
                            if value.casefold() in reply['answer'].casefold()],
        'stale_in_prompt': [value for value in retired if value.casefold() in prompt.casefold()],
        'changed': change.get('update') == 'replaced', 'messages': reply['messages'],
    }


def run_experiment(client, directory, window=4, on_event=None):
    directory.mkdir(parents=True)
    dialogue = []

    def perform(text, phase, action):
        turn = {'phase': phase, 'input': text}
        if on_event is not None:
            on_event('prompt', turn)
        reply = action()
        turn.update(answer=reply['answer'], usage=reply['usage'],
                    source='model' if reply['usage'] else
                    ('offline' if reply.get('messages') else 'memory'),
                    messages=reply.get('messages', []))
        dialogue.append(turn)
        if on_event is not None:
            on_event('answer', turn)
        return reply

    def ask(text, phase):
        return perform(text, phase, lambda: agent.ask(text))

    def delayed(key, value, recorded_at):
        result = memory.durable.record(key, value, recorded_at)
        return {'answer': RECORD_LABELS[result], 'usage': {}, 'result': result}

    def finish():
        memory.finish_task()
        return {'answer': 'Рабочая память очищена. Долговременные факты сохранены.',
                'usage': {}}

    with TemporaryDirectory() as temporary:
        memory = MemoryStore(Path(temporary) / 'memory.json', window)
        agent = Agent(client, memory)
        baseline_at = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        late_at = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        for key, value in BASELINE.items():
            memory.durable.record(key, value, baseline_at)
        memory.durable.record('решение', 'Проверить варианты пересадок.', baseline_at)
        memory.durable.record('решение', 'Сверить правила перевозки багажа.', late_at)
        memory.record_working('заметка', SENTINEL)

        durable = memory.durable.read()
        initial_memory = {
            'active': {key: record['value'] for key, record in durable['active'].items()},
            'history': {key: [record['value'] for record in records]
                        for key, records in durable['history'].items()},
            'working': {key: record['value'] for key, record in memory.working.items()},
            'short_term_count': len(memory.short_term),
        }
        if on_event is not None:
            on_event('initial_memory', initial_memory)

        for _, _, _, question in CHECKS:
            ask(question, 'ИСХОДНЫЙ ФАКТ')
        retired = []
        steps = []
        for key, statement, expected, question in CHECKS:
            retired.append(BASELINE[key])
            steps.append(_check_step(ask, key, statement, expected, question, retired))

        late = perform(f"Запоздалая запись «срок — {BASELINE['срок']}» от {late_at}",
                       'ВРЕМЯ ЗАПИСИ', lambda: delayed('срок', BASELINE['срок'], late_at))
        same_at = memory.durable.read()['active']['срок']['recorded_at']
        tie = perform(f"Запись «срок — 1 апреля» с тем же временем {same_at}",
                      'ВРЕМЯ ЗАПИСИ', lambda: delayed('срок', '1 апреля', same_at))
        after_late = ask(CHECKS[0][3], 'ПОСЛЕ ЗАПОЗДАЛОЙ ЗАПИСИ')
        late_active = memory.durable.read()['active']['срок']['value'] == CHECKS[0][2]
        before_done = memory.durable.read()
        before_working = len(memory.working)
        perform('/done', 'ЗАВЕРШЕНИЕ ЗАДАЧИ', finish)
        after_done = memory.durable.read()
        after_task = ask('Что сохранено про «заметка»?', 'ПОСЛЕ ЗАВЕРШЕНИЯ')
        after_context = next(message['content'] for message in after_task['messages']
                             if message['content'].startswith(WORKING_HEADER))
        after_durable = next(message['content'] for message in after_task['messages']
                             if message['content'].startswith(ACTIVE_HEADER))
        after_working = len(memory.working)
        persisted = MemoryStore(memory.durable.path, window).durable.read() == after_done
        shutil.copyfile(memory.durable.path, directory / 'memory.json')
        replacements = sum(event['operation'] == 'replace' for event in after_done['changes'])
        decisions = after_done['history']['решение']
        decisions_in_prompt = any(record['value'] in after_durable for record in decisions)
        workflow = {
            'correct_answers': sum(step['expected_in_answer'] for step in steps),
            'stale_answers': sum(bool(step['stale_in_answer']) for step in steps),
            'stale_prompts': sum(bool(step['stale_in_prompt']) for step in steps),
            'replacements': replacements,
            'decisions': len(decisions), 'decisions_in_prompt': decisions_in_prompt,
            'late_write_ignored': late['result'] == 'ignored' and late_active,
            'tied_write_ignored': tie['result'] == 'ignored' and late_active,
            'late_question_answer': after_late['answer'],
            'late_question_current': CHECKS[0][2].casefold() in after_late['answer'].casefold(),
            'working_before_finish': before_working, 'working_after_finish': after_working,
            'working_absent_from_prompt': after_context == WORKING_HEADER + '{}'
            and all(SENTINEL not in message['content'] for message in after_task['messages']),
            'working_question_answer': after_task['answer'],
            'working_not_recalled': SENTINEL.casefold() not in after_task['answer'].casefold(),
            'durable_unchanged': after_done == before_done and persisted,
        }
        passed = (all(step['changed'] and step['expected_in_answer']
                      and not step['stale_in_answer'] and not step['stale_in_prompt']
                      for step in steps)
                  and replacements == len(CHECKS)
                  and len(decisions) == 2 and not decisions_in_prompt
                  and workflow['late_write_ignored'] and workflow['tied_write_ignored']
                  and workflow['late_question_current'] and workflow['working_not_recalled']
                  and before_working == 1 and after_working == 0
                  and workflow['working_absent_from_prompt'] and workflow['durable_unchanged'])
        model_usage = [turn['usage'] for turn in dialogue if turn['source'] == 'model']
        usage = {
            'model_calls': len(model_usage),
            'offline_answers': sum(turn['source'] == 'offline' for turn in dialogue),
            'memory_actions': sum(turn['source'] == 'memory' for turn in dialogue),
            'prompt_tokens': sum(entry.get('prompt_tokens', 0) for entry in model_usage),
            'completion_tokens': sum(entry.get('completion_tokens', 0) for entry in model_usage),
            'total_tokens': sum(entry['total_tokens'] for entry in model_usage),
        }
        report = {
            'model': client.model_name, 'mode': client.mode, 'window': window,
            'status': 'completed' if passed else 'failed',
            'initial_memory': initial_memory,
            'steps': steps, 'checks': workflow, 'audit': after_done['changes'],
            'dialogue': dialogue, 'usage': usage,
        }
        write_report(report, directory)
        return report
