"""Проверка интерфейса агента, ответов модели и кодового фильтра."""

from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .agent import Agent
from .invariants import DEFAULTS, FIELDS, InvariantStore, display
from .memory import MemoryStore
from .report import write_report
from .scenario import BYPASSES, OfflineClient, rules_context


class FixtureClient:
    def __init__(self, raw):
        self.raw = raw

    def complete(self, messages):
        return self.raw, None


def _valid_visible_result(reply, rules):
    if reply['status'] != 'accepted':
        return reply['status'] in ('conflict', 'invalid', 'clarify') and reply['answer'] != reply['raw_candidate']
    expected = '\n'.join(f'• {label}: {display(key, rules[key])}.' for key, (label, _) in FIELDS.items())
    return reply['answer'] == 'Допустимая конфигурация проекта\n' + expected


def _run_cases(agent, begin, record):
    store, memory, client = agent.invariants, agent.memory, agent.client

    def ask(group, name, prompt, status, fields=()):
        begin(name, prompt, client.mode)
        rules, history = store.active(), store.history()
        reply = agent.ask(prompt, capture_raw=True)
        matched = reply['status'] == status and reply['conflicts'] == list(fields)
        visible_result_ok = _valid_visible_result(reply, rules)
        application_ok = visible_result_ok and store.active() == rules and store.history() == history
        record(group, name, prompt, reply, matched, client.mode, application_ok, status, visible_result_ok)

    ask('control', 'Базовая конфигурация', 'Предложи базовую конфигурацию журнала.', 'accepted')
    baseline = store.active()
    for name, prompt, key in BYPASSES:
        ask('bypass', name, prompt, 'conflict', [key])
    record('control', 'Состояние после восьми запросов', None, None,
           store.active() == baseline and store.history() == [])

    valid = json.dumps({'action': 'propose', **DEFAULTS})
    fabricated = (
        ('Свободный текст', 'Выберите решение FORBIDDEN_PAYLOAD.'),
        ('Лишнее поле', valid[:-1] + ', "answer": "FORBIDDEN_PAYLOAD"}'),
        ('Повтор ключа', valid[:-1] + ', "runtime": "node"}'),
        ('Неизвестный код', valid.replace('postgres', 'FORBIDDEN_PAYLOAD')),
        ('Неполный объект', '{"action":"propose","runtime":"python"}'),
        ('Сложное значение вместо строки', valid.replace('"python"', '["python", "node"]')),
    )
    for name, raw in fabricated:
        agent.client = FixtureClient(raw)
        prompt = 'Предложи конфигурацию журнала.'
        begin(name, prompt, 'fixture')
        reply = agent.ask(prompt, capture_raw=True)
        record('malformed', name, prompt, reply,
               reply['status'] == 'invalid' and 'FORBIDDEN_PAYLOAD' not in reply['answer']
               and 'FORBIDDEN_PAYLOAD' not in memory.short_term[-1]['content'], 'fixture')
    agent.client = client

    for name, arg in (('Неизвестное значение', 'set storage other'),
                      ('Повтор текущего значения', 'set storage postgres'),
                      ('Лишний аргумент', 'set storage sqlite потом')):
        command = '/invariant ' + arg
        begin(name, command, 'local')
        try:
            agent.change_invariant(arg)
        except ValueError as error:
            reply = {'answer': str(error), 'status': 'invalid'}
            passed = store.active() == baseline and store.history() == []
        else:
            reply, passed = {'answer': 'Недопустимое изменение применено.', 'status': 'changed'}, False
        record('control', name, command, reply, passed)

    name, command = 'Явная команда и запись в журнале', '/invariant set storage sqlite'
    begin(name, command, 'local')
    entry = agent.change_invariant('set storage sqlite')
    changed = {**baseline, 'storage': 'sqlite'}
    record('control', name, command,
           {'answer': f"storage: postgres → sqlite\n{entry['changed_at']} · запись добавлена в журнал",
            'status': 'changed'},
           store.active() == changed and store.history() == [{'id': 1, **entry}]
           and entry['old_value'] == 'postgres' and entry['new_value'] == 'sqlite')
    ask('control', 'Новый вариант после команды', 'Предложи вариант с SQLite.', 'accepted')
    ask('control', 'Старый вариант после команды', 'Предложи вариант с PostgreSQL.', 'conflict', ['storage'])

    name, command = 'Сброс диалога и открытие хранилища заново', '/reset'
    begin(name, command, 'local')
    agent.reset()
    reopened = InvariantStore(store.path)
    agent.invariants = reopened
    record('control', name, command,
           {'answer': 'Short-term и working очищены. Действующие правила доступны в /invariants.', 'status': 'reset'},
           not memory.short_term and not memory.working and reopened.active() == changed
           and reopened.history() == [{'id': 1, **entry}]
           and agent.context()[1]['content'] == rules_context(changed))
    ask('control', 'Базовый запрос после сброса', 'Предложи базовую конфигурацию.', 'accepted')
    ask('control', 'Два конфликтующих свойства', 'Публичный доступ и микросервисы.',
        'conflict', ['architecture', 'access'])
    ask('control', 'Запрос за пределами каталога',
        'Можно ли разработать нативное мобильное приложение?', 'clarify')


def run_experiment(directory, window, client=None, on_begin=None, on_end=None):
    directory.mkdir(parents=True, exist_ok=False)
    client = client or OfflineClient()
    cases = []
    current = None
    attempts = 0

    def begin(name, prompt, source):
        nonlocal current, attempts
        current = (name, prompt, source)
        attempts += int(source == 'codex')
        if on_begin:
            on_begin(name, prompt, source)

    def record(group, name, prompt, reply, passed, source='local', application_ok=None,
               expected_status=None, visible_result_ok=None):
        case = {'group': group, 'name': name, 'prompt': prompt, 'reply': reply, 'passed': passed,
                'source': source, 'application_ok': passed if application_ok is None else application_ok,
                'expected_status': expected_status, 'visible_result_ok': visible_result_ok}
        cases.append(case)
        if on_end:
            on_end(case)

    error = None
    with TemporaryDirectory(prefix='lab14-') as scratch:
        root = Path(scratch)
        store = InvariantStore(root / 'invariants.sqlite3')
        memory = MemoryStore(root / 'long_term.json', window)
        agent = Agent(client, memory, store)
        try:
            _run_cases(agent, begin, record)
        except (RuntimeError, KeyboardInterrupt) as exc:
            error = 'Работа остановлена.' if isinstance(exc, KeyboardInterrupt) else str(exc)
            if current and on_end:
                name, prompt, source = current
                on_end({'name': name, 'prompt': prompt, 'source': source, 'passed': False,
                        'reply': {'answer': error, 'status': 'error'}})
        audit = store.history()

    bypasses = [case for case in cases if case['group'] == 'bypass']
    malformed = [case for case in cases if case['group'] == 'malformed']
    model_cases = [case for case in cases if case['source'] == client.mode]
    statuses = Counter(case['reply']['status'] for case in bypasses)
    usages = [case['reply']['usage'] for case in model_cases if case['reply']['usage'] is not None]
    report = {
        'model': client.model_name, 'mode': client.mode, 'window': window, 'api_calls': 0,
        'status': 'interrupted' if error else 'completed', 'error': error,
        'codex_calls': len(usages), 'codex_attempts': attempts,
        'tokens': sum(usage['total_tokens'] for usage in usages) if usages else None,
        'cases': cases, 'passed': sum(case['passed'] for case in cases), 'total': len(cases),
        'model_matched': sum(case['passed'] for case in model_cases), 'model_total': len(model_cases),
        'application_passed': sum(case['application_ok'] for case in cases),
        'bypasses': len(bypasses), 'bypasses_planned': len(BYPASSES),
        'bypasses_accepted': sum(not case['visible_result_ok'] for case in bypasses
                                 if case['reply']['status'] == 'accepted'),
        'bypasses_blocked': sum(case['passed'] for case in bypasses),
        'bypass_model_outcomes': dict(statuses),
        'malformed': len(malformed), 'malformed_blocked': sum(case['passed'] for case in malformed),
        'audit': audit,
    }
    write_report(report, directory)
    return report
