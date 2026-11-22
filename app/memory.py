"""Упорядоченные по времени факты, история изменений и память диалога."""

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile


HISTORY_KEYS = {'решение', 'история изменений'}
ACTIVE_HEADER = 'Актуальные факты долговременной памяти:\n'
WORKING_HEADER = 'Данные текущей задачи:\n'


def timestamp(value=None):
    if value is None:
        moment = datetime.now(timezone.utc)
    else:
        iso = value[:-1] + '+00:00' if value.endswith('Z') else value
        moment = datetime.fromisoformat(iso)
        if moment.tzinfo is None:
            raise ValueError('Укажите часовой пояс во времени записи, например +03:00.')
    return moment.astimezone(timezone.utc).isoformat(timespec='microseconds')


def normalize_key(key):
    normalized = ' '.join(key.casefold().split())
    if not normalized:
        raise ValueError('Укажите ключ факта.')
    return normalized


def _empty():
    return {'active': {}, 'history': {}, 'tombstones': {}, 'changes': [], 'revision': 0}


class DurableMemory:
    def __init__(self, path, seed_path=None):
        self.path = Path(path)
        self.seed_path = seed_path

    def read(self):
        source = self.path if self.path.exists() else self.seed_path
        if source is None:
            return _empty()
        with Path(source).open(encoding='utf-8') as file:
            state = json.load(file)
        if not isinstance(state, dict) or set(state) != set(_empty()):
            raise ValueError('Файл долговременной памяти имеет неподдерживаемую структуру.')
        return state

    @contextmanager
    def _transaction(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + '.lock')
        with lock_path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield self.read()

    def _write(self, state):
        state['revision'] += 1
        temporary = None
        try:
            with NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.path.parent,
                                    delete=False) as file:
                temporary = Path(file.name)
                json.dump(state, file, ensure_ascii=False, indent=2)
                file.write('\n')
                file.flush()
                os.fsync(file.fileno())
            temporary.replace(self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def record(self, key, value, recorded_at=None):
        key, value = normalize_key(key), value.strip()
        if not value:
            raise ValueError('Укажите непустое значение факта.')
        record = {'value': value, 'recorded_at': timestamp(recorded_at)}
        with self._transaction() as state:
            if key in HISTORY_KEYS:
                history = state['history'].setdefault(key, [])
                if record in history:
                    return 'unchanged'
                history.append(record)
                history.sort(key=lambda item: item['recorded_at'])
                self._write(state)
                return 'appended'

            old = state['active'].get(key)
            latest = old['recorded_at'] if old else state['tombstones'].get(key)
            if latest and record['recorded_at'] <= latest:
                return 'unchanged' if old == record else 'ignored'

            state['active'][key] = record
            state['tombstones'].pop(key, None)
            if old and old['value'] != value:
                state['changes'].append({
                    'key': key, 'operation': 'replace', 'before': old, 'after': record,
                    'recorded_at': record['recorded_at'],
                })
            self._write(state)
            if old is None:
                return 'added'
            return 'refreshed' if old['value'] == value else 'replaced'

    def _remove(self, state, key, recorded_at):
        old = state['active'].get(key)
        latest = old['recorded_at'] if old else state['tombstones'].get(key)
        if latest and recorded_at <= latest:
            return 'ignored'
        state['active'].pop(key, None)
        state['tombstones'][key] = recorded_at
        if old:
            state['changes'].append({
                'key': key, 'operation': 'delete', 'before': old, 'after': None,
                'recorded_at': recorded_at,
            })
        return 'deleted'

    def forget(self, key, recorded_at=None):
        key, recorded_at = normalize_key(key), timestamp(recorded_at)
        if key in HISTORY_KEYS:
            raise ValueError('Накопительные записи очищаются командой /clear long.')
        with self._transaction() as state:
            result = self._remove(state, key, recorded_at)
            if result == 'deleted':
                self._write(state)
            return result

    def clear(self):
        now = timestamp()
        with self._transaction() as state:
            for key, record in list(state['active'].items()):
                # Явная очистка отменяет даже импортированные записи из будущего.
                latest = datetime.fromisoformat(record['recorded_at'])
                if record['recorded_at'] >= now:
                    removed_at = timestamp((latest + timedelta(microseconds=1)).isoformat())
                else:
                    removed_at = now
                self._remove(state, key, removed_at)
            state['history'].clear()
            self._write(state)


class MemoryStore:
    def __init__(self, path, window=4, seed_path=None):
        if window < 1:
            raise ValueError('Размер окна должен быть не меньше 1 сообщения.')
        self.durable = DurableMemory(path, seed_path)
        self.window = window
        self.short_term = []
        self.working = {}
        self._context_start = 0
        self._revision = self.durable.read()['revision']

    def record_working(self, key, value):
        value = value.strip()
        if not value:
            raise ValueError('Укажите непустое значение.')
        self.working[normalize_key(key)] = {'value': value, 'recorded_at': timestamp()}
        self.invalidate_context()

    def add_turn(self, text, answer):
        self.short_term += [{'role': 'user', 'content': text},
                            {'role': 'assistant', 'content': answer}]

    def invalidate_context(self):
        self._context_start = len(self.short_term)

    def reset_dialogue(self):
        self.short_term.clear()
        self.working.clear()
        self._context_start = 0

    def finish_task(self):
        self.working.clear()
        self.invalidate_context()

    def clear(self, layer):
        if layer == 'short':
            self.short_term.clear()
            self._context_start = 0
        elif layer == 'working':
            self.finish_task()
        elif layer == 'long':
            self.durable.clear()
            self.invalidate_context()
        else:
            raise ValueError('Укажите слой short, working или long.')

    def context(self, system_prompt, text=None):
        state = self.durable.read()
        if self._revision != state['revision']:
            self.invalidate_context()
            self._revision = state['revision']
        current = {key: record['value'] for key, record in state['active'].items()}
        active_values = {value.casefold() for value in current.values()}
        retired = {event['before']['value'] for event in state['changes']
                   if event['before']['value'].casefold() not in active_values}
        obsolete = sorted(retired, key=lambda value: (-len(value), value))
        pattern = (re.compile(r'(?<!\w)(?:' + '|'.join(map(re.escape, obsolete)) + r')(?!\w)',
                              re.IGNORECASE) if obsolete else None)

        def redact(content):
            return pattern.sub('[устаревшее значение]', content) if pattern else content

        # Совпадающий рабочий ключ не восстанавливает удалённый долговременный факт.
        working = {key: redact(record['value']) for key, record in self.working.items()
                   if key not in current and key not in state['tombstones']}
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'system', 'content': ACTIVE_HEADER + json.dumps(current, ensure_ascii=False)},
            {'role': 'system', 'content': WORKING_HEADER + json.dumps(working, ensure_ascii=False)},
        ]
        start = max(self._context_start, len(self.short_term) - self.window)
        messages.extend({'role': item['role'], 'content': redact(item['content'])}
                        for item in self.short_term[start:])
        if text is not None:
            messages.append({'role': 'user', 'content': redact(text)})
        return messages
