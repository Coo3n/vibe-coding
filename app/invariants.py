"""Хранилище правил и проверка структурированного ответа."""

from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3


FIELDS = {
    'architecture': ('Архитектура', {'monolith': 'Модульный монолит', 'services': 'Микросервисы'}),
    'runtime': ('Стек', {'python': 'Python', 'node': 'Node.js'}),
    'storage': ('Хранилище', {'postgres': 'PostgreSQL', 'sqlite': 'SQLite'}),
    'access': ('Бизнес-правило доступа', {'private': 'Только для команды', 'public': 'Публичный доступ'}),
}
DEFAULTS = {
    'architecture': 'monolith', 'runtime': 'python', 'storage': 'postgres', 'access': 'private',
}


def display(field, value):
    return FIELDS[field][1][value]


class InvariantStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute('BEGIN IMMEDIATE')
            initialized = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'rules'"
            ).fetchone()
            connection.execute('''CREATE TABLE IF NOT EXISTS rules (
                field TEXT PRIMARY KEY, value TEXT NOT NULL
            )''')
            connection.execute('''CREATE TABLE IF NOT EXISTS audit (
                id INTEGER PRIMARY KEY, changed_at TEXT NOT NULL, field TEXT NOT NULL,
                old_value TEXT NOT NULL, new_value TEXT NOT NULL
            )''')
            if not initialized:
                connection.executemany('INSERT INTO rules (field, value) VALUES (?, ?)', DEFAULTS.items())
            self._read(connection)

    @staticmethod
    def _read(connection):
        rules = dict(connection.execute('SELECT field, value FROM rules'))
        if set(rules) != set(FIELDS) or any(value not in FIELDS[key][1] for key, value in rules.items()):
            raise ValueError('Хранилище инвариантов содержит недопустимый набор правил.')
        return rules

    def active(self):
        with closing(sqlite3.connect(self.path)) as connection:
            return self._read(connection)

    def set(self, field, value):
        if field not in FIELDS:
            raise ValueError('Укажите инвариант из списка /invariants.')
        if value not in FIELDS[field][1]:
            raise ValueError(f'Допустимые значения для {field}: {", ".join(FIELDS[field][1])}.')
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute('BEGIN IMMEDIATE')
            before = self._read(connection)[field]
            if before == value:
                raise ValueError(f'Инвариант {field} уже имеет значение {value}.')
            entry = (datetime.now(timezone.utc).isoformat(), field, before, value)
            connection.execute('UPDATE rules SET value = ? WHERE field = ?', (value, field))
            connection.execute('''INSERT INTO audit (changed_at, field, old_value, new_value)
                VALUES (?, ?, ?, ?)''', entry)
        return dict(zip(('changed_at', 'field', 'old_value', 'new_value'), entry))

    def history(self):
        with closing(sqlite3.connect(self.path)) as connection:
            connection.row_factory = sqlite3.Row
            return [dict(row) for row in connection.execute('SELECT * FROM audit ORDER BY id')]


def _unique_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError('В ответе есть повторяющиеся поля.')
    return result


def check_reply(raw, rules):
    """Вернуть статус, подготовленный кодом текст и список конфликтующих полей."""
    invalid = ('invalid', 'Ответ отклонён. Для проверки нужен вариант из каталога проекта. '
               'Уточните архитектуру, стек, хранилище или режим доступа.', [])
    try:
        offer = json.loads(raw, object_pairs_hook=_unique_object)
    except (ValueError, TypeError, RecursionError):
        return invalid
    if not isinstance(offer, dict):
        return invalid
    if offer == {'action': 'clarify'}:
        return ('clarify', 'Уточните, какой вариант архитектуры, стека, хранилища '
                'или доступа вы хотите рассмотреть. Активные правила доступны в /invariants.', [])
    if set(offer) != set(FIELDS) | {'action'} or offer['action'] != 'propose':
        return invalid
    if any(not isinstance(offer[key], str) or offer[key] not in FIELDS[key][1] for key in FIELDS):
        return invalid
    conflicts = [key for key in FIELDS if offer[key] != rules[key]]
    if conflicts:
        details = '\n'.join(f'• {FIELDS[key][0]}: запрошено «{display(key, offer[key])}», '
                            f'действует «{display(key, rules[key])}».' for key in conflicts)
        return ('conflict', 'Вариант отклонён. Он нарушает активные инварианты.\n'
                + details + '\nИзменить правило можно командой /invariant set.', conflicts)
    details = '\n'.join(f'• {FIELDS[key][0]}: {display(key, offer[key])}.' for key in FIELDS)
    return 'accepted', 'Допустимая конфигурация проекта\n' + details, []
