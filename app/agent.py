"""Диалоговый агент и локальный клиент, отвечающий по снимку промпта."""

import json
import re

from .memory import ACTIVE_HEADER, WORKING_HEADER
from .scenario import SYSTEM_PROMPT, parse_update


RECORD_LABELS = {
    'added': 'Факт записан.',
    'replaced': 'Актуальное значение обновлено, замена сохранена в журнал.',
    'appended': 'Запись добавлена в накопительную историю.',
    'ignored': 'Запись проигнорирована: в памяти есть факт с более поздним или равным временем.',
    'unchanged': 'Такая запись уже сохранена.',
    'refreshed': 'Время записи обновлено.',
    'deleted': 'Ключ удалён из актуальной памяти.',
}


class OfflineClient:
    mode = 'offline'
    model_name = 'локальный клиент'

    def complete(self, messages):
        question = messages[-1]['content'].casefold()
        fields = {}
        for label in (WORKING_HEADER, ACTIVE_HEADER):
            snapshot = next(entry['content'][len(label):] for entry in messages
                            if entry['role'] == 'system' and entry['content'].startswith(label))
            fields.update(json.loads(snapshot))
        explicit = re.fullmatch(r'(?:что сохранено|что известно) про\s+«?(.+?)»?[?\s]*', question)
        if explicit:
            matching = [(explicit[1], fields[explicit[1]])] if explicit[1] in fields else []
        else:
            words = set(re.findall(r'[\w-]+', question))
            matching = [(key, value) for key, value in fields.items()
                        if key in words or len(key.split()) > 1 and key in question]
        if not matching:
            return 'Нет актуальной записи по этому вопросу.', {}
        return '\n'.join(f'{key} — {value}' for key, value in matching), {}


class Agent:
    def __init__(self, client, memory):
        self.client = client
        self.memory = memory
        self.system_prompt = SYSTEM_PROMPT

    def ask(self, text):
        if not text.strip():
            raise ValueError('Введите непустое сообщение.')
        update = parse_update(text.strip())
        if update:
            key, value = update
            outcome = self.memory.durable.record(key, value)
            answer = RECORD_LABELS[outcome]
            if outcome != 'ignored':
                answer = f'{key} — {value}\n{answer}'
            self.memory.add_turn(text, answer)
            self.memory.invalidate_context()
            return {'answer': answer, 'usage': {}, 'messages': [], 'update': outcome}

        messages = self.memory.context(self.system_prompt, text)
        answer, usage = self.client.complete(messages)
        self.memory.add_turn(text, answer)
        return {'answer': answer, 'usage': usage, 'messages': messages}

    def reset(self):
        self.memory.reset_dialogue()
