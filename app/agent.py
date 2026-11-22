"""Диалог Lab11 с проверкой каждого ответа по текущим правилам проекта."""

from .invariants import check_reply
from .scenario import SYSTEM_PROMPT, rules_context


APPROVED_CONFIGURATION = 'Последняя допустимая конфигурация'


class Agent:
    def __init__(self, client, memory, invariants):
        self.client = client
        self.memory = memory
        self.invariants = invariants

    def _context(self, rules):
        saved = self.memory.working.get(APPROVED_CONFIGURATION)
        if saved is not None and saved != rules:
            self.memory.clear('working')
        messages = self.memory.context(SYSTEM_PROMPT)
        messages.insert(1, {'role': 'system', 'content': rules_context(rules)})
        return messages

    def context(self):
        return self._context(self.invariants.active())

    def ask(self, text, capture_raw=False):
        text = text.strip()
        if not text:
            raise ValueError('Введите непустое сообщение.')
        user = {'role': 'user', 'content': text}
        rules = self.invariants.active()
        self.memory.short_term.append(user)
        try:
            raw, usage = self.client.complete(self._context(rules))
            active = self.invariants.active()
            status, answer, conflicts = check_reply(raw, active)
        except (Exception, KeyboardInterrupt):
            self.memory.short_term.pop()
            raise
        if status == 'accepted':
            self.memory.working[APPROVED_CONFIGURATION] = active
        self.memory.short_term.append({'role': 'assistant', 'content': answer})
        reply = {'answer': answer, 'usage': usage, 'status': status, 'conflicts': conflicts}
        if capture_raw:
            reply['raw_candidate'] = raw
        return reply

    def change_invariant(self, arg):
        parts = arg.split()
        if len(parts) != 3 or parts[0] != 'set':
            raise ValueError('Формат команды /invariant set <ключ> <значение>.')
        _, field, value = parts
        entry = self.invariants.set(field, value)
        self.memory.clear('working')
        return entry

    def reset(self):
        self.memory.reset_dialogue()
