"""Агент: диалог поверх трёхслойной памяти из app/memory.py."""

from .scenario import SYSTEM_PROMPT


class Agent:
    def __init__(self, client, memory):
        self.client = client
        self.memory = memory
        self.system_prompt = SYSTEM_PROMPT

    def set_window(self, window):
        if window < 1:
            raise ValueError('Размер окна должен быть не меньше 1 сообщения.')
        self.memory.window = window

    def ask(self, text, skip=()):
        """Один обмен: реплика пользователя → обновление working → ответ.

        skip — слои, не попадающие в промпт этого запроса (для эксперимента).
        """
        if not text.strip():
            raise ValueError('Введите непустое сообщение.')
        previous = self.memory.snapshot()
        before_tokens, before_calls = self.memory.side_tokens, self.memory.side_calls
        self.memory.short_term.append({'role': 'user', 'content': text})
        try:
            if 'working' not in skip:
                self.memory.update_working(text, self.client.complete)
            answer, usage = self.client.complete(
                self.memory.context(self.system_prompt, skip))
        except (ValueError, RuntimeError, KeyboardInterrupt):
            self.memory.restore(previous)
            raise
        self.memory.short_term.append({'role': 'assistant', 'content': answer})
        return {'answer': answer, 'usage': usage,
                'side_tokens': self.memory.side_tokens - before_tokens,
                'side_calls': self.memory.side_calls - before_calls}

    def reset(self):
        """Новый диалог: short и working очищаются, long-term переживает сброс."""
        self.memory.reset_dialogue()
