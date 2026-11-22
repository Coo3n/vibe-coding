"""Агент: диалог поверх трёхслойной памяти и профиля пользователя (День 12).

Профиль — отдельная сущность ПОВЕРХ слоёв памяти: подключается к сборке
промпта явно (system-блок после long-term) и пополняется из диалога
служебным LLM-вызовом после каждой реплики пользователя.
"""

from .scenario import SYSTEM_PROMPT


class Agent:
    def __init__(self, client, memory, profile=None):
        self.client = client
        self.memory = memory
        self.profile = profile
        self.system_prompt = SYSTEM_PROMPT

    def set_window(self, window):
        if window < 1:
            raise ValueError('Размер окна должен быть не меньше 1 сообщения.')
        self.memory.window = window

    def ask(self, text, skip=()):
        """Один обмен: реплика → извлечение предпочтений → обновление working → ответ.

        Порядок важен: предпочтения из реплики попадают в профиль ДО основного
        вызова, поэтому ответ на «теперь отвечай подробно» уже подробный.
        skip — слои/профиль, не попадающие в промпт этого запроса ('short',
        'working', 'long', 'profile').
        """
        if not text.strip():
            raise ValueError('Введите непустое сообщение.')
        previous = self.memory.snapshot()
        before_tokens, before_calls = self.memory.side_tokens, self.memory.side_calls
        before_ptokens = self.profile.side_tokens if self.profile else 0
        before_pcalls = self.profile.side_calls if self.profile else 0
        self.memory.short_term.append({'role': 'user', 'content': text})
        try:
            profile_changes = []
            if self.profile is not None and 'profile' not in skip:
                profile_changes = self.profile.update_from_dialogue(
                    text, self.client.complete)
            if 'working' not in skip:
                block = (self.profile.block(enforce=False)
                         if self.profile is not None and 'profile' not in skip else None)
                self.memory.update_working(text, self.client.complete, block)
            answer, usage = self.client.complete(
                self.memory.context(self.system_prompt, skip,
                                    profile_block=self.profile.block() if self.profile else None))
        except (OSError, ValueError, RuntimeError, KeyboardInterrupt):
            self.memory.restore(previous)
            raise
        self.memory.short_term.append({'role': 'assistant', 'content': answer})
        return {'answer': answer, 'usage': usage,
                'side_tokens': self.memory.side_tokens - before_tokens,
                'side_calls': self.memory.side_calls - before_calls,
                'profile_changes': profile_changes,
                'profile_side_tokens': (self.profile.side_tokens - before_ptokens)
                                       if self.profile else 0,
                'profile_side_calls': (self.profile.side_calls - before_pcalls)
                                      if self.profile else 0}

    def reset(self):
        """Новый диалог: short и working очищаются, long-term переживает сброс."""
        self.memory.reset_dialogue()
