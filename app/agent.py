"""Агент владеет историей, стратегией памяти и снимками веток."""

from copy import deepcopy

from .context import make_strategy
from .scenario import SYSTEM_PROMPT


class Agent:
    def __init__(self, client, strategy='sliding', window=4):
        self.client = client
        self.messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
        self.strategy = make_strategy(strategy, window)
        self.current = 'main'
        self.branches = {self.current: self.snapshot()}

    def snapshot(self):
        return deepcopy({'messages': self.messages, 'strategy': self.strategy})

    def reset(self):
        self.messages = self.messages[:1]
        self.strategy = make_strategy(self.strategy.name, self.strategy.window)

    def set_strategy(self, name):
        self.strategy = make_strategy(name, self.strategy.window)

    def set_window(self, window):
        if window < 2:
            raise ValueError('Размер окна должен быть не меньше 2 сообщений.')
        self.strategy.window = window

    def checkpoint(self, name, switch=False):
        if not name:
            raise ValueError('Укажите имя снимка или ветки.')
        if name in self.branches:
            raise ValueError(f'Ветка «{name}» уже существует. Выберите другое имя.')
        self.branches[self.current] = self.snapshot()
        self.branches[name] = self.snapshot()
        if switch:
            self.current = name

    def switch(self, name):
        if name not in self.branches:
            raise ValueError(f'Нет ветки «{name}». Доступно: {", ".join(self.branches)}')
        self.branches[self.current] = self.snapshot()
        state = deepcopy(self.branches[name])
        self.messages, self.strategy = state['messages'], state['strategy']
        self.current = name

    def tree(self):
        self.branches[self.current] = self.snapshot()
        return [{'name': name, 'current': name == self.current,
                 'messages': len(state['messages']) - 1, 'strategy': state['strategy'].name}
                for name, state in self.branches.items()]

    def ask(self, text):
        if not text.strip():
            raise ValueError('Введите непустое сообщение.')
        previous = self.snapshot()
        before_tokens, before_calls = self.strategy.side_tokens, self.strategy.side_calls
        self.messages.append({'role': 'user', 'content': text})
        try:
            self.strategy.after_user_message(self.messages, self.client.complete)
            answer, usage = self.client.complete(self.strategy.context(self.messages))
        except (ValueError, RuntimeError, KeyboardInterrupt):
            self.messages, self.strategy = previous['messages'], previous['strategy']
            raise
        self.messages.append({'role': 'assistant', 'content': answer})
        warning = None
        try:
            self.strategy.after_exchange(self.messages, self.client.complete)
        except (ValueError, RuntimeError) as error:
            warning = str(error)
        return {'answer': answer, 'usage': usage, 'warning': warning,
                'side_tokens': self.strategy.side_tokens - before_tokens,
                'side_calls': self.strategy.side_calls - before_calls}
