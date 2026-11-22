"""Модель памяти ассистента: три слоя, хранятся и обновляются отдельно."""

import json
from copy import deepcopy
from pathlib import Path


LONG_TERM_HEADER = 'Долговременная память (профиль, решения, знания):'
WORKING_HEADER = 'Данные текущей задачи:'


class MemoryStore:
    """Слои памяти и правила сборки промпта из них.

    short-term — сообщения текущего диалога (список). В промпт попадают
      последние window сообщений дословно; текущая реплика — всегда.
    working — сводка сохранённого состояния задачи, целиком попадает в промпт.
    long-term — решения, знания, профиль. Хранится на диске отдельно
      (переживает перезапуск), пополняется ТОЛЬКО явно (remember/forget),
      в промпт попадает целиком.
    """

    def __init__(self, long_term_path, window=4):
        if window < 1:
            raise ValueError('Размер окна должен быть не меньше 1 сообщения.')
        self.window = window
        self.long_term_path = Path(long_term_path)
        self.short_term = []
        self.working = {}
        self.long_term = self._load_long_term()

    # ---- долговременная память: диск и явные операции ----

    def _load_long_term(self):
        if self.long_term_path.exists():
            data = json.loads(self.long_term_path.read_text(encoding='utf-8'))
            if not isinstance(data, dict):
                raise ValueError('long_term.json должен содержать JSON-словарь.')
            return data
        return {}

    def _save_long_term(self):
        self.long_term_path.parent.mkdir(parents=True, exist_ok=True)
        self.long_term_path.write_text(
            json.dumps(self.long_term, ensure_ascii=False, indent=2), encoding='utf-8')

    def remember(self, text):
        """Явное сохранение в long-term: «ключ: значение» или текст под автоключом."""
        key, _, value = text.partition(':')
        if value.strip() and key.strip():
            key, value = key.strip(), value.strip()
        else:
            key, value = f'запись {len(self.long_term) + 1}', text.strip()
        if not value:
            raise ValueError('Укажите, что запомнить: /remember текст')
        self.long_term[key] = value
        self._save_long_term()
        return key

    def forget(self, key):
        if key not in self.long_term:
            raise ValueError(f'Ключа «{key}» нет в долговременной памяти. '
                             f'Доступно: {", ".join(self.long_term) or "—"}')
        del self.long_term[key]
        self._save_long_term()

    # ---- очистка слоёв ----

    def clear(self, layer):
        if layer == 'short':
            self.short_term = []
        elif layer == 'working':
            self.working = {}
        elif layer == 'long':
            self.long_term = {}
            self._save_long_term()
        else:
            raise ValueError('Слой: short, working или long.')

    def reset_dialogue(self):
        """Новый диалог очищает короткую историю и локальную сводку задачи."""
        self.clear('short')
        self.clear('working')

    # ---- сборка промпта из слоёв по правилам ----

    def context(self, system_prompt, current=None):
        """system → long-term → working → окно сообщений с текущей репликой."""
        messages = [{'role': 'system', 'content': system_prompt}]
        if self.long_term:
            messages.append({'role': 'system', 'content':
                             LONG_TERM_HEADER + '\n'
                             + json.dumps(self.long_term, ensure_ascii=False, indent=2)})
        if self.working:
            messages.append({'role': 'system', 'content':
                             WORKING_HEADER + '\n'
                             + json.dumps(self.working, ensure_ascii=False, indent=2)})
        tail = self.short_term[-self.window:]
        if current is not None:
            tail = (tail + [{'role': 'user', 'content': current}])[-self.window:]
        return messages + deepcopy(tail)
