"""Модель памяти ассистента: три слоя, хранятся и обновляются отдельно."""

import json
from copy import deepcopy
from pathlib import Path


# Правило для working-слоя: служебный LLM-вызов после реплики пользователя.
WORKING_SYSTEM_PROMPT = (
    'Ты — извлекатель данных о текущей задаче пользователя. Тебе дают текущий словарь '
    'и новое сообщение пользователя. Верни ОБНОВЛЁННЫЙ словарь как чистый JSON без '
    'markdown и без пояснений: ключи — короткие названия (цель, даты, бюджет, '
    'ограничения, предпочтения, договорённости), значения — конкретные данные. '
    'Устаревшее удаляй, новое добавляй, существующее уточняй. '
    'Предпочтения о стиле ответов обрабатываются отдельно и в словарь задачи не входят.')

# Правила попадания каждого слоя в промпт (см. MemoryStore.context).
LONG_TERM_HEADER = 'Долговременная память (профиль, решения, знания):'
WORKING_HEADER = 'Данные текущей задачи:'


class MemoryStore:
    """Слои памяти и правила сборки промпта из них.

    short-term — сообщения текущего диалога (список). В промпт попадают
      последние window сообщений дословно; текущая реплика — всегда.
    working — данные текущей задачи (словарь как sticky facts Дня 10).
      Обновляется служебным LLM-вызовом после реплики пользователя,
      в промпт попадает целиком.
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
        self.side_calls = 0
        self.side_tokens = 0

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

    # ---- рабочая память: автоматическое обновление служебным вызовом ----

    def update_working(self, user_text, chat, profile_block=None):
        prompt = [
            {'role': 'system', 'content': WORKING_SYSTEM_PROMPT},
            {'role': 'user', 'content':
             f'Текущий словарь данных задачи:\n{json.dumps(self.working, ensure_ascii=False)}\n\n'
             f'Новое сообщение пользователя:\n{user_text}'},
        ]
        if profile_block is not None:
            prompt.insert(1, {'role': 'system', 'content': profile_block})
        text, usage = chat(prompt)
        if usage is not None:
            self.side_calls += 1
            self.side_tokens += usage['total_tokens']
        payload = text.strip()
        if payload.startswith('```') and payload.endswith('```'):
            payload = '\n'.join(payload.splitlines()[1:-1])
        facts = json.loads(payload)
        if not isinstance(facts, dict):
            raise ValueError('Данные задачи должны быть словарём ключ-значение.')
        self.working = facts

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
        """Новый диалог: short и working очищаются, long-term остаётся."""
        self.clear('short')
        self.clear('working')

    # ---- сборка промпта из слоёв по правилам ----

    def context(self, system_prompt, skip=(), profile_block=None):
        """system → long-term блок → блок профиля → working блок → окно N.

        skip — множество слоёв, которые НЕ попадают в промпт (для ablation).
        profile_block — текст блока «Профиль пользователя и предпочтения»
        (День 12): отдельная сущность поверх слоёв, инжектится явно и
        измеримо; 'profile' в skip отключает блок, не трогая слои памяти.
        Последняя реплика пользователя входит всегда, иначе задать вопрос
        невозможно.
        """
        messages = [{'role': 'system', 'content': system_prompt}]
        if 'long' not in skip and self.long_term:
            messages.append({'role': 'system', 'content':
                             LONG_TERM_HEADER + '\n'
                             + json.dumps(self.long_term, ensure_ascii=False, indent=2)})
        if profile_block is not None and 'profile' not in skip:
            messages.append({'role': 'system', 'content': profile_block})
        if 'working' not in skip and self.working:
            messages.append({'role': 'system', 'content':
                             WORKING_HEADER + '\n'
                             + json.dumps(self.working, ensure_ascii=False, indent=2)})
        if 'short' not in skip:
            tail = self.short_term[-self.window:]
        else:
            tail = self.short_term[-1:]  # только текущая реплика
        return messages + deepcopy(tail)

    # ---- снимок для ablation: воспроизводимое состояние слоёв ----

    def snapshot(self):
        return {'short_term': deepcopy(self.short_term), 'working': deepcopy(self.working)}

    def restore(self, state):
        self.short_term = deepcopy(state['short_term'])
        self.working = deepcopy(state['working'])
