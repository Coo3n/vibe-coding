"""Стратегии контекста: скользящее окно, ключевые факты и резюме."""

import json


def tail(history, window):
    count = min(window, len(history) - 1)
    return history[-count:] if count > 0 else []


class SlidingWindow:
    name = 'sliding'
    description = 'Только последние N сообщений; старые сообщения удаляются.'

    def __init__(self, window=4):
        self.window = window
        self.side_calls = 0
        self.side_tokens = 0

    def after_user_message(self, history, chat):
        pass

    def after_exchange(self, history, chat):
        del history[1:max(1, len(history) - self.window)]

    def context(self, history):
        return [history[0]] + tail(history, self.window)


class StickyFacts:
    name = 'sticky'
    description = 'Словарь фактов + последние N сообщений; обновление после каждой реплики пользователя.'

    def __init__(self, window=4):
        self.window = window
        self.facts = {}
        self.side_calls = 0
        self.side_tokens = 0

    def after_user_message(self, history, chat):
        prompt = [
            {'role': 'system', 'content':
             'Ты — извлекатель фактов. Тебе дают текущий словарь фактов о задаче пользователя '
             'и его новое сообщение. Верни ОБНОВЛЁННЫЙ словарь фактов как чистый JSON '
             'без markdown и без пояснений: ключи — короткие названия (цель, даты, бюджет, '
             'ограничения, предпочтения, договорённости), значения — конкретные данные. '
             'Устаревшее удаляй, новое добавляй, существующее уточняй.'},
            {'role': 'user', 'content':
             f'Текущие факты:\n{json.dumps(self.facts, ensure_ascii=False)}\n\n'
             f'Новое сообщение пользователя:\n{history[-1]["content"]}'},
        ]
        text, usage = chat(prompt)
        self.side_calls += 1
        self.side_tokens += usage['total_tokens']
        payload = text.strip()
        if payload.startswith('```') and payload.endswith('```'):
            payload = '\n'.join(payload.splitlines()[1:-1])
        facts = json.loads(payload)
        if not isinstance(facts, dict):
            raise ValueError('Факты должны быть словарём ключ-значение.')
        self.facts = facts

    def after_exchange(self, history, chat):
        pass

    def context(self, history):
        messages = [history[0]]
        if self.facts:
            messages.append({'role': 'system', 'content': 'Факты, известные о задаче пользователя:\n'
                             + json.dumps(self.facts, ensure_ascii=False, indent=2)})
        return messages + tail(history, self.window)


class SummaryStrategy:
    name = 'summary'
    description = 'Старые сообщения сворачиваются в резюме + последние N сообщений.'

    def __init__(self, window=4):
        self.window = window
        self.summary = ''
        self.folded_upto = 0
        self.side_calls = 0
        self.side_tokens = 0

    def after_user_message(self, history, chat):
        # Новая реплика тоже сдвигает окно: свёртка нужна до основного запроса.
        self.after_exchange(history, chat)

    def after_exchange(self, history, chat):
        fold_end = len(history) - self.window
        if fold_end <= self.folded_upto + 1:
            return
        messages = history[self.folded_upto + 1:fold_end]
        prompt = [
            {'role': 'system', 'content':
             'Ты сжимаешь историю диалога. Сохрани ВСЕ конкретные детали: даты, числа, '
             'имена, города, ограничения, предпочтения, договорённости. '
             'Верни только обновлённое резюме — краткими буллетами, без вступлений.'},
            {'role': 'user', 'content': f'Текущее резюме:\n{self.summary or "(пусто)"}\n\n'
             'Сообщения, которые нужно добавить в резюме:\n'
             + '\n'.join(f'[{message["role"]}] {message["content"]}' for message in messages)},
        ]
        text, usage = chat(prompt)
        self.side_calls += 1
        self.side_tokens += usage['total_tokens']
        if not text.strip():
            raise ValueError('Модель вернула пустое summary.')
        self.summary = text
        self.folded_upto = fold_end - 1

    def context(self, history):
        messages = [history[0]]
        if self.summary:
            messages.append({'role': 'system', 'content': 'Краткое резюме начала диалога:\n' + self.summary})
        # Увеличение окна не должно повторно отправлять уже свёрнутые сообщения.
        start = max(1, len(history) - self.window, self.folded_upto + 1)
        return messages + history[start:]


STRATEGY_CLASSES = {cls.name: cls for cls in (SlidingWindow, StickyFacts, SummaryStrategy)}


def make_strategy(name, window=4):
    if name not in STRATEGY_CLASSES:
        raise ValueError('Доступно: sliding, sticky, summary.')
    if window < 2:
        raise ValueError('Размер окна должен быть не меньше 2 сообщений.')
    return STRATEGY_CLASSES[name](window)
