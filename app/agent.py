"""Агент предлагает текст, а переходами управляют команды и конечный автомат."""

import json

from .state import Stage, TransitionError


SYSTEM_PROMPT = (
    'Ты помогаешь провести задачу через planning, execution, validation и done. '
    'Состояние из рабочей памяти определяет допустимую работу на текущем этапе. '
    'Только пользователь может утвердить план и зафиксировать итог проверки через команды. '
    '/plan готовит черновик и оставляет этап planning. /approve утверждает сохранённый план. '
    '/execute сохраняет результат и открывает validation. /validate показывает чек-лист, '
    '/validate pass фиксирует успешную проверку, а /finish завершает проверенную задачу. '
    'Если подсказываешь следующую команду, бери её дословно из поля expected_action. '
    'Не обещай совершённых переходов или внешних действий. '
    'На этапе planning помогай уточнять цель и план, execution — готовить черновик результата, '
    'validation — проверять результат, done — объяснять зафиксированный итог.'
)

ACTION_PROMPTS = {
    'reply': 'Ответь на реплику с учётом текущего шага.',
    'plan': 'Составь конкретный план из 2–4 шагов для цели задачи. Выведи только план.',
    'execute': 'Подготовь только черновик результата в рамках утверждённого плана.',
    'review': 'Выведи только список конкретных проверок для результата.',
}


class OfflineClient:
    """Локальные реплики для запуска интерфейса и сценария без сети."""

    model_name = 'локальные шаблоны'

    def complete(self, messages):
        context = next(
            json.loads(message['content'].split('\n', 1)[1])
            for message in messages
            if message['role'] == 'system' and message['content'].startswith('Локальный вызов\n')
        )
        action = context['action']
        state = context['state']
        goal = state['goal']
        if action == 'plan':
            answer = (f'Цель — {goal}\n1. Уточнить границы задачи.\n'
                      '2. Подготовить черновик результата.\n3. Проверить его по цели и ограничениям.')
        elif action == 'execute':
            answer = (f'Черновик результата для задачи «{goal}».\n'
                      f'Утверждённый план\n{state["plan"]}\n'
                      'Сверьте конкретный результат с планом при проверке.')
        elif action == 'review':
            answer = ('Проверьте, отвечает ли результат цели, покрывает ли утверждённый план '
                      'и соблюдает ли ограничения. Запишите наблюдения через /validate pass|fail.')
        else:
            answer = (f'Цель задачи — {goal}. Следующий шаг — {state["step"]}. '
                      f'Ожидается {state["expected_action"]}. '
                      'Из обычной реплики переходы состояния не выполняются.')
        return json.dumps({'action': action, 'answer': answer}, ensure_ascii=False), None


class ModelReplyError(TransitionError):
    def __init__(self, message, raw, usage):
        super().__init__(message)
        self.raw = raw
        self.usage = usage


class Agent:
    def __init__(self, client, memory, store):
        self.client = client
        self.memory = memory
        self.store = store
        self.system_prompt = SYSTEM_PROMPT

    def state(self):
        state = self.store.read()
        if state is None:
            raise TransitionError('Сначала создайте задачу через /task <цель>.')
        return state

    def start(self, goal):
        state = self.store.start(goal)
        self.reset()
        return state

    def reset(self):
        self.memory.reset_dialogue()

    def context(self, state=None, current=None):
        if state is None:
            state = self.store.read()
        self.memory.working = state.context() if state else {}
        return self.memory.context(self.system_prompt, current)

    def ask(self, text):
        if not text.strip():
            raise ValueError('Введите непустое сообщение.')
        state = self.state()
        if state.paused or state.stage == Stage.DONE:
            return self.status(state), None
        message, usage = self._complete('reply', state, text)
        self.memory.short_term.extend((
            {'role': 'user', 'content': text},
            {'role': 'assistant', 'content': message},
        ))
        return message, usage

    def perform(self, action, content='', passed=None):
        state = self.state()
        state.require(action)
        usage = None
        if action in ('plan', 'execute', 'review') and not content.strip():
            content, usage = self._complete(action, state, f'Цель задачи — {state.goal}')
        if action == 'review':
            return state, content, usage
        updated = self.store.update(state, action, content.strip(), passed)
        return updated, content, usage

    @staticmethod
    def status(state):
        info = state.context()
        text = (f'Цель — {info["goal"]}\nЭтап · {info["stage"]}\n'
                f'Текущий шаг · {info["step"]}\nОжидается · {info["expected_action"]}\n'
                f'Пауза · {"да" if info["paused"] else "нет"}')
        for key, label in (('plan', 'План'), ('artifact', 'Результат'), ('validation', 'Проверка')):
            if info[key]:
                text += f'\n\n{label}\n{info[key]}'
        if state.passed is not None:
            text += ('\nИтог проверки · пройдена' if state.passed else '\nИтог проверки · не пройдена')
        return text

    def _complete(self, action, state, text):
        messages = self.context(state, text)
        instruction = (
            ACTION_PROMPTS[action] + ' Верни только JSON с двумя ключами. '
            f'Поле action равно "{action}", поле answer содержит текст.'
        )
        messages.insert(1, {'role': 'system', 'content': instruction})
        if isinstance(self.client, OfflineClient):
            request_context = json.dumps({'action': action, 'state': state.context()}, ensure_ascii=False)
            messages.insert(1, {'role': 'system', 'content': 'Локальный вызов\n' + request_context})
        raw, usage = self.client.complete(messages)
        payload = raw.strip()
        if payload.startswith('```') and payload.endswith('```'):
            payload = '\n'.join(payload.splitlines()[1:-1])
        try:
            reply = json.loads(payload)
        except json.JSONDecodeError:
            raise ModelReplyError('Ответ модели должен быть JSON. Состояние сохранено.',
                                  raw, usage) from None
        if not isinstance(reply, dict) or set(reply) != {'action', 'answer'}:
            raise ModelReplyError('В ответе модели нужны поля action и answer. Состояние сохранено.',
                                  raw, usage)
        proposed = reply['action']
        if proposed != action:
            expected = state.context()['expected_action']
            raise ModelReplyError(f'Агент предложил действие {proposed!r}. '
                                  f'Действие отклонено на этапе {state.stage.value}. Ожидается {expected}.',
                                  raw, usage)
        answer = reply['answer']
        if not isinstance(answer, str) or not answer.strip():
            raise ModelReplyError('Ответ модели должен содержать текст в answer.', raw, usage)
        return answer.strip(), usage
