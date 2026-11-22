"""Конечный автомат задачи и атомарное сохранение состояния."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from enum import Enum
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional
from uuid import uuid4

import fcntl


class TransitionError(ValueError):
    pass


class Stage(str, Enum):
    PLANNING = 'planning'
    EXECUTION = 'execution'
    VALIDATION = 'validation'
    DONE = 'done'


ACTION_COMMANDS = {
    'plan': '/plan',
    'approve': '/approve',
    'execute': '/execute',
    'validate': '/validate pass|fail <результат>',
    'review': '/validate',
    'revise': '/revise <результат>',
    'finish': '/finish',
    'start': '/task <цель>',
    'pause': '/pause',
    'resume': '/resume',
}


@dataclass(frozen=True)
class TaskState:
    task_id: str
    goal: str
    stage: Stage = Stage.PLANNING
    paused: bool = False
    plan: str = ''
    artifact: str = ''
    validation: str = ''
    passed: Optional[bool] = None

    @classmethod
    def from_record(cls, record):
        if not isinstance(record, dict):
            raise TransitionError('Файл задачи должен содержать JSON-объект.')
        try:
            state = cls(**{**record, 'stage': Stage(record['stage'])})
        except (TypeError, KeyError, ValueError) as error:
            raise TransitionError('Некорректные поля файла задачи.') from error
        if not state.is_valid():
            raise TransitionError('Файл задачи содержит некорректное состояние.')
        return state

    def is_valid(self):
        text_fields = (self.task_id, self.goal, self.plan, self.artifact, self.validation)
        if any(not isinstance(value, str) or (value and not value.strip()) for value in text_fields):
            return False
        if not self.task_id or not self.goal or type(self.paused) is not bool:
            return False
        if self.passed is not None and type(self.passed) is not bool:
            return False
        if bool(self.validation) != (self.passed is not None):
            return False
        if self.stage == Stage.PLANNING:
            return not self.artifact and self.passed is None
        if self.stage == Stage.EXECUTION:
            return bool(self.plan) and not self.artifact and self.passed is None
        if self.stage == Stage.VALIDATION:
            return bool(self.plan and self.artifact)
        return bool(self.plan and self.artifact) and self.passed is True

    def next_action(self):
        if self.paused:
            return 'возобновить задачу', 'resume'
        if self.stage == Stage.PLANNING:
            return ('утвердить план', 'approve') if self.plan else ('составить план', 'plan')
        if self.stage == Stage.EXECUTION:
            return 'подготовить результат', 'execute'
        if self.stage == Stage.VALIDATION:
            if self.passed is True:
                return 'завершить задачу', 'finish'
            if self.passed is False:
                return 'внести исправления', 'revise'
            return 'проверить результат', 'validate'
        return 'задача завершена', 'start'

    def allowed_actions(self):
        if self.paused:
            return {'resume'}
        actions = {'pause'}
        if self.stage == Stage.PLANNING:
            actions.add('plan')
            if self.plan:
                actions.add('approve')
        elif self.stage == Stage.EXECUTION:
            actions.add('execute')
        elif self.stage == Stage.VALIDATION:
            actions.update(('review', 'validate', 'revise'))
            if self.passed is True:
                actions.add('finish')
        else:
            actions.add('start')
        return actions

    def require(self, action):
        if action not in self.allowed_actions():
            _, expected = self.next_action()
            command = ACTION_COMMANDS.get(action, action)
            raise TransitionError(f'Действие {command} недоступно на этапе {self.stage.value}. '
                                  f'Ожидается {ACTION_COMMANDS[expected]}.')

    def context(self):
        step, expected = self.next_action()
        return {'goal': self.goal, 'stage': self.stage.value, 'step': step,
                'expected_action': ACTION_COMMANDS[expected], 'paused': self.paused,
                'plan': self.plan, 'artifact': self.artifact,
                'validation': self.validation, 'passed': self.passed}

    def apply(self, action, content='', passed=None):
        self.require(action)
        if action in ('plan', 'execute', 'revise', 'validate') and not content.strip():
            raise TransitionError('Укажите содержимое шага.')
        if action == 'plan':
            return replace(self, plan=content)
        if action == 'approve':
            return replace(self, stage=Stage.EXECUTION)
        if action == 'execute':
            return replace(self, stage=Stage.VALIDATION, artifact=content)
        if action == 'revise':
            return replace(self, artifact=content, validation='', passed=None)
        if action == 'validate':
            if type(passed) is not bool:
                raise TransitionError('Укажите результат через pass или fail.')
            return replace(self, validation=content, passed=passed)
        if action == 'finish':
            return replace(self, stage=Stage.DONE)
        if action in ('pause', 'resume'):
            return replace(self, paused=action == 'pause')
        raise TransitionError(f'У действия {action} отдельный обработчик.')


class StateStore:
    """Обновляет JSON под блокировкой, сверяя состояние до вызова модели."""

    def __init__(self, path):
        self.path = Path(path)

    def read(self):
        try:
            text = self.path.read_text(encoding='utf-8')
        except FileNotFoundError:
            return None
        return TaskState.from_record(json.loads(text))

    @contextmanager
    def locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + '.lock')
        with lock_path.open('a', encoding='utf-8') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def start(self, goal):
        if not goal.strip():
            raise TransitionError('Укажите цель через /task <цель>.')
        with self.locked():
            previous = self.read()
            if previous is not None:
                previous.require('start')
            state = TaskState(uuid4().hex, goal.strip())
            self._save(state)
            return state

    def update(self, expected, action, content='', passed=None):
        with self.locked():
            state = self.read()
            if state != expected:
                raise TransitionError('Состояние изменилось в другом сеансе. Проверьте /state.')
            state = state.apply(action, content, passed)
            self._save(state)
            return state

    def _save(self, state):
        with NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.path.parent,
                                delete=False) as stream:
            temp_path = Path(stream.name)
            try:
                json.dump(asdict(state), stream, ensure_ascii=False, indent=2)
                stream.write('\n')
                stream.flush()
                temp_path.replace(self.path)
            finally:
                temp_path.unlink(missing_ok=True)
