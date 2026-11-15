"""Interactive agent; failed turns never replace saved history."""

from copy import deepcopy

from .context import ContextError
from .protocol import ProtocolError, dumps, loads, invoke, validate
from .tools import TOOL_SCHEMAS, execute


class Agent:
    def __init__(self, client, context, path=None, on_event=None, max_steps=8):
        self.client = client
        self.context = context
        self.path = path
        self.trace = []
        self.on_event = on_event or (lambda event: None)
        self.max_steps = max_steps
        self.last_turn = None
        self.reply = None

    def ask(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ContextError('Введите непустое сообщение.')
        working = deepcopy(self.context)
        self.trace = []
        # Retry any pending compaction before adding another turn.
        working.compress(self.client, self.trace)
        turn = [{'role': 'user', 'content': dumps({'message': text})}]
        used_ids = set()
        for step in range(1, self.max_steps + 1):
            self.on_event({'kind': 'request', 'step': step})
            answer = invoke(self.client, working.request_messages() + turn, 'chat', self.trace, TOOL_SCHEMAS)
            message = answer['tool_message']
            call = message['tool_calls'][0]
            if call['id'] in used_ids:
                raise ProtocolError('API повторил ID вызова инструмента.')
            used_ids.add(call['id'])
            function = call['function']
            arguments = loads(function['arguments'])
            if function['name'] == 'respond_to_user':
                answer = validate(arguments, 'chat')
                turn.extend([message, {'role': 'tool', 'tool_call_id': call['id'],
                                       'content': dumps({'status': 'accepted'})}])
                break
            result = execute(function['name'], arguments)
            turn.extend([message, {'role': 'tool', 'tool_call_id': call['id'], 'content': dumps(result)}])
            self.on_event({'kind': 'tool', 'name': function['name'], 'arguments': arguments, 'result': result})
        else:
            raise ProtocolError(f'За {self.max_steps} шагов агент не завершил ответ; ход не сохранён.')
        working.append_turn(turn)
        working.compress(self.client, self.trace)
        if self.path:
            working.save(self.path)
        self.context = working
        self.last_turn = deepcopy(turn)
        self.reply = answer
        return answer['answer']
