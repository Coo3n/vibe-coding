"""Агент владеет историей, протоколом tool calling и циклом до финального ответа."""

from copy import deepcopy
import json
import time

from .client import ClientError
from .tools import MAX_PRICE, TOOL_SCHEMAS


SYSTEM_PROMPT = 'Ты помогаешь выбрать квартиру. Данные учебные.'


class AgentError(RuntimeError):
    pass


def matching_apartment(events, goal):
    """Условие выполнено данными функций, а не обещанием в тексте модели."""
    found = {}
    approved = set()
    for event in events:
        result = event['result']
        if 'error' in result:
            continue
        apartment = result.get('apartment')
        if event['name'] == 'choose_apartment' and apartment and apartment['price'] <= MAX_PRICE:
            found[apartment['id']] = apartment
        elif event['name'] == 'check_mortgage' and result.get('affordable'):
            selected = found.get(result['apartment_id'])
            if selected and selected['price'] == result['price'] <= result['limit']:
                approved.add(result['apartment_id'])
    for apartment in reversed(list(found.values())):
        if goal == 'price' or (goal == 'mortgage' and apartment['id'] in approved):
            return apartment
    return None


class Agent:
    def __init__(self, client, tools, max_steps=None, on_event=None, on_progress=None):
        self.client = client
        self.tools = tools
        self.max_steps = max_steps if max_steps is not None else 2 * len(tools.catalog) + 5
        self.on_event = on_event
        self.on_progress = on_progress
        self.messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
        self.events = []
        self.trace = []
        self.goal = None

    def ask(self, text, goal=None):
        if not isinstance(text, str) or not text.strip():
            raise AgentError('Введите непустой запрос.')
        if goal not in (None, 'price', 'mortgage'):
            raise AgentError('Неизвестное условие завершения подбора.')
        self.goal = goal
        self.events = []
        self.trace = []
        start = len(self.messages)
        tool_state = (self.tools.seen.copy(), self.tools.bank_limit, self.tools.rng.getstate())
        self.messages.append({'role': 'user', 'content': text})
        try:
            return self._run()
        except (AgentError, ClientError):
            # Не оставляем незавершённый tool-протокол в следующем HTTP-запросе.
            del self.messages[start:]
            self.tools.seen, self.tools.bank_limit, rng_state = tool_state
            self.tools.rng.setstate(rng_state)
            raise

    def _run(self):
        stagnant = 0
        checked = set()
        for step in range(1, self.max_steps + 1):
            request = {'messages': deepcopy(self.messages), 'tools': deepcopy(TOOL_SCHEMAS)}
            started = time.perf_counter()
            if self.on_progress:
                self.on_progress({'kind': 'request', 'step': step})
            try:
                response = self.client.complete(**request)
            except ClientError as error:
                self.trace.append({'step': step, 'request': request, 'error': str(error),
                                   'seconds': time.perf_counter() - started})
                if self.on_progress:
                    self.on_progress({'kind': 'error', 'step': step})
                raise
            self.trace.append({'step': step, 'request': request, 'response': deepcopy(response),
                               'seconds': time.perf_counter() - started})
            if self.on_progress:
                self.on_progress({'kind': 'response', 'step': step,
                                  'seconds': self.trace[-1]['seconds']})
            message, tool_calls = self._parse(response)
            self.messages.append(message)
            if not tool_calls:
                feedback = self._completion_feedback()
                if feedback:
                    stagnant += 1
                    if stagnant >= 3:
                        raise AgentError('Модель не продвигается в подборе: три ответа без результата. '
                                         'Подходящая квартира не подтверждена.')
                    self.messages.append({'role': 'user', 'content': feedback})
                    if self.on_progress:
                        self.on_progress({'kind': 'continue', 'step': step, 'message': feedback})
                    continue
                return message['content']
            before = (len(self.tools.seen), self.tools.bank_limit, len(checked))
            for call in tool_calls:
                function = call['function']
                result = self.tools.execute(function['name'], function['arguments'])
                event = {'step': step, 'call_id': call['id'], 'name': function['name'],
                         'arguments': function['arguments'], 'result': deepcopy(result)}
                self.events.append(event)
                if function['name'] == 'check_mortgage' and 'error' not in result:
                    checked.add(result['apartment_id'])
                self.messages.append({'role': 'tool', 'tool_call_id': call['id'],
                                      'content': json.dumps(result, ensure_ascii=False)})
                if self.on_event:
                    self.on_event(event)
            after = (len(self.tools.seen), self.tools.bank_limit, len(checked))
            stagnant = stagnant + 1 if before == after else 0
            if stagnant >= 3:
                raise AgentError('Три шага без новых результатов: подбор остановлен, условие не подтверждено.')
        raise AgentError(f'Достигнут лимит {self.max_steps} шагов. Попробуйте уточнить запрос или /new.')

    def _completion_feedback(self):
        if self.goal is None or matching_apartment(self.events, self.goal):
            return None
        budget = self.tools.bank_limit if self.goal == 'mortgage' else None
        limit = MAX_PRICE if budget is None else min(MAX_PRICE, budget)
        found_ids = {e['result']['apartment']['id'] for e in self.events
                     if e['name'] == 'choose_apartment' and e['result'].get('apartment')}
        if not any(a['price'] <= limit and (a['id'] not in self.tools.seen or a['id'] in found_ids)
                   for a in self.tools.catalog):
            raise AgentError('Подходящих квартир больше нет: доступные варианты каталога исчерпаны.')
        if self.goal == 'price':
            return ('Подбор не завершён: ещё не найдена квартира, прошедшая проверку допустимой цены. '
                    'Нужен подходящий вариант.')
        note = f' Лимит банка — {budget:,} ₽.'.replace(',', ' ') if budget is not None else ''
        return ('Подбор не завершён: ещё нет найденной квартиры, прошедшей проверку цены, '
                'для которой подтверждено полное финансирование банка.' + note +
                ' Нужен подходящий вариант, а не только сообщение об отказе.')

    @staticmethod
    def _parse(response):
        try:
            choice = response['choices'][0]
            raw = choice['message']
            reason = choice.get('finish_reason')
            if reason in ('length', 'content_filter'):
                raise AgentError('Ответ модели обрезан или заблокирован; финал не получен.')
            calls = raw.get('tool_calls') or []
            content = raw.get('content')
            if not isinstance(calls, list):
                raise ValueError
            if calls:
                ids = set()
                for call in calls:
                    if (not isinstance(call['id'], str) or not call['id'] or call['id'] in ids
                            or call.get('type') != 'function'
                            or not isinstance(call['function']['name'], str)
                            or not isinstance(call['function']['arguments'], str)):
                        raise ValueError
                    ids.add(call['id'])
                message = {'role': 'assistant', 'content': content if isinstance(content, str) else None,
                           'tool_calls': calls}
            else:
                if reason != 'stop' or not isinstance(content, str) or not content.strip():
                    raise ValueError
                message = {'role': 'assistant', 'content': content}
            return message, calls
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            raise AgentError('Некорректный или пустой ответ API: нет финала или корректных tool_calls.') from None
