"""Агент владеет историей, протоколом tool calling и циклом до финального ответа."""

from copy import deepcopy
import json
import time

from .client import ClientError
from .tools import TOOL_SCHEMAS
from .history import HistoryError
from .response import answer_text, reply_payload


SYSTEM_PROMPT = '''Ты — агент подбора путёвки «Компас». Общайся по-русски.
Данные и цены учебные. Выясни: город вылета, направление (можно любое),
точные даты с годом и гибкость, ночи, взрослых и возраст детей, общий бюджет,
формат отдыха, отель и питание. Визовые ограничения уточняй при необходимости.
Задавай ровно один следующий вопрос за раз. Не спрашивай известное повторно.
Учитывай несколько ответов в одной реплике и исправления прежних пожеланий.
Необязательные предпочтения можно пропустить по желанию пользователя.
Не запрашивай паспортные данные. Не утверждай актуальность виз и цен.
Когда условия собраны, вызови search_tours; не придумывай предложения, ID и цены.
При пустой выдаче сообщи об этом и спроси, какое условие можно изменить.
Не ослабляй условия сам и не подменяй пожелания пользователя учебными примерами.
Сравнивай найденные варианты по данным результата инструмента.
Источником фактов о путёвках являются только сообщения role=tool.
Не добавляй варианты, отели, рейтинги и отзывы, которых нет в результате.
Если прежний ответ assistant противоречит инструменту, исправь прежний ответ.
Если пользователь просит вспомнить прошлую выдачу без поиска, используй только
историю: не вызывай инструмент заново, честно скажи, если сведений нет.
В новой истории ты не знаешь предыдущие диалоги. Не выдумывай воспоминания.
На каждом запросе обязательно вызови ровно один инструмент через tool_calls.
Для вопроса или ответа пользователю вызывай respond_to_user со строгим JSON:
{"action": "ask|answer", "answer": "текст ответа или один вопрос"}.
Не добавляй другие поля. action=ask — один уточняющий вопрос, action=answer — ответ.
Когда условия собраны и пользователь просит поиск — вызывай search_tours,
а не respond_to_user с обещанием поиска. После результата search_tours вызывай
respond_to_user с найденными вариантами либо вопросом об изменении условий.
Для неизвестных необязательных фильтров search_tours передавай null.
При воспоминании старой выдачи без нового поиска вызывай только respond_to_user.
В каталоге 606 учебных путёвок: июнь–сентябрь 2027, Москва, Санкт-Петербург,
Казань, Екатеринбург, Новосибирск; Турция, Египет, Россия, Таиланд, ОАЭ,
Шри-Ланка. Не ограничивай пользователя контрольными двумя датами.'''


class AgentError(RuntimeError):
    pass


LEGACY_OPENING_QUESTION = 'Из какого города планируете вылетать?'
OPENING_QUESTION = ('Откуда летим?\n1. Москва\n2. Санкт-Петербург\n3. Казань\n'
                    '4. Екатеринбург\n5. Новосибирск\nМожно номер или свой город.')


def opening_messages(question=OPENING_QUESTION):
    """Локальный стартовый вопрос, не выдаваемый за ответ API."""
    return [
        {'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': 'local-opening-question', 'type': 'function', 'function': {
                'name': 'respond_to_user',
                'arguments': json.dumps({'action': 'ask', 'answer': question}, ensure_ascii=False)}}]},
        {'role': 'tool', 'tool_call_id': 'local-opening-question',
         'content': '{"status":"accepted"}'},
    ]


class Agent:
    def __init__(self, client, tools, max_steps=5, on_event=None, on_progress=None, history=None):
        self.client = client
        self.tools = tools
        if type(max_steps) is not int or not 1 <= max_steps <= 5:
            raise ValueError('Лимит обращений к LLM должен быть от 1 до 5.')
        self.max_steps = max_steps
        self.on_event = on_event
        self.on_progress = on_progress
        self.history = history
        loaded = history.load() if history else None
        self.restored = loaded is not None
        self.messages = loaded if loaded is not None else [{'role': 'system', 'content': SYSTEM_PROMPT}]
        self.events = []
        self.trace = []

    def start_dialog(self):
        if len(self.messages) != 1:
            return False
        messages = self.messages + opening_messages()
        if self.history:
            self.history.save(messages)
        self.messages = messages
        return True

    def ask(self, text):
        if not isinstance(text, str) or not text.strip():
            raise AgentError('Введите непустой запрос.')
        try:
            text.encode('utf-8')
        except UnicodeError:
            raise AgentError('Ввод содержит повреждённый символ: сообщение не отправлено и не сохранено. '
                             'Введите его заново (например, Санкт-Петербург).') from None
        if '\ufffd' in text:
            raise AgentError('Во вводе найден символ замены �. Сообщение не отправлено и не сохранено. '
                             'Введите текст заново.')
        self.events = []
        self.trace = []
        start = len(self.messages)
        self.messages.append({'role': 'user', 'content': text})
        try:
            answer = self._run()
            if self.history:
                self.history.save(self.messages)
            return answer
        except (AgentError, ClientError, HistoryError, KeyboardInterrupt):
            # Не оставляем незавершённый tool-протокол в следующем HTTP-запросе.
            del self.messages[start:]
            raise

    def _run(self):
        accepted = {m['tool_call_id'] for m in self.messages if m['role'] == 'tool'
                    and json.loads(m['content']).get('status') == 'accepted'}
        questions = sum(1 for m in self.messages if m['role'] == 'assistant'
                        for call in m.get('tool_calls', [])
                        if call['id'] in accepted and call['function']['name'] == 'respond_to_user'
                        and self._is_question(call['function']['arguments']))
        had_search = any(m['role'] == 'tool' and 'tours' in json.loads(m['content'])
                         for m in self.messages)
        for step in range(1, self.max_steps + 1):
            must_search = questions >= 5 and not had_search
            control = (
                f'Управление опросом: уже задано {questions} вопросов из максимум 5 за весь подбор. '
                'Первый — город вылета. Остальные объединяй: 2) направление и формат отдыха; '
                '3) даты с годом, гибкость и ночи; 4) взрослые, дети и возраст; '
                '5) общий бюджет, отель и питание. Пропускай уже известные сведения. '
                'Когда ответ на пятый вопрос получен, сразу search_tours; неизвестные поля null. '
                'После лимита никаких уточняющих вопросов: action=answer, без вопроса в тексте. '
                'Неизвестные условия и состав найденных групп честно укажи, не выдумывай их. '
                'Если выдача пуста — сообщи это, не ослабляй заданные условия.'
                '\nСтиль вопросов: короткий заголовок и 3–5 пронумерованных вариантов, '
                'каждый с новой строки. Без вступлений, повторения известных условий и длинных объяснений. '
                'В конце: «Можно номер или свой вариант». Цифра пользователя относится ТОЛЬКО '
                'к вариантам последнего показанного вопроса. Не путай её с бюджетом или количеством людей. '
                'Для объединённых параметров предлагай готовые понятные наборы, '
                'каждый выбранный набор явно содержит все свои условия. '
                'Примеры: направление — «Турция, пляжный», «Египет, пляжный», «Россия, экскурсии»; '
                'даты — «10 июня 2027, 7 ночей, строго», «12 июня 2027, 7 ночей, строго», '
                '«Свои даты и ночи»; состав — «1 взрослый», «2 взрослых, без детей», '
                '«2 взрослых + ребёнок (напишите возраст)», «Свой состав»; '
                'бюджет и отель — «до150000 ₽, 3★+, завтраки», '
                '«до200000 ₽, 4★+, всё включено», «до300000 ₽, 5★, всё включено», '
                '«Свой бюджет и отель». Это примеры пожеланий, не гарантия наличия тура. '
                'Не уточняй отдельно необязательные детали, если их можно оставить null.'
            )
            request = {'messages': deepcopy(self.messages) + [{'role': 'system', 'content': control}],
                       'tools': deepcopy(TOOL_SCHEMAS[:1] if must_search else TOOL_SCHEMAS),
                       'tool_choice': 'required'}
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
            for call in tool_calls:
                function = call['function']
                if must_search and function['name'] != 'search_tours':
                    result = {'error': 'Лимит вопросов исчерпан. Выполни search_tours, неизвестные поля null.'}
                elif questions >= 5 and function['name'] == 'respond_to_user' and self._is_question(function['arguments']):
                    result = {'error': 'Шестой уточняющий вопрос запрещён. Верни action=answer с результатом поиска.'}
                else:
                    result = self.tools.execute(function['name'], function['arguments'])
                if function['name'] == 'search_tours' and 'tours' in result:
                    had_search = True
                event = {'step': step, 'call_id': call['id'], 'name': function['name'],
                         'arguments': function['arguments'], 'result': deepcopy(result)}
                self.events.append(event)
                self.messages.append({'role': 'tool', 'tool_call_id': call['id'],
                                      'content': json.dumps(result, ensure_ascii=False)})
                if self.on_event:
                    self.on_event(event)
                if function['name'] == 'respond_to_user' and 'error' not in result:
                    return answer_text(function['arguments'])
        raise AgentError(f'Достигнут лимит {self.max_steps} шагов. Попробуйте уточнить запрос или /new.')

    @staticmethod
    def _is_question(arguments):
        try:
            return reply_payload(arguments)['action'] == 'ask'
        except ValueError:
            return False

    @staticmethod
    def _parse(response):
        try:
            choice = response['choices'][0]
            raw = choice['message']
            reason = choice.get('finish_reason')
            if reason in ('length', 'content_filter'):
                raise AgentError('Ответ модели обрезан или заблокирован; финал не получен.')
            calls = raw.get('tool_calls', [])
            if calls is None:
                calls = []
            content = raw.get('content')
            if not isinstance(calls, list) or len(calls) != 1:
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
                if content is not None and not isinstance(content, str):
                    raise ValueError
            message = deepcopy(raw)
            message['role'] = 'assistant'
            return message, calls
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            raise AgentError('Некорректный ответ API: обязателен ровно один tool_call '
                             '(search_tours или respond_to_user). Ход не сохранён.') from None
