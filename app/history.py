"""Версионированная полная история; один процесс, атомарная замена JSON."""

import json
import os
from pathlib import Path
import tempfile

from .response import reply_payload


class HistoryError(RuntimeError):
    pass


def validate_tool_result(result):
    """Проверить типы полей, которые использует повторный показ выдачи."""
    if not isinstance(result, dict):
        raise ValueError
    if 'tours' not in result:
        return
    if not isinstance(result['tours'], list):
        raise ValueError
    for tour in result['tours']:
        if not isinstance(tour, dict):
            raise ValueError
        for key in ('id', 'hotel', 'departure_city', 'country', 'resort', 'departure_date', 'meal'):
            if not isinstance(tour.get(key), str):
                raise ValueError
        for key in ('nights', 'adults', 'children', 'child_age_min', 'child_age_max', 'price_total', 'stars'):
            if type(tour.get(key)) is not int:
                raise ValueError
        if not isinstance(tour.get('features'), list) or any(not isinstance(f, str) for f in tour['features']):
            raise ValueError


def validate_messages(messages):
    # Поздний импорт не создаёт цикла при загрузке Agent.
    from .agent import SYSTEM_PROMPT, opening_messages, LEGACY_OPENING_QUESTION

    try:
        if not isinstance(messages, list) or not messages:
            raise ValueError
        if messages[0] != {'role': 'system', 'content': SYSTEM_PROMPT}:
            raise ValueError
        expected = 'user'
        pending = None
        used = set()
        remaining = messages[1:]
        if remaining[:2] in (opening_messages(), opening_messages(LEGACY_OPENING_QUESTION)):
            remaining = remaining[2:]
            used.add('local-opening-question')
        for message in remaining:
            if not isinstance(message, dict):
                raise ValueError
            role = message['role']
            content = message.get('content')
            if pending is not None:
                identifier, function = pending
                if role != 'tool' or message.get('tool_call_id') != identifier:
                    raise ValueError
                if not isinstance(content, str):
                    raise ValueError
                result = json.loads(content)
                validate_tool_result(result)
                accepted_reply = False
                if 'error' in result:
                    if not isinstance(result['error'], str):
                        raise ValueError
                elif function['name'] == 'respond_to_user':
                    if result != {'status': 'accepted'}:
                        raise ValueError
                    reply_payload(function['arguments'])
                    accepted_reply = True
                elif function['name'] == 'search_tours':
                    if 'tours' not in result:
                        raise ValueError
                else:
                    raise ValueError
                pending = None
                expected = 'user' if accepted_reply else 'assistant'
                continue
            if role != expected:
                raise ValueError
            if role == 'user':
                if not isinstance(content, str) or not content.strip() or 'tool_calls' in message:
                    raise ValueError
                expected = 'assistant'
            else:
                calls = message.get('tool_calls')
                if not isinstance(calls, list) or len(calls) != 1:
                    raise ValueError
                if content is not None and not isinstance(content, str):
                    raise ValueError
                call = calls[0]
                identifier = call['id']
                function = call['function']
                if (not isinstance(identifier, str) or not identifier or identifier in used
                        or call['type'] != 'function'
                        or not isinstance(function['name'], str) or not function['name']
                        or not isinstance(function['arguments'], str)):
                    raise ValueError
                # Некорректные JSON-аргументы допустимы: инструмент возвращает ошибку модели.
                used.add(identifier)
                pending = (identifier, function)
        if pending is not None or expected != 'user':
            raise ValueError
    except (KeyError, TypeError, ValueError, AttributeError):
        raise HistoryError('История повреждена или несовместима: проверьте версию, инструкцию и tool-протокол. '
                           'Файл не изменён; сохраните его копию перед новым диалогом.') from None


class HistoryStore:
    def __init__(self, path):
        self.path = Path(path)

    def load(self):
        try:
            with self.path.open(encoding='utf-8') as stream:
                data = json.load(stream)
        except FileNotFoundError:
            return None
        except (OSError, ValueError, UnicodeError):
            raise HistoryError('Не удалось прочитать историю. Файл не изменён.') from None
        if not isinstance(data, dict) or type(data.get('version')) is not int or data['version'] != 1:
            raise HistoryError('Неподдерживаемая версия истории. Файл не изменён.')
        messages = data.get('messages')
        validate_messages(messages)
        return messages

    def save(self, messages):
        validate_messages(messages)
        temporary = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.path.parent,
                                             prefix='.session-', suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                json.dump({'version': 1, 'messages': messages}, stream, ensure_ascii=False, indent=2,
                          allow_nan=False)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except (OSError, ValueError, TypeError):
            raise HistoryError('Не удалось сохранить ход. Он отменён; предыдущая история не заменена.') from None
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
