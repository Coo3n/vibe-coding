"""Единый строгий формат ответа API и его отображение в консоли."""

import json


RESPONSE_FORMAT = {
    'type': 'json_schema',
    'json_schema': {
        'name': 'travel_reply',
        'strict': True,
        'schema': {
            'type': 'object',
            'properties': {
                'action': {'type': 'string', 'enum': ['ask', 'answer']},
                'answer': {'type': 'string'},
            },
            'required': ['action', 'answer'],
            'additionalProperties': False,
        },
    },
}


def reply_payload(content):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Повторяющееся поле в JSON ответа')
            result[key] = value
        return result

    try:
        data = json.loads(content, object_pairs_hook=unique_keys)
    except (ValueError, TypeError):
        raise ValueError('Ответ API должен быть JSON-объектом с полями action и answer') from None
    if (not isinstance(data, dict) or set(data) != {'action', 'answer'}
            or data['action'] not in ('ask', 'answer')
            or not isinstance(data['answer'], str) or not data['answer'].strip()):
        raise ValueError('Ответ API не соответствует схеме action/answer')
    return data


def answer_text(content):
    return reply_payload(content)['answer']
