"""OpenAI-compatible HTTP client; based on task-6, strict JSON only."""

from http.client import HTTPException
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .tokens import TokenTracker


class ClientError(RuntimeError):
    pass


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        return None


def load_settings(path):
    settings = {}
    path = Path(path)
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                key, value = line.split('=', 1)
                settings[key.strip()] = value.strip().strip('\"\'')
    for key in ('LLM_API_URL', 'LLM_API_KEY', 'LLM_MODEL'):
        if key in os.environ:
            settings[key] = os.environ[key]
    if not all(settings.get(key) for key in ('LLM_API_URL', 'LLM_API_KEY')):
        raise ClientError('Заполните LLM_API_URL и LLM_API_KEY в .env или окружении.')
    if urlparse(settings['LLM_API_URL']).scheme not in ('http', 'https'):
        raise ClientError('LLM_API_URL должен быть HTTP(S) адресом chat/completions.')
    settings['LLM_MODEL'] = settings.get('LLM_MODEL') or 'coder-mini'
    return settings


class ChatClient:
    def __init__(self, settings, timeout=120):
        self.settings = settings
        self.timeout = timeout
        self.model_name = settings['LLM_MODEL']
        self.opener = build_opener(NoRedirects())
        self.token_tracker = TokenTracker()
        self.phase = 'chat'
        self.on_trace = None

    def complete(self, messages, schema, name, tools=None):
        payload = {'model': self.settings['LLM_MODEL'], 'messages': messages,
                   'temperature': 0,
                   'response_format': {'type': 'json_schema', 'json_schema': {
                       'name': name, 'strict': True, 'schema': schema}}}
        if tools:
            payload.update(tools=tools, tool_choice='required', parallel_tool_calls=False)
        request = Request(self.settings['LLM_API_URL'],
                          data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8'),
                          headers={'Authorization': 'Bearer ' + self.settings['LLM_API_KEY'],
                                   'Content-Type': 'application/json'})
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode('utf-8'))
        except HTTPError as error:
            status = error.code
            error.close()
            detail = (' API должен поддерживать response_format=json_schema и strict=true; '
                      'текстовый режим не используется.' if status in (400, 422) else '')
            raise ClientError(f'API вернул HTTP {status}.{detail}') from None
        except (URLError, OSError, TimeoutError, HTTPException):
            raise ClientError('Не удалось связаться с API: проверьте сеть/VPN и таймаут.') from None
        except (ValueError, UnicodeError):
            raise ClientError('API вернул некорректный JSON.') from None
        if isinstance(data, dict) and isinstance(data.get('model'), str):
            self.model_name = data['model']
        return data
