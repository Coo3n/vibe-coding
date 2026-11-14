"""HTTP-клиент OpenAI Chat Completions без SDK."""

from http.client import HTTPException
import json
import os
from pathlib import Path
import socket
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener


class ClientError(RuntimeError):
    pass


# Соответствие маршрута сообщено пользователем; конкретное имя из API приоритетнее.
KNOWN_MODELS = {'coder': 'glm-53-flash'}


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        # Не передавать Authorization на адрес из Location.
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
        raise ClientError('Заполните LLM_API_URL и LLM_API_KEY в .env или окружении')
    if urlparse(settings['LLM_API_URL']).scheme not in ('http', 'https'):
        raise ClientError('LLM_API_URL должен быть HTTP(S) адресом chat/completions')
    settings['LLM_MODEL'] = settings.get('LLM_MODEL') or 'coder'
    return settings


class ChatClient:
    def __init__(self, settings, timeout=120):
        self.settings = settings
        self.timeout = timeout
        self.opener = build_opener(NoRedirects())
        self.model_catalog = {}
        self.actual_model = KNOWN_MODELS.get(settings['LLM_MODEL'])

    @property
    def model_name(self):
        return self.actual_model or 'API не раскрывает название модели'

    def load_model_catalog(self):
        url = self.settings['LLM_API_URL'].rstrip('/').removesuffix('/chat/completions') + '/models'
        try:
            data = self._request(url, timeout=10)
            self.model_catalog = {row['id']: row['real_model'] for row in data['data']
                                  if isinstance(row, dict) and isinstance(row.get('id'), str)
                                  and isinstance(row.get('real_model'), str)}
            self.actual_model = self._resolve_model(self.settings['LLM_MODEL'])
        except (ClientError, KeyError, TypeError):
            # Служебный каталог необязателен; сбой не блокирует сам чат.
            pass

    def _resolve_model(self, identifier):
        seen = set()
        while identifier in self.model_catalog:
            if identifier in seen:
                return KNOWN_MODELS.get(identifier)
            seen.add(identifier)
            identifier = self.model_catalog[identifier]
        if not identifier or identifier == self.settings['LLM_MODEL']:
            return KNOWN_MODELS.get(identifier)
        return identifier

    def complete(self, messages, tools=None):
        payload = {'model': self.settings['LLM_MODEL'], 'messages': messages,
                   'temperature': 0}
        if tools:
            payload.update(tools=tools, tool_choice='auto', parallel_tool_calls=False)
        data = self._request(self.settings['LLM_API_URL'], payload)
        if isinstance(data, dict):
            identifier = data.get('model')
            if isinstance(identifier, str):
                self.actual_model = self._resolve_model(identifier)
        return data

    def _request(self, url, payload=None, timeout=None):
        request = Request(url,
                          data=None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                          headers={'Authorization': 'Bearer ' + self.settings['LLM_API_KEY'],
                                   'Content-Type': 'application/json'})
        try:
            with self.opener.open(request, timeout=self.timeout if timeout is None else timeout) as response:
                return json.loads(response.read().decode('utf-8'))
        except HTTPError as error:
            status = error.code
            error.close()
            raise ClientError(f'API вернул HTTP {status}. Проверьте доступ и модель.') from None
        except URLError as error:
            if isinstance(error.reason, socket.gaierror):
                raise ClientError('Адрес API не найден в DNS. Проверьте LLM_API_URL и доступ к корпоративной сети/VPN.') from None
            raise ClientError('Не удалось связаться с API: сеть или таймаут.') from None
        except (OSError, TimeoutError, HTTPException):
            raise ClientError('Не удалось связаться с API: сеть или таймаут.') from None
        except (ValueError, UnicodeError):
            raise ClientError('API вернул некорректный JSON.') from None
