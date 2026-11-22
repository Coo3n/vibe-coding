"""OpenAI-compatible API-клиент с проверкой ответа и учётом usage."""

import os
from pathlib import Path
import time

from dotenv import dotenv_values


class ClientError(RuntimeError):
    pass


def load_settings(path):
    values = {**dotenv_values(Path(path)), **os.environ}
    settings = {key: values.get(key) or values.get(alias) for key, alias in (
        ('LLM_API_KEY', 'API_TOKEN'), ('LLM_API_URL', 'API_URL'), ('LLM_MODEL', 'MODEL'))}
    return settings


class ChatClient:
    def __init__(self, settings, timeout=300):
        self.settings = settings
        self.model_name = settings.get('LLM_MODEL') or ''
        self.timeout = timeout

    def complete(self, messages):
        if not self.settings.get('LLM_API_KEY') or not self.model_name:
            raise ClientError('Заполните LLM_API_KEY и LLM_MODEL в .env или окружении.')
        url = (self.settings.get('LLM_API_URL') or '').rstrip('/')
        if not url.startswith(('http://', 'https://')):
            raise ClientError('Укажите HTTP(S) адрес API через LLM_API_URL или API_URL.')
        import requests

        if not url.endswith('/chat/completions'):
            url += '/chat/completions' if url.endswith('/v1') else '/v1/chat/completions'
        for attempt in range(2):
            try:
                response = requests.post(url, json={'model': self.model_name, 'messages': messages},
                                         headers={'Authorization': 'Bearer ' + self.settings['LLM_API_KEY']},
                                         timeout=self.timeout)
                response.raise_for_status()
                return self.parse(response.json())
            except (requests.Timeout, requests.ConnectionError):
                if attempt == 0:
                    time.sleep(2)
                    continue
                raise ClientError('Не удалось связаться с API: проверьте сеть, DNS/VPN и таймаут.') from None
            except requests.HTTPError as error:
                status = error.response.status_code if error.response is not None else '?'
                if status == 429 and attempt == 0:
                    time.sleep(5)
                    continue
                raise ClientError(f'API вернул HTTP {status}.') from None
            except requests.RequestException:
                raise ClientError('Ошибка запроса к API.') from None
            except ValueError:
                raise ClientError('API вернул некорректный JSON.') from None

    @staticmethod
    def parse(data):
        try:
            choice = data['choices'][0]
            text = choice['message']['content']
            usage = data['usage']
            if choice.get('finish_reason') in ('length', 'content_filter'):
                raise ClientError('Ответ модели обрезан или заблокирован.')
            if not isinstance(text, str) or not text.strip():
                raise ClientError('API вернул пустой или некорректный текст ответа.')
            if not isinstance(usage, dict) or type(usage.get('total_tokens')) is not int or usage['total_tokens'] < 0:
                raise ClientError('API не вернул total_tokens: честное сравнение расхода невозможно.')
            return text.strip(), usage
        except (KeyError, IndexError, TypeError, AttributeError):
            raise ClientError('Некорректный ответ API: нужны choices[0].message.content и usage.') from None
