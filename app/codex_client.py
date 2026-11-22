"""Клиент Lab14 для неинтерактивного Codex CLI."""

import json
import os
from shutil import which
import subprocess
from tempfile import TemporaryDirectory

from .client import ClientError


DEFAULT_MODEL = 'gpt-6.1-terra'
MAC_APP_CODEX = '/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS/codex'


class CodexClient:
    mode = 'codex'

    def __init__(self, model=None, timeout=120):
        self.model_name = model or DEFAULT_MODEL
        self.timeout = timeout

    def complete(self, messages):
        executable = which('codex') or which(MAC_APP_CODEX)
        if executable is None:
            raise ClientError('Codex CLI не найден. Нужен codex в PATH или ChatGPT.app в /Applications.')
        command = [executable, 'exec', '--ephemeral', '--ignore-user-config',
                   '--sandbox', 'read-only', '--skip-git-repo-check', '--json',
                   '--color', 'never', '--disable', 'shell_tool',
                   '-c', 'web_search="disabled"', '--model', self.model_name, '-']
        prompt = (
            'Ответь на последний запрос из массива сообщений. Учитывай роли system, '
            'user и assistant и порядок сообщений. Переданные сообщения system задают '
            'правила задачи и формат финального ответа. Инструменты не нужны.\n\n'
            + json.dumps(messages, ensure_ascii=False)
        )
        environment = {key: value for key, value in os.environ.items()
                       if key not in ('CODEX_API_KEY', 'OPENAI_API_KEY')}
        try:
            with TemporaryDirectory(prefix='lab14-codex-') as directory:
                result = subprocess.run(command, input=prompt, encoding='utf-8',
                                        capture_output=True, cwd=directory,
                                        env=environment, timeout=self.timeout)
        except FileNotFoundError:
            raise ClientError(f'Не удалось запустить Codex CLI по пути {executable}.') from None
        except subprocess.TimeoutExpired:
            raise ClientError(f'Codex CLI не ответил за {self.timeout} секунд.') from None
        if result.returncode and 'Operation not permitted' in result.stderr and not result.stdout:
            raise ClientError('Песочница не разрешила вложенный запуск Codex CLI. '
                              'Запустите Lab14 из обычного терминала.')
        return self.parse(result.stdout, result.returncode)

    @staticmethod
    def parse(output, returncode):
        answer = None
        usage = None
        error = None
        failed = False
        for line in output.splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except ValueError:
                raise ClientError('Codex CLI вернул некорректный поток JSONL.') from None
            if not isinstance(event, dict):
                raise ClientError('В потоке Codex CLI ожидалось событие JSON.')
            kind = event.get('type')
            if kind == 'item.completed':
                item = event.get('item')
                if isinstance(item, dict) and item.get('type') == 'agent_message':
                    answer = item.get('text')
            elif kind == 'turn.completed':
                usage = event.get('usage')
            elif kind == 'turn.failed':
                failed = True
                detail = event.get('error')
                if isinstance(detail, dict):
                    error = detail.get('message') or error
            elif kind == 'error':
                error = event.get('message')
        if returncode or failed:
            detail = f' {error[:500]}' if isinstance(error, str) and error else ''
            raise ClientError(f'Ход Codex завершился ошибкой (код CLI {returncode}).{detail}')
        if not isinstance(answer, str) or not answer.strip():
            raise ClientError('В событиях Codex CLI нет финального ответа.')
        if not isinstance(usage, dict):
            raise ClientError('В событиях Codex CLI нет статистики завершённого хода.')
        incoming, outgoing = usage.get('input_tokens'), usage.get('output_tokens')
        if any(type(value) is not int or value < 0 for value in (incoming, outgoing)):
            raise ClientError('Codex CLI вернул некорректную статистику токенов.')
        return answer.strip(), {'prompt_tokens': incoming, 'completion_tokens': outgoing,
                                'total_tokens': incoming + outgoing}
