"""Полный запуск CLI с локальным сервером и тремя параллельными очередями."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import Barrier, Lock, Thread
import unittest

from main import MODELS


class RunnerTests(unittest.TestCase):
    def test_cli_runs_every_model_and_task_with_medium_and_preserves_results(self):
        self.run_cli('medium', explicit=False)

    def test_cli_runs_high_without_changing_medium_default(self):
        self.run_cli('high')

    def run_cli(self, effort, explicit=True):
        root = Path(__file__).resolve().parent
        dataset = json.loads((root / 'tasks.json').read_text())
        tasks = dataset['tasks']
        models = {model['route']: model['expected_model'] for model in MODELS}
        received = []
        first_requests = set()
        lock = Lock()
        barrier = Barrier(3, timeout=10)

        class Handler(BaseHTTPRequestHandler):
            def send_json(self, payload):
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode())

            def do_GET(self):
                self.send_json({'data': [{'id': route, 'real_model': model}
                                         for route, model in models.items()]})

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                route = payload['model']
                with lock:
                    received.append(payload)
                    first = route not in first_requests
                    first_requests.add(route)
                if first:
                    barrier.wait()
                prompt = payload['messages'][0]['content']
                task = next(task for task in tasks if prompt.startswith(task['prompt'] + '\n\n'))
                self.send_json({'model': models[route], 'choices': [{
                    'message': {'content': json.dumps({'answer': task['answer']})},
                    'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 50, 'completion_tokens': 30,
                              'reasoning_tokens': 20, 'total_tokens': 80}})

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                config = directory / '.env'
                config.write_text(f'LLM_API_URL=http://127.0.0.1:{server.server_port}/chat/completions\n'
                                  'LLM_API_KEY=offline-test-only\n')
                output = directory / 'run'
                command = [sys.executable, str(root / 'main.py'), '--env', str(config),
                           '--output-dir', str(output)]
                if explicit:
                    command += ['--effort', effort]
                result = subprocess.run(command, capture_output=True, text=True,
                                        timeout=40, cwd=root)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                progress = [line for line in result.stdout.splitlines()
                            if ' · ' in line and 'токены:' in line]
                self.assertEqual(len(progress), len(tasks) * 3)
                self.assertTrue(all('стоимость: $' in line and ' с · ' in line for line in progress))
                self.assertIn('ИТОГОВАЯ ТАБЛИЦА', result.stdout)
                self.assertIn('Оценка, USD', result.stdout)
                self.assertIn('$0.00115000', result.stdout)
                self.assertNotIn('сценарная оценка по публичным тарифам', result.stdout)
                self.assertNotIn('спецификациями', result.stdout)
                for name in ('Qwen 3.8 27B', 'DeepSeek V4 Flash', 'GLM 5.3 Flash'):
                    self.assertIn(name, result.stdout)
                self.assertNotIn('llm-small', result.stdout)
                self.assertNotIn('llm-medium', result.stdout)
                self.assertNotIn('glm-53-flash', result.stdout)
                report = json.loads((output / 'comparison.json').read_text())
                self.assertTrue(report['completed'])
                self.assertEqual(report['method']['reasoning_setting'], effort)
                self.assertEqual(len(report['records']), len(tasks) * 3)
                self.assertEqual(len(received), len(tasks) * 3)
                self.assertTrue(all(p['reasoning_effort'] == effort for p in received))
                self.assertTrue(all(s['correct'] == len(tasks) for s in report['summary']))
                self.assertTrue(all(s['total_tokens'] == 80 * len(tasks) for s in report['summary']))
                self.assertNotIn('offline-test-only', (output / 'comparison.json').read_text())
                self.assertIn(f'reasoning_effort={effort}', (output / 'comparison.md').read_text())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
