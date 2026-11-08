"""Интеграционные тесты с локальным HTTP API; ключи и внешний WB не используются."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.client import IncompleteRead
import json
from pathlib import Path
import tempfile
from threading import Thread
import unittest
from urllib.error import HTTPError

import main


@contextmanager
def local_api(response):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield {'LLM_API_URL': f'http://127.0.0.1:{server.server_port}/chat/completions',
               'LLM_API_KEY': 'local-test-only'}, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


MODEL = {'route': 'small', 'expected_model': 'real-model'}
TASK = {'id': 'case', 'prompt': 'Выбери верный вариант.', 'answer': 'B'}


def envelope(**changes):
    return {'model': 'real-model', 'choices': [{'message': {'content': '{"answer":"B"}'},
                                              'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}, **changes}


class ApiTests(unittest.TestCase):
    def test_rejected_medium_is_recorded_without_retry_or_parameter_substitution(self):
        requests = []

        def reject(config, payload):
            requests.append(payload)
            raise HTTPError('http://localhost/chat/completions', 400, 'unsupported effort', {}, None)

        result = main.run_case({}, MODEL, TASK, request_fn=reject)
        self.assertEqual(result['error'], 'API HTTP 400')
        self.assertFalse(result['correct'])
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]['reasoning_effort'], 'medium')
        self.assertEqual(result['request']['reasoning_effort'], 'medium')

    def test_incomplete_http_read_is_recorded_and_next_case_can_run(self):
        def incomplete(*args):
            raise IncompleteRead(b'partial')

        failed = main.run_case({}, MODEL, TASK, request_fn=incomplete)
        following = main.run_case({}, MODEL, TASK, request_fn=lambda *_: envelope())
        self.assertIn('error', failed)
        self.assertFalse(failed['correct'])
        self.assertTrue(following['correct'])

    def test_nonstop_finish_reasons_are_never_correct(self):
        for reason in (None, 'content_filter', 'tool_calls', 'length'):
            with self.subTest(reason=reason):
                response = envelope(choices=[{'message': {'content': '{"answer":"B"}'},
                                             'finish_reason': reason}])
                result = main.run_case({}, MODEL, TASK, request_fn=lambda *_: response)
                self.assertFalse(result['correct'])
                self.assertFalse(result['format_valid'])

    def test_real_http_response_is_scored_and_prompt_does_not_leak_answer_key(self):
        with local_api(envelope()) as (config, requests):
            result = main.run_case(config, MODEL, TASK)
        self.assertTrue(result['correct'])
        self.assertTrue(result['eligible'])
        self.assertEqual(result['usage']['total_tokens'], 15)
        self.assertEqual(len(requests), 1)
        self.assertEqual(set(requests[0]), {'model', 'temperature', 'messages', 'reasoning_effort'})
        self.assertEqual(requests[0]['reasoning_effort'], 'medium')
        self.assertNotIn('expected_answer', requests[0])
        self.assertNotIn('local-test-only', json.dumps(result))

    def test_wrong_model_is_excluded_but_usage_saved(self):
        with local_api(envelope(model='another')) as (config, _):
            result = main.run_case(config, MODEL, TASK)
        self.assertFalse(result['eligible'])
        self.assertFalse(result['correct'])
        self.assertEqual(result['usage']['total_tokens'], 15)
        self.assertNotIn('error', result)

    def test_malformed_envelope_is_not_a_wrong_answer(self):
        with local_api({'model': 'real-model', 'choices': []}) as (config, _):
            result = main.run_case(config, MODEL, TASK)
        self.assertIn('error', result)
        self.assertFalse(result['correct'])
        self.assertTrue(result['response_received'])

    def test_missing_usage_does_not_break_scoring(self):
        with local_api(envelope(usage=None)) as (config, _):
            result = main.run_case(config, MODEL, TASK)
        self.assertTrue(result['correct'])
        self.assertIsNone(main.summarize([result], [MODEL], 1)[0]['total_tokens'])

    def test_error_does_not_reveal_exception_or_credentials(self):
        def fail(*args):
            raise OSError('secret-local-key')

        result = main.run_case({}, MODEL, TASK, request_fn=fail)
        self.assertIn('error', result)
        self.assertNotIn('secret-local-key', json.dumps(result))

    def test_saved_report_is_readable_and_updated_after_each_record(self):
        report = {'started': 'local test', 'completed': False, 'models': [MODEL],
                  'dataset': {'tasks': [TASK]}, 'records': []}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            main.save_report(path, report)
            self.assertEqual(json.loads((path / 'comparison.json').read_text())['records'], [])
            record = main.run_case({}, MODEL, TASK, request_fn=lambda *_: envelope())
            report['records'].append(record)
            main.save_report(path, report)
            self.assertEqual(json.loads((path / 'comparison.json').read_text())['summary'][0]['correct'], 1)
            self.assertIn('1/1', (path / 'comparison.md').read_text())
            self.assertEqual(list(path.glob('*.tmp')), [])


if __name__ == '__main__':
    unittest.main()
