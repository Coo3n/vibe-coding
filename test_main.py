import unittest

from main import summarize, render_report, render_terminal_summary, MODELS as BENCHMARK_MODELS


MODELS = [{'route': 'small', 'expected_model': 'actual'}]


def record(**changes):
    row = {'route': 'small', 'response_model': 'actual', 'seconds': 2,
           'format_valid': True, 'correct': True, 'eligible': True,
           'usage': {'prompt_tokens': 10, 'completion_tokens': 20,
                     'reasoning_tokens': 15, 'total_tokens': 30}}
    return {**row, **changes}


class SummaryTests(unittest.TestCase):
    def test_api_rejections_are_not_reported_as_solved_accuracy_comparison(self):
        rows = [record(route='llm-small', error='API HTTP 400', eligible=False,
                       correct=False, usage=None, response_received=False),
                record(route='llm-medium'), record(route='coder')]
        rows = [{**row, 'task_id': 'task'} for row in rows]
        report = {'started': 'test', 'completed': True, 'models': BENCHMARK_MODELS,
                  'dataset': {'tasks': [{'id': 'task', 'prompt': 'Вопрос', 'answer': 'A'}]},
                  'records': rows, 'summary': summarize(rows, BENCHMARK_MODELS, 1),
                  'method': {'reasoning_setting': 'high'}}
        terminal = render_terminal_summary(report)
        qwen_line = next(line for line in terminal.splitlines() if 'Qwen' in line)
        self.assertIn('нет ответов', qwen_line)
        self.assertIn('Сбои API', terminal)
        rendered = render_report(report)
        self.assertIn('сравнение точности с малой моделью недоступно', rendered)
        self.assertNotIn('разница точности', rendered)

    def test_report_uses_its_own_reasoning_setting(self):
        report = {'started': 'test', 'completed': False, 'models': MODELS,
                  'dataset': {'tasks': [{'id': 'task', 'prompt': 'Вопрос', 'answer': 'A'}]},
                  'records': [], 'summary': summarize([], MODELS, 1),
                  'method': {'reasoning_setting': 'medium'}, 'video': '../demo.mp4'}
        rendered = render_report(report)
        self.assertIn('reasoning_effort=medium', rendered)
        self.assertIn('[Видео запуска](../demo.mp4)', rendered)
        report['method']['reasoning_setting'] = None
        self.assertNotIn('reasoning_effort=medium', render_report(report))

    def test_summary_includes_public_price_estimate_for_known_model(self):
        rows = [record(route='llm-small', usage={'prompt_tokens': 898, 'completion_tokens': 72,
                                               'total_tokens': 970})]
        models = [{'route': 'llm-small', 'expected_model': 'actual'}]
        summary = summarize(rows, models, 6)[0]
        self.assertAlmostEqual(summary['cost'], 0.000665)
        self.assertEqual(summary['cost_basis'], 'public_price_estimate')

    def test_counts_accuracy_latency_and_usage_without_double_counting_reasoning(self):
        summary = summarize([record(), record(seconds=4, correct=False)], MODELS, 6)[0]
        self.assertEqual(summary['correct'], 1)
        self.assertAlmostEqual(summary['accuracy'], 1 / 6)
        self.assertEqual(summary['mean_seconds'], 3)
        self.assertEqual(summary['median_seconds'], 3)
        self.assertEqual(summary['total_tokens'], 60)
        self.assertEqual(summary['reasoning_tokens'], 30)
        self.assertEqual(summary['usage_missing'], 0)
        self.assertIsNone(summary['cost'])

    def test_missing_usage_is_unknown_not_zero(self):
        summary = summarize([record(usage=None)], MODELS, 6)[0]
        self.assertIsNone(summary['total_tokens'])
        self.assertEqual(summary['usage_missing'], 1)

    def test_excludes_wrong_model_but_keeps_its_resource_usage(self):
        summary = summarize([record(eligible=False)], MODELS, 6)[0]
        self.assertEqual(summary['correct'], 0)
        self.assertEqual(summary['excluded'], 1)
        self.assertEqual(summary['total_tokens'], 30)

    def test_api_failure_and_format_failure_are_separate(self):
        rows = [record(error='HTTP 500', eligible=False, usage=None),
                record(format_valid=False, correct=False)]
        summary = summarize(rows, MODELS, 6)[0]
        self.assertEqual(summary['api_errors'], 1)
        self.assertEqual(summary['format_errors'], 1)
        self.assertEqual(summary['correct'], 0)
        self.assertIsNone(summary['total_tokens'])
        self.assertEqual(summary['known_total_tokens'], 30)

    def test_empty_run_does_not_divide_by_zero(self):
        summary = summarize([], MODELS, 6)[0]
        self.assertEqual(summary['correct'], 0)
        self.assertIsNone(summary['mean_seconds'])
        self.assertIsNone(summary['total_tokens'])


if __name__ == '__main__':
    unittest.main()
