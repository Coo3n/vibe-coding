import unittest

from pricing import estimate_cost


class PricingTests(unittest.TestCase):
    def test_qwen_uses_input_and_output_rates(self):
        self.assertAlmostEqual(estimate_cost('llm-small', 898, 72), 0.000665)

    def test_deepseek_uses_historical_offpeak_scenario(self):
        self.assertAlmostEqual(estimate_cost('llm-medium', 939, 42), 0.0002343)

    def test_glm_output_already_includes_reasoning(self):
        self.assertAlmostEqual(estimate_cost('coder', 926, 1224), 0.0007509)

    def test_missing_or_invalid_counts_and_unknown_models_are_unknown(self):
        for route, prompt, output in [('other', 1, 1), ('coder', None, 1),
                                      ('coder', 1, None), ('coder', -1, 1),
                                      ('coder', True, 1)]:
            with self.subTest(route=route, prompt=prompt, output=output):
                self.assertIsNone(estimate_cost(route, prompt, output))

    def test_zero_usage_has_zero_estimated_cost(self):
        self.assertEqual(estimate_cost('coder', 0, 0), 0)


if __name__ == '__main__':
    unittest.main()
