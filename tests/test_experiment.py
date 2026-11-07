import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import main


class ExperimentTests(unittest.TestCase):
    def test_same_cold_engineer_career_prompt_across_modes(self):
        payloads = [main.build_payload(mode, 128) for mode in main.MODES]
        prompts = [payload["messages"][-1]["content"] for payload in payloads]
        self.assertEqual(len(set(prompts)), 1)
        self.assertGreater(len(prompts[0]), 500)
        self.assertIn("инженер", prompts[0])
        self.assertIn("ИИ", prompts[0])
        self.assertIn("замен", prompts[0])
        self.assertEqual(payloads[3]["stop"], ["."])

    def test_each_mode_changes_only_its_own_control(self):
        baseline = main.build_payload("baseline", 128)
        formatted = main.build_payload("format", 128)
        limited = main.build_payload("max_tokens", 128)
        stopped = main.build_payload("stop", 128)
        structured = main.build_payload("schema", 128)
        self.assertEqual(set(baseline), {"model", "messages"})
        self.assertEqual(len(baseline["messages"]), 1)
        self.assertEqual(set(formatted), set(baseline))
        self.assertEqual(formatted["messages"][0], {"role": "system", "content": main.FORMAT})
        self.assertEqual(set(limited) - set(baseline), {"max_tokens"})
        self.assertEqual(limited["messages"], baseline["messages"])
        self.assertEqual(limited["max_tokens"], 128)
        self.assertEqual(set(stopped) - set(baseline), {"stop"})
        self.assertEqual(stopped["messages"], baseline["messages"])
        self.assertEqual(set(structured) - set(baseline), {"response_format"})
        self.assertEqual(structured["messages"], baseline["messages"])
        self.assertTrue(structured["response_format"]["json_schema"]["strict"])

    def test_pretty_output_compares_validity_and_shows_failure_reason(self):
        full_answer = "начало\n" + "полный текст ответа " * 30 + "\nКОНЕЦ_ОТВЕТА"
        report = {
            "records": [
                {"mode": "baseline", "run": 1, "content": full_answer, "json_valid": False,
                 "schema_valid": False, "finish_reason": "stop", "validation_error": "JSON: invalid"},
                {"mode": "schema", "run": 1, "content": '{"tips": []}', "json_valid": True,
                 "schema_valid": False, "finish_reason": "length", "validation_error": "мало элементов"},
            ],
            "summary": main.summarize([
                {"mode": "baseline", "run": 1, "content": "Обычный текст", "json_valid": False,
                 "schema_valid": False, "finish_reason": "stop", "refusal": None},
                {"mode": "schema", "run": 1, "content": '{"tips": []}', "json_valid": True,
                 "schema_valid": False, "finish_reason": "length", "refusal": None},
            ]),
            "completed": True,
        }
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main.print_report(report)
        text = output.getvalue()
        self.assertIn("Без ограничений", text)
        self.assertIn("Строгая схема", text)
        self.assertIn("0/1", text)
        self.assertIn("1/1", text)
        self.assertIn("JSON: invalid", text)
        self.assertIn("finish: length", text)
        for line in full_answer.splitlines():
            self.assertIn(f"  │ {line}\n", text)
        self.assertNotIn("{'mode':", text)
        self.assertIn("01 · БЕЗ ОГРАНИЧЕНИЙ", text)
        self.assertIn("05 · СТРОГАЯ СХЕМА", text)
        self.assertIn("ПРОГОН 1", text)
        self.assertIn("ПРОВЕРКА ОТВЕТОВ", text)
        self.assertNotIn("  ИТОГ", text)

    def test_small_limit_applies_only_to_limited_modes(self):
        self.assertLess(main.MAX_TOKENS, 256)
        for mode in ("baseline", "format", "stop", "schema"):
            self.assertNotIn("max_tokens", main.build_payload(mode, main.MAX_TOKENS))
        self.assertEqual(main.build_payload("max_tokens", main.MAX_TOKENS)["max_tokens"], main.MAX_TOKENS)

    def test_network_failure_is_not_displayed_as_failed_json(self):
        records = [{"mode": "baseline", "run": 1, "error": {"kind": "network"}}]
        report = {"records": records, "summary": main.summarize(records), "completed": False}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main.print_report(report)
        text = output.getvalue()
        self.assertIn("⚠ API: network", text)
        self.assertNotIn("0/0", text)
        self.assertNotIn("  ИТОГ", text)

    def test_saved_report_keeps_counts_and_answers_without_text_conclusions(self):
        report = {"records": [{"mode": "baseline", "run": 1, "content": "ответ",
                               "json_valid": False, "schema_valid": False,
                               "finish_reason": "stop", "refusal": None}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            main.save_report(path, report)
            saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(saved["records"][0]["content"], "ответ")
        self.assertEqual(saved["summary"]["baseline"]["attempts"], 1)
        self.assertNotIn("conclusions", saved)


if __name__ == "__main__":
    unittest.main()
