"""Offline tests for strict answer grading and the frozen task corpus."""

import ast
import json
import itertools
from functools import lru_cache
from pathlib import Path
import re
import subprocess
import unittest

from grading import score_answer


class ScoreAnswerTests(unittest.TestCase):
    def assert_invalid(self, content, json_valid, finish_reason="stop"):
        result = score_answer(content, "A", finish_reason)
        self.assertEqual(result["json_valid"], json_valid)
        self.assertIs(result["format_valid"], False)
        self.assertIs(result["correct"], False)
        self.assertIsInstance(result["validation_error"], str)
        self.assertTrue(result["validation_error"])

    def test_correct_strict_object(self):
        for answer in "ABCD":
            with self.subTest(answer=answer):
                self.assertEqual(score_answer(json.dumps({"answer": answer}), answer), {
                    "json_valid": True,
                    "format_valid": True,
                    "correct": True,
                    "validation_error": None,
                })

    def test_wrong_answer_is_not_a_format_error(self):
        self.assertEqual(score_answer('{"answer":"B"}', "A"), {
            "json_valid": True,
            "format_valid": True,
            "correct": False,
            "validation_error": None,
        })

    def test_surrounding_json_whitespace_is_allowed(self):
        self.assertTrue(score_answer(' \n {"answer": "A"}\t', "A")["correct"])

    def test_empty_and_malformed_content(self):
        for content in ("", "  \n", '{"answer":', '{"answer":"A",}', "A"):
            with self.subTest(content=content):
                self.assert_invalid(content, False)

    def test_fences_prose_and_multiple_objects_are_rejected(self):
        for content in ('```json\n{"answer":"A"}\n```',
                        'Ответ: {"answer":"A"}',
                        '{"answer":"A"} {"answer":"A"}'):
            with self.subTest(content=content):
                self.assert_invalid(content, False)

    def test_duplicate_keys_are_rejected_even_when_identical(self):
        for content in ('{"answer":"B","answer":"A"}',
                        '{"answer":"A","answer":"A"}',
                        '{"answer":"A","answ\\u0065r":"A"}'):
            with self.subTest(content=content):
                self.assert_invalid(content, False)

    def test_nonstandard_json_constants_are_rejected(self):
        for token in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(token=token):
                self.assert_invalid('{"answer":' + token + '}', False)

    def test_extra_or_missing_fields_are_format_errors(self):
        for content in ('{}', '{"answer":"A","reason":"because"}',
                        '{"Answer":"A"}'):
            with self.subTest(content=content):
                self.assert_invalid(content, True)

    def test_wrong_top_level_types_are_format_errors(self):
        for content in ('["A"]', '"A"', 'null', '1', 'true'):
            with self.subTest(content=content):
                self.assert_invalid(content, True)

    def test_wrong_answer_types_or_values_are_format_errors(self):
        for value in (None, True, 1, [], {}, "a", "E", "А", " A", "A ", "AB"):
            with self.subTest(value=value):
                self.assert_invalid(json.dumps({"answer": value}), True)

    def test_nontext_content_is_rejected_without_crashing(self):
        for content in (None, 12, {}, [], b'{"answer":"A"}'):
            with self.subTest(content=content):
                self.assert_invalid(content, False)

    def test_length_rejects_otherwise_correct_json(self):
        self.assert_invalid('{"answer":"A"}', True, "length")

    def test_length_does_not_make_malformed_json_valid(self):
        self.assert_invalid('{"answer":', False, "length")


class TaskCorpusTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parent
        self.corpus = json.loads((root / "tasks.json").read_text(encoding="utf-8"))

    def added_task(self, category):
        return next(task for task in self.corpus["tasks"][6:]
                    if task["category"] == category)

    def json_example(self, task):
        return json.loads(re.search(r"```json\n(.*?)\n```", task["prompt"], re.S)[1])

    def options(self, task):
        return dict(re.findall(r"^([A-D])\) (.+)$", task["prompt"], re.M))

    def assert_unique_option(self, task, expected):
        options = self.options(task)
        self.assertEqual(set(options), set("ABCD"))
        self.assertEqual([key for key, value in options.items() if value == expected],
                         [task["answer"]])

    def test_frozen_corpus_preserves_all_original_prompts_and_keys(self):
        root = Path(__file__).resolve().parent
        corpus = json.loads((root / "tasks.json").read_text(encoding="utf-8"))
        self.assertEqual(corpus["source_branch"], "task-3")
        self.assertRegex(corpus["source_commit"], r"^[0-9a-f]{40}$")
        source = subprocess.check_output(
            ["git", "show", corpus["source_commit"] + ":main.py"],
            cwd=root, text=True,
        )
        assignment = next(node for node in ast.parse(source).body
                          if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "TASKS"
                                  for target in node.targets))
        original = ast.literal_eval(assignment.value)
        self.assertEqual(len(corpus["tasks"]), 10)
        self.assertEqual(corpus["tasks"][:6], [
            {"id": task_id, "prompt": prompt, "answer": answer}
            for task_id, prompt, answer in original
        ])
        self.assertEqual([task["answer"] for task in corpus["tasks"][:6]],
                         ["A", "B", "C", "B", "B", "D"])

    def test_version_metadata_and_unique_ids(self):
        self.assertEqual(self.corpus["source_task_count"], 6)
        self.assertEqual(self.corpus["dataset_version"], "day5-10-v2")
        self.assertTrue(self.corpus["added_tasks_note"].strip())
        tasks = self.corpus["tasks"]
        self.assertEqual(len({task["id"] for task in tasks}), 10)
        additions = tasks[6:]
        self.assertEqual({task["category"] for task in additions}, {
            "structured_data_extraction", "reading_comprehension",
            "python_code_tracing", "dependency_scheduling",
        })
        self.assertEqual({task["answer"] for task in additions}, set("ABCD"))
        for task in additions:
            self.assertTrue(task["reference_solution"].strip())
            self.assertEqual(set(self.options(task)), set("ABCD"))

    def test_extraction_answer_is_sum_of_paid_non_cancelled_orders(self):
        task = self.added_task("structured_data_extraction")
        orders = self.json_example(task)
        selected = [order for order in orders
                    if order["paid"] and not order["cancelled"]]
        self.assertEqual(len(selected), 3)
        total = sum(order["amount_rub"] for order in selected)
        self.assertEqual(total, 3250)
        self.assert_unique_option(task, str(total))

    def test_reading_answer_follows_explicit_policy_only(self):
        task = self.added_task("reading_comprehension")
        requests = self.json_example(task)
        # The stated policy includes day 14, excludes personalized goods,
        # and explicitly does not exclude discounted purchases.
        eligible = [request["id"] for request in requests
                    if 0 <= request["days"] <= 14
                    and request["unused"] and request["proof"]
                    and not request["personalized"]]
        self.assertEqual(eligible, ["R2"])
        self.assertTrue(next(r for r in requests if r["id"] == "R2")["discounted"])
        self.assert_unique_option(task, ", ".join(eligible))

    def test_python_answer_matches_execution_of_presented_example(self):
        task = self.added_task("python_code_tracing")
        code = re.search(r"```python\n(.*?)\n```", task["prompt"], re.S)[1]
        namespace = {}
        exec(compile(code, "<task-example>", "exec"),
             {"__builtins__": {"len": len}}, namespace)
        self.assertEqual(namespace["result"], [3, 3, 2, 2, 1])
        self.assert_unique_option(task, repr(namespace["result"]))

    def test_schedule_answer_is_exhaustive_two_worker_nonpreemptive_optimum(self):
        task = self.added_task("dependency_scheduling")
        jobs = self.json_example(task)
        durations = {job["id"]: job["duration"] for job in jobs}
        dependencies = {job["id"]: set(job["after"]) for job in jobs}

        @lru_cache(None)
        def remaining_time(done, running):
            if len(done) == len(jobs):
                return 0
            busy = {name for name, _ in running}
            ready = sorted(name for name in durations
                           if name not in done and name not in busy
                           and dependencies[name] <= set(done))
            candidates = []
            for count in range(min(2 - len(running), len(ready)) + 1):
                for starts in itertools.combinations(ready, count):
                    active = running + tuple((name, durations[name]) for name in starts)
                    if not active:
                        continue
                    elapsed = min(duration for _, duration in active)
                    finished = frozenset(name for name, duration in active
                                         if duration == elapsed)
                    next_running = tuple(sorted((name, duration - elapsed)
                                                for name, duration in active
                                                if duration > elapsed))
                    candidates.append(elapsed + remaining_time(done | finished, next_running))
            return min(candidates)

        optimum = remaining_time(frozenset(), ())
        self.assertEqual(optimum, 11)
        self.assert_unique_option(task, str(optimum))


if __name__ == "__main__":
    unittest.main()
