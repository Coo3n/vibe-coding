import json
import unittest

from main import inspect_answer


class AnswerValidationTests(unittest.TestCase):
    def test_accepts_one_idea(self):
        result = inspect_answer('ideas', json.dumps({'ideas': ['Мою посуду: губка просит отпуск.']}))
        self.assertTrue(result['format_valid'])

    def test_rejects_wrong_idea_count_and_invalid_strings(self):
        for ideas in ([], ['Идея'] * 5, [''], ['слово ' * 16], ['Первая\nВторая'], [1]):
            with self.subTest(ideas=ideas):
                self.assertFalse(inspect_answer('ideas', json.dumps({'ideas': ideas}))['format_valid'])

    def test_scores_multistage_reservoir_answer(self):
        # 1/3 + 2*(1/12+1/18-1/36) + 3*(1/18-1/36) = 23/36.
        # Remaining 13/36 takes 6.5 h with the second pipe: total 690 min.
        self.assertTrue(inspect_answer('math', '{"answer":690}')['answer_correct'])
        self.assertFalse(inspect_answer('math', '{"answer":960}')['answer_correct'])


if __name__ == '__main__':
    unittest.main()
