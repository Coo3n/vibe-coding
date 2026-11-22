from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from app.agent import Agent, OfflineClient
from app.memory import MemoryStore


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'memory.json'
        self.memory = MemoryStore(self.path)
        self.agent = Agent(OfflineClient(), self.memory)

    def test_late_update_with_different_timezone_does_not_replace_current_value(self):
        self.memory.durable.record('срок', '27 марта', '2026-11-20T14:00:00+03:00')
        result = self.memory.durable.record('срок', '20 марта', '2026-11-20T10:30:00Z')
        self.assertEqual(result, 'ignored')
        self.assertEqual(self.agent.ask('Какой сейчас срок?')['answer'], 'срок — 27 марта')
        self.assertEqual(self.memory.durable.read()['changes'], [])

    def test_other_session_update_invalidates_old_turns_and_masks_retired_values(self):
        self.memory.durable.record('срок', '20 марта')
        self.memory.record_working('черновик', 'Вылет 20 марта.')
        self.agent.ask('Какой сейчас срок?')
        writer = MemoryStore(self.path)
        writer.durable.record('срок', '27 марта')
        reply = self.agent.ask('Подтверждаете срок 20 марта?')
        prompt = '\n'.join(entry['content'] for entry in reply['messages'])
        self.assertNotIn('20 марта', prompt)
        self.assertNotIn('Какой сейчас срок?', prompt)
        self.assertIn('27 марта', reply['answer'])

    def test_deleted_fact_does_not_reappear_from_working_or_delayed_write(self):
        self.memory.record_working('срок', 'рабочая версия')
        self.memory.durable.record('срок', '20 марта', '2026-11-20T11:00:00Z')
        self.memory.durable.forget('срок', '2026-11-20T12:00:00Z')
        self.assertEqual(
            self.memory.durable.record('срок', '20 марта', '2026-11-20T11:59:59Z'), 'ignored')
        reply = self.agent.ask('Какой сейчас срок?')
        self.assertNotIn('рабочая версия', reply['answer'])
        self.assertEqual(reply['answer'], 'Нет актуальной записи по этому вопросу.')
        self.assertEqual(self.memory.durable.read()['changes'][0]['operation'], 'delete')

    def test_reactivation_retires_only_currently_obsolete_values(self):
        field = 'версия'
        self.memory.durable.record(field, 'запуск X')
        self.memory.durable.record(field, 'запуск Y')
        self.memory.durable.record(field, 'запуск X')
        reply = self.agent.ask('Какая версия?')
        self.assertEqual(reply['answer'], 'версия — запуск X')
        self.assertNotIn('запуск Y', '\n'.join(entry['content'] for entry in reply['messages']))


if __name__ == '__main__':
    unittest.main()
