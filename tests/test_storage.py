import tempfile
import unittest
from unittest.mock import patch

from backend import storage


class StorageConversationReuseTest(unittest.TestCase):
    def test_reusable_empty_conversation_returns_newest_empty_thread(self):
        with tempfile.TemporaryDirectory() as data_dir:
            with patch("backend.storage.DATA_DIR", data_dir):
                first = storage.create_conversation("first-empty")
                storage.create_conversation("used-thread")
                storage.add_user_message("used-thread", "hello")
                second = storage.create_conversation("second-empty")

                reusable = storage.get_reusable_empty_conversation()

        self.assertEqual(reusable["id"], second["id"])
        self.assertEqual(len(first["messages"]), 0)

    def test_list_conversations_collapses_duplicate_empty_threads(self):
        with tempfile.TemporaryDirectory() as data_dir:
            with patch("backend.storage.DATA_DIR", data_dir):
                storage.create_conversation("first-empty")
                storage.create_conversation("second-empty")
                storage.create_conversation("used-thread")
                storage.add_user_message("used-thread", "hello")

                conversations = storage.list_conversations()

        empty_conversations = [
            conversation
            for conversation in conversations
            if conversation["message_count"] == 0
        ]
        used_conversations = [
            conversation
            for conversation in conversations
            if conversation["id"] == "used-thread"
        ]

        self.assertEqual(len(empty_conversations), 1)
        self.assertEqual(len(used_conversations), 1)


if __name__ == "__main__":
    unittest.main()
