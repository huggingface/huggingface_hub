import unittest
from pathlib import Path

from huggingface_hub.utils import SoftTemporaryDirectory, yaml_dump


class TestYamlDump(unittest.TestCase):
    def test_yaml_dump_emoji(self) -> None:
        self.assertEqual(yaml_dump({"emoji": "👀"}), "emoji: 👀\n")

    def test_yaml_dump_japanese_characters(self) -> None:
        self.assertEqual(yaml_dump({"some unicode": "日本か"}), "some unicode: 日本か\n")

    def test_yaml_dump_explicit_no_unicode(self) -> None:
        self.assertEqual(yaml_dump({"emoji": "👀"}, allow_unicode=False), 'emoji: "\\U0001F440"\n')


class TestTemporaryDirectory(unittest.TestCase):
    def test_temporary_directory(self) -> None:
        with SoftTemporaryDirectory(prefix="prefix", suffix="suffix") as path:
            self.assertIsInstance(path, Path)
            self.assertTrue(path.name.startswith("prefix"))
            self.assertTrue(path.name.endswith("suffix"))
            self.assertTrue(path.is_dir())
        # Tmpdir is deleted
        self.assertFalse(path.is_dir())
