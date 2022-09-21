import unittest

from huggingface_hub.utils import yaml_dump


class TestYamlDump(unittest.TestCase):
    def test_yaml_dump_emoji(self) -> None:
        self.assertEqual(yaml_dump({"emoji": "👀"}), 'emoji: "👀"\n')

    def test_yaml_dump_japanese_characters(self) -> None:
        self.assertEqual(yaml_dump({"some unicode": "日本か"}), 'some unicode: "日本か"\n')
