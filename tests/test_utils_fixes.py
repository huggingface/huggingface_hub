import logging
import unittest
from pathlib import Path

import filelock
import pytest

from huggingface_hub.utils import SoftTemporaryDirectory, WeakFileLock, yaml_dump


class TestYamlDump(unittest.TestCase):
    def test_yaml_dump_emoji(self) -> None:
        assert yaml_dump({"emoji": "👀"}) == "emoji: 👀\n"

    def test_yaml_dump_japanese_characters(self) -> None:
        assert yaml_dump({"some unicode": "日本か"}) == "some unicode: 日本か\n"

    def test_yaml_dump_explicit_no_unicode(self) -> None:
        assert yaml_dump({"emoji": "👀"}, allow_unicode=False) == 'emoji: "\\U0001F440"\n'


class TestTemporaryDirectory(unittest.TestCase):
    def test_temporary_directory(self) -> None:
        with SoftTemporaryDirectory(prefix="prefix", suffix="suffix") as path:
            assert isinstance(path, Path)
            assert path.name.startswith("prefix")
            assert path.name.endswith("suffix")
            assert path.is_dir()
        # Tmpdir is deleted
        assert not (path.is_dir())


class TestWeakFileLock:
    def test_lock_log_every(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr("huggingface_hub.constants.FILELOCK_LOG_EVERY_SECONDS", 0.1)
        lock_file = tmp_path / ".lock"

        with caplog.at_level(logging.INFO, logger="huggingface_hub.utils._fixes"):
            with WeakFileLock(lock_file):
                with pytest.raises(filelock.Timeout) as exc_info:
                    with WeakFileLock(lock_file, timeout=0.3):
                        pass
                assert exc_info.value.lock_file == str(lock_file)

        assert len(caplog.records) >= 3
        assert caplog.records[0].message.startswith(f"Still waiting to acquire lock on {lock_file}")
