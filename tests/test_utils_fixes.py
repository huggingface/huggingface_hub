import logging
import unittest
from pathlib import Path

import filelock
import pytest
from filelock import FileLock, SoftFileLock

from huggingface_hub.utils import SoftTemporaryDirectory, WeakFileLock, yaml_dump
from huggingface_hub.utils._fixes import (
    _CIFS_SUPER_MAGIC,
    _GPFS_SUPER_MAGIC,
    _LUSTRE_SUPER_MAGIC,
    _NFS_SUPER_MAGIC,
    _SMB_SUPER_MAGIC,
    _flock_actually_serializes,
    _should_use_soft_lock,
)


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


@pytest.fixture(autouse=True)
def _clear_soft_lock_caches() -> None:
    """`_should_use_soft_lock` and `_flock_actually_serializes` are `lru_cache`'d per cache_dir.

    Tests parametrize over magics / env vars for the same `tmp_path`, so we must clear both caches
    before and after each test, otherwise the second test in a class would see a stale decision.
    """
    _should_use_soft_lock.cache_clear()
    _flock_actually_serializes.cache_clear()
    yield
    _should_use_soft_lock.cache_clear()
    _flock_actually_serializes.cache_clear()


class TestSoftLockDetection:
    @pytest.mark.parametrize(
        "magic",
        [_LUSTRE_SUPER_MAGIC, _GPFS_SUPER_MAGIC, _NFS_SUPER_MAGIC, _SMB_SUPER_MAGIC, _CIFS_SUPER_MAGIC],
    )
    def test_known_broken_magic_uses_soft_lock(
        self, magic: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("huggingface_hub.utils._fixes._fs_magic", lambda _path: magic)
        with WeakFileLock(tmp_path / ".lock") as lock:
            assert isinstance(lock, SoftFileLock)

    def test_local_filesystem_uses_native_flock(self, tmp_path: Path) -> None:
        # `tmp_path` lives under `/tmp` (ext4/tmpfs in CI and on dev boxes), which is not in the
        # known-broken set. This also exercises the real `_flock_actually_serializes` probe.
        with WeakFileLock(tmp_path / ".lock") as lock:
            assert isinstance(lock, FileLock)
            assert not isinstance(lock, SoftFileLock)

    def test_env_force_soft_overrides_detection(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("huggingface_hub.constants.HF_HUB_USE_SOFT_FILELOCK", True)
        # Even with a "looks healthy" magic, the env var must win.
        monkeypatch.setattr("huggingface_hub.utils._fixes._fs_magic", lambda _path: 0xEF53)  # ext4
        with WeakFileLock(tmp_path / ".lock") as lock:
            assert isinstance(lock, SoftFileLock)

    def test_env_force_flock_overrides_detection(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("huggingface_hub.constants.HF_HUB_FORCE_FLOCK", True)
        # Even when the FS reports a known-broken magic, the env var must win.
        monkeypatch.setattr("huggingface_hub.utils._fixes._fs_magic", lambda _path: _LUSTRE_SUPER_MAGIC)
        with WeakFileLock(tmp_path / ".lock") as lock:
            assert isinstance(lock, FileLock)
            assert not isinstance(lock, SoftFileLock)

    def test_probe_failure_falls_back_to_soft(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Unknown magic (0) → falls through to the runtime probe; force probe to fail-closed.
        monkeypatch.setattr("huggingface_hub.utils._fixes._fs_magic", lambda _path: 0)
        monkeypatch.setattr("huggingface_hub.utils._fixes._flock_actually_serializes", lambda _path: False)
        with WeakFileLock(tmp_path / ".lock") as lock:
            assert isinstance(lock, SoftFileLock)
