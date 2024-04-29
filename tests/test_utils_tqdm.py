import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest
from pytest import CaptureFixture

from huggingface_hub.utils import (
    SoftTemporaryDirectory,
    are_progress_bars_disabled,
    disable_progress_bars,
    enable_progress_bars,
    tqdm,
    tqdm_stream_file,
)


class CapsysBaseTest(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def capsys(self, capsys: CaptureFixture) -> None:
        """Workaround to make capsys work in unittest framework.

        Capsys is a convenient pytest fixture to capture stdout.
        See https://waylonwalker.com/pytest-capsys/.

        Taken from https://github.com/pytest-dev/pytest/issues/2504#issuecomment-309475790.
        """
        self.capsys = capsys


class TestTqdmUtils(CapsysBaseTest):
    def setUp(self) -> None:
        """Get verbosity to set it back after the tests."""
        self._previous_are_progress_bars_disabled = are_progress_bars_disabled()
        return super().setUp()

    def tearDown(self) -> None:
        """Set back progress bars verbosity as before testing."""
        if self._previous_are_progress_bars_disabled:
            disable_progress_bars()
        else:
            enable_progress_bars()

    @patch("huggingface_hub.utils._tqdm.HF_HUB_DISABLE_PROGRESS_BARS", None)
    def test_tqdm_helpers(self) -> None:
        """Test helpers to enable/disable progress bars."""
        disable_progress_bars()
        assert are_progress_bars_disabled()

        enable_progress_bars()
        assert not are_progress_bars_disabled()

    @patch("huggingface_hub.utils._tqdm.HF_HUB_DISABLE_PROGRESS_BARS", True)
    def test_cannot_enable_tqdm_when_env_variable_is_set(self) -> None:
        """
        Test helpers cannot enable/disable progress bars when
        `HF_HUB_DISABLE_PROGRESS_BARS` is set.
        """
        disable_progress_bars()
        assert are_progress_bars_disabled()

        with self.assertWarns(UserWarning):
            enable_progress_bars()
        assert are_progress_bars_disabled()  # Still disabled

    @patch("huggingface_hub.utils._tqdm.HF_HUB_DISABLE_PROGRESS_BARS", False)
    def test_cannot_disable_tqdm_when_env_variable_is_set(self) -> None:
        """
        Test helpers cannot enable/disable progress bars when
        `HF_HUB_DISABLE_PROGRESS_BARS` is set.
        """
        enable_progress_bars()
        assert not are_progress_bars_disabled()

        with self.assertWarns(UserWarning):
            disable_progress_bars()
        assert not are_progress_bars_disabled()  # Still enabled

    @patch("huggingface_hub.utils._tqdm.HF_HUB_DISABLE_PROGRESS_BARS", None)
    def test_tqdm_disabled(self) -> None:
        """Test TQDM not outputting anything when globally disabled."""
        disable_progress_bars()
        for _ in tqdm(range(10)):
            pass

        captured = self.capsys.readouterr()
        self.assertEqual(captured.out, "")
        self.assertEqual(captured.err, "")

    @patch("huggingface_hub.utils._tqdm.HF_HUB_DISABLE_PROGRESS_BARS", None)
    def test_tqdm_disabled_cannot_be_forced(self) -> None:
        """Test TQDM cannot be forced when globally disabled."""
        disable_progress_bars()
        for _ in tqdm(range(10), disable=False):
            pass

        captured = self.capsys.readouterr()
        self.assertEqual(captured.out, "")
        self.assertEqual(captured.err, "")

    @patch("huggingface_hub.utils._tqdm.HF_HUB_DISABLE_PROGRESS_BARS", None)
    def test_tqdm_can_be_disabled_when_globally_enabled(self) -> None:
        """Test TQDM can still be locally disabled even when globally enabled."""
        enable_progress_bars()
        for _ in tqdm(range(10), disable=True):
            pass

        captured = self.capsys.readouterr()
        self.assertEqual(captured.out, "")
        self.assertEqual(captured.err, "")

    @patch("huggingface_hub.utils._tqdm.HF_HUB_DISABLE_PROGRESS_BARS", None)
    def test_tqdm_enabled(self) -> None:
        """Test TQDM work normally when globally enabled."""
        enable_progress_bars()
        for _ in tqdm(range(10)):
            pass

        captured = self.capsys.readouterr()
        self.assertEqual(captured.out, "")
        self.assertIn("10/10", captured.err)  # tqdm log

    def test_tqdm_stream_file(self) -> None:
        with SoftTemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "config.json"
            with filepath.open("w") as f:
                f.write("#" * 1000)

            with tqdm_stream_file(filepath) as f:
                while True:
                    data = f.read(100)
                    if not data:
                        break
                    time.sleep(0.001)  # Simulate a delay between each chunk

            captured = self.capsys.readouterr()
            self.assertEqual(captured.out, "")
            self.assertIn("config.json: 100%", captured.err)  # log file name
            self.assertIn("|█████████", captured.err)  # tqdm bar
            self.assertIn("1.00k/1.00k", captured.err)  # size in B


class TestTqdmGroup(CapsysBaseTest):
    def setUp(self):
        """Set up the initial condition for each test."""
        super().setUp()
        enable_progress_bars()  # Ensure all are enabled before each test

    def tearDown(self):
        """Clean up after each test."""
        super().tearDown()
        enable_progress_bars()

    @patch("huggingface_hub.utils._tqdm.HF_HUB_DISABLE_PROGRESS_BARS", None)
    def test_disable_specific_group(self):
        """Test disabling a specific group only affects that group and its subgroups."""
        disable_progress_bars("peft.foo")
        assert not are_progress_bars_disabled("peft")
        assert not are_progress_bars_disabled("peft.something")
        assert are_progress_bars_disabled("peft.foo")
        assert are_progress_bars_disabled("peft.foo.bar")

    @patch("huggingface_hub.utils._tqdm.HF_HUB_DISABLE_PROGRESS_BARS", None)
    def test_enable_specific_subgroup(self):
        """Test that enabling a subgroup does not affect the disabled state of its parent."""
        disable_progress_bars("peft.foo")
        enable_progress_bars("peft.foo.bar")
        assert are_progress_bars_disabled("peft.foo")
        assert not are_progress_bars_disabled("peft.foo.bar")

    @patch("huggingface_hub.utils._tqdm.HF_HUB_DISABLE_PROGRESS_BARS", True)
    def test_disable_override_by_environment_variable(self):
        """Ensure progress bars are disabled regardless of local settings when environment variable is set."""
        with self.assertWarns(UserWarning):
            enable_progress_bars()
        assert are_progress_bars_disabled("peft")
        assert are_progress_bars_disabled("peft.foo")

    @patch("huggingface_hub.utils._tqdm.HF_HUB_DISABLE_PROGRESS_BARS", False)
    def test_enable_override_by_environment_variable(self):
        """Ensure progress bars are enabled regardless of local settings when environment variable is set."""
        with self.assertWarns(UserWarning):
            disable_progress_bars("peft.foo")
        assert not are_progress_bars_disabled("peft.foo")

    @patch("huggingface_hub.utils._tqdm.HF_HUB_DISABLE_PROGRESS_BARS", None)
    def test_partial_group_name_not_affected(self):
        """Ensure groups with similar names but not exactly matching are not affected."""
        disable_progress_bars("peft.foo")
        assert not are_progress_bars_disabled("peft.footprint")

    @patch("huggingface_hub.utils._tqdm.HF_HUB_DISABLE_PROGRESS_BARS", None)
    def test_nested_subgroup_behavior(self):
        """Test enabling and disabling nested subgroups."""
        disable_progress_bars("peft")
        enable_progress_bars("peft.foo")
        disable_progress_bars("peft.foo.bar")
        assert are_progress_bars_disabled("peft")
        assert not are_progress_bars_disabled("peft.foo")
        assert are_progress_bars_disabled("peft.foo.bar")

    @patch("huggingface_hub.utils._tqdm.HF_HUB_DISABLE_PROGRESS_BARS", None)
    def test_empty_group_is_root(self):
        """Test the behavior with invalid or empty group names."""
        disable_progress_bars("")
        assert not are_progress_bars_disabled("peft")

        enable_progress_bars("123.invalid.name")
        assert not are_progress_bars_disabled("123.invalid.name")

    @patch("huggingface_hub.utils._tqdm.HF_HUB_DISABLE_PROGRESS_BARS", None)
    def test_multiple_level_toggling(self):
        """Test multiple levels of enabling and disabling."""
        disable_progress_bars("peft")
        enable_progress_bars("peft.foo")
        disable_progress_bars("peft.foo.bar.something")
        assert are_progress_bars_disabled("peft")
        assert not are_progress_bars_disabled("peft.foo")
        assert are_progress_bars_disabled("peft.foo.bar.something")

    def test_progress_bar_respects_group(self) -> None:
        disable_progress_bars("foo.bar")
        for _ in tqdm(range(10), name="foo.bar.something"):
            pass
        captured = self.capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

        enable_progress_bars("foo.bar.something")
        for _ in tqdm(range(10), name="foo.bar.something"):
            pass
        captured = self.capsys.readouterr()
        assert captured.out == ""
        assert "10/10" in captured.err
