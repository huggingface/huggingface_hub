import unittest

import pytest
from pytest import CaptureFixture

from huggingface_hub.utils import (
    are_progress_bars_disabled,
    disable_progress_bars,
    enable_progress_bars,
    tqdm,
)


class TestTqdmUtils(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def capsys(self, capsys: CaptureFixture) -> None:
        """Workaround to make capsys work in unittest framework.

        Capsys is a convenient pytest fixture to capture stdout.
        See https://waylonwalker.com/pytest-capsys/.

        Taken from https://github.com/pytest-dev/pytest/issues/2504#issuecomment-309475790.
        """
        self.capsys = capsys

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

    def test_tqdm_helpers(self) -> None:
        """Test helpers to enable/disable progress bars."""
        disable_progress_bars()
        self.assertTrue(are_progress_bars_disabled())

        enable_progress_bars()
        self.assertFalse(are_progress_bars_disabled())

    def test_tqdm_disabled(self) -> None:
        """Test TQDM not outputing anything when globally disabled."""
        disable_progress_bars()
        for _ in tqdm(range(10)):
            pass

        captured = self.capsys.readouterr()
        self.assertEqual(captured.out, "")
        self.assertEqual(captured.err, "")

    def test_tqdm_disabled_cannot_be_forced(self) -> None:
        """Test TQDM cannot be forced when globally disabled."""
        disable_progress_bars()
        for _ in tqdm(range(10), disable=False):
            pass

        captured = self.capsys.readouterr()
        self.assertEqual(captured.out, "")
        self.assertEqual(captured.err, "")

    def test_tqdm_can_be_disabled_when_globally_enabled(self) -> None:
        """Test TQDM can still be locally disabled even when globally enabled."""
        enable_progress_bars()
        for _ in tqdm(range(10), disable=True):
            pass

        captured = self.capsys.readouterr()
        self.assertEqual(captured.out, "")
        self.assertEqual(captured.err, "")

    def test_tqdm_enabled(self) -> None:
        """Test TQDM work normally when globally enabled."""
        enable_progress_bars()
        for _ in tqdm(range(10)):
            pass

        captured = self.capsys.readouterr()
        self.assertEqual(captured.out, "")
        self.assertIn("10/10", captured.err)  # tqdm log
