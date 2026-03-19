import os
import sys
import unittest
from unittest import mock

from huggingface_hub.utils._terminal import ANSI, tabulate


class TestCLIUtils(unittest.TestCase):
    def _tty_stream(self):
        return mock.Mock(isatty=mock.Mock(return_value=True))

    def _non_tty_stream(self):
        return mock.Mock(isatty=mock.Mock(return_value=False))

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_ansi_utils(self) -> None:
        """Test `ANSI` works as expected."""
        tty_stream = self._tty_stream()

        self.assertEqual(
            ANSI.bold("this is bold", file=tty_stream),
            "\x1b[1mthis is bold\x1b[0m",
        )

        self.assertEqual(
            ANSI.gray("this is gray", file=tty_stream),
            "\x1b[90mthis is gray\x1b[0m",
        )

        self.assertEqual(
            ANSI.red("this is red", file=tty_stream),
            "\x1b[1m\x1b[31mthis is red\x1b[0m",
        )

        self.assertEqual(
            ANSI.gray(ANSI.bold("this is bold and grey", file=tty_stream), file=tty_stream),
            "\x1b[90m\x1b[1mthis is bold and grey\x1b[0m\x1b[0m",
        )

    @mock.patch.dict(os.environ, {"NO_COLOR": "1"}, clear=True)
    def test_ansi_no_color(self) -> None:
        """Test `ANSI` respects `NO_COLOR` env var."""
        tty_stream = self._tty_stream()

        self.assertEqual(
            ANSI.bold("this is bold", file=tty_stream),
            "this is bold",
        )

        self.assertEqual(
            ANSI.gray("this is gray", file=tty_stream),
            "this is gray",
        )

        self.assertEqual(
            ANSI.red("this is red", file=tty_stream),
            "this is red",
        )

        self.assertEqual(
            ANSI.gray(ANSI.bold("this is bold and grey", file=tty_stream), file=tty_stream),
            "this is bold and grey",
        )

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_ansi_defaults_to_stdout_stream(self) -> None:
        with mock.patch.object(sys, "stdout", self._tty_stream()):
            self.assertEqual(
                ANSI.bold("this is bold"),
                "\x1b[1mthis is bold\x1b[0m",
            )

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_ansi_skips_color_for_non_tty_stdout(self) -> None:
        with mock.patch.object(sys, "stdout", self._non_tty_stream()):
            self.assertEqual(
                ANSI.bold("this is bold"),
                "this is bold",
            )

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_ansi_skips_color_for_non_tty_stream(self) -> None:
        self.assertEqual(
            ANSI.yellow("warning", file=self._non_tty_stream()),
            "warning",
        )

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_ansi_uses_target_stream_not_stdout(self) -> None:
        with mock.patch.object(sys, "stdout", self._non_tty_stream()):
            self.assertEqual(
                ANSI.yellow("warning", file=self._tty_stream()),
                "\x1b[33mwarning\x1b[0m",
            )

    def test_tabulate_utility(self) -> None:
        """Test `tabulate` works as expected."""
        rows = [[1, 2, 3], ["a very long value", "foo", "bar"], ["", 123, 456]]
        headers = ["Header 1", "something else", "a third column"]
        self.assertEqual(
            tabulate(rows=rows, headers=headers),
            "Header 1          something else a third column\n"
            "----------------- -------------- --------------\n"
            "1                 2              3             \n"
            "a very long value foo            bar           \n"
            "                  123            456           ",
        )

    def test_tabulate_utility_with_too_short_row(self) -> None:
        """
        Test `tabulate` throw IndexError when a row has less values than the header
        list.
        """
        self.assertRaises(
            IndexError,
            tabulate,
            rows=[[1]],
            headers=["Header 1", "Header 2"],
        )
