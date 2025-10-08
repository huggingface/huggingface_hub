import os
import unittest
from unittest import mock

from huggingface_hub.utils._terminal import ANSI, tabulate


class TestCLIUtils(unittest.TestCase):
    @mock.patch.dict(os.environ, {}, clear=True)
    def test_ansi_utils(self) -> None:
        """Test `ANSI` works as expected."""
        self.assertEqual(
            ANSI.bold("this is bold"),
            "\x1b[1mthis is bold\x1b[0m",
        )

        self.assertEqual(
            ANSI.gray("this is gray"),
            "\x1b[90mthis is gray\x1b[0m",
        )

        self.assertEqual(
            ANSI.red("this is red"),
            "\x1b[1m\x1b[31mthis is red\x1b[0m",
        )

        self.assertEqual(
            ANSI.gray(ANSI.bold("this is bold and grey")),
            "\x1b[90m\x1b[1mthis is bold and grey\x1b[0m\x1b[0m",
        )

    @mock.patch.dict(os.environ, {"NO_COLOR": "1"}, clear=True)
    def test_ansi_no_color(self) -> None:
        """Test `ANSI` respects `NO_COLOR` env var."""

        self.assertEqual(
            ANSI.bold("this is bold"),
            "this is bold",
        )

        self.assertEqual(
            ANSI.gray("this is gray"),
            "this is gray",
        )

        self.assertEqual(
            ANSI.red("this is red"),
            "this is red",
        )

        self.assertEqual(
            ANSI.gray(ANSI.bold("this is bold and grey")),
            "this is bold and grey",
        )

    def test_tabulate_utility(self) -> None:
        """Test `tabulate` works as expected."""
        rows = [[1, 2, 3], ["a very long value", "foo", "bar"], ["", 123, 456]]
        headers = ["Header 1", "something else", "a third column"]
        self.assertEqual(
            tabulate(rows=rows, headers=headers),
            "Header 1          something else a third column \n"
            "----------------- -------------- -------------- \n"
            "                1              2              3 \n"
            "a very long value foo            bar            \n"
            "                             123            456 ",
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
