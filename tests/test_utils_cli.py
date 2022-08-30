import unittest

from huggingface_hub.commands._cli_utils import ANSI, tabulate


class TestCLIUtils(unittest.TestCase):
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

    def test_tabulate_utility(self) -> None:
        """Test `tabulate` works as expected."""
        rows = [[1, 2, 3], ["a very long value", "foo", "bar"], ["", 123, 456]]
        headers = ["Header 1", "something else", "a third column"]
        self.assertEqual(
            tabulate(rows=rows, headers=headers).strip(),
            """
Header 1          something else a third column 
----------------- -------------- -------------- 
                1              2              3 
a very long value foo            bar            
                             123            456""".strip(),
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
