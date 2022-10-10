import unittest

from huggingface_hub.commands._cli_utils import ANSI, tabulate


class TestCLIUtils(unittest.TestCase):
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
