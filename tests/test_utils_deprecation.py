import re
import unittest
import warnings

import pytest

from huggingface_hub.utils._deprecation import (
    _deprecate_arguments,
    _deprecate_positional_args,
)


def dummy(a, b="b", c="c") -> str:
    return f"{a}{b}{c}"


def dummy_kwonly(a, *, b="b", c="c") -> str:
    return dummy(a, b, c)


class TestDeprecationUtils(unittest.TestCase):
    def test_deprecate_positional_args(self):
        """Test warnings are triggered when using deprecated positional args.

        Also test that the values are well passed to the decorated function.
        """
        dummy_position_deprecated = _deprecate_positional_args(
            dummy_kwonly, version="xxx"
        )

        with warnings.catch_warnings():
            # Assert no warnings when used correctly.
            # Taken from https://docs.pytest.org/en/latest/how-to/capture-warnings.html#additional-use-cases-of-warnings-in-tests
            warnings.simplefilter("error")
            self.assertEqual(dummy_position_deprecated(a="A", b="B", c="C"), "ABC")
            self.assertEqual(dummy_position_deprecated("A", b="B", c="C"), "ABC")

        with pytest.warns(FutureWarning):
            self.assertEqual(dummy_position_deprecated("A", "B", c="C"), "ABC")

        with pytest.warns(FutureWarning):
            self.assertEqual(dummy_position_deprecated("A", "B", "C"), "ABC")

    def test_deprecate_arguments(self):
        """Test warnings are triggered when using deprecated arguments.

        Also test that the values are well passed to the decorated function.
        """
        dummy_c_deprecated = _deprecate_arguments(
            dummy, version="xxx", deprecated_args={"c"}
        )
        dummy_b_c_deprecated = _deprecate_arguments(
            dummy, version="xxx", deprecated_args={"b", "c"}
        )

        with warnings.catch_warnings():
            # Assert no warnings when used correctly.
            # Taken from https://docs.pytest.org/en/latest/how-to/capture-warnings.html#additional-use-cases-of-warnings-in-tests
            warnings.simplefilter("error")
            self.assertEqual(dummy_c_deprecated("A"), "Abc")
            self.assertEqual(dummy_c_deprecated("A", "B"), "ABc")
            self.assertEqual(dummy_c_deprecated("A", b="B"), "ABc")

            self.assertEqual(dummy_b_c_deprecated("A"), "Abc")

        with pytest.warns(FutureWarning):
            self.assertEqual(dummy_c_deprecated("A", "B", "C"), "ABC")

        with pytest.warns(FutureWarning):
            self.assertEqual(dummy_c_deprecated("A", c="C"), "AbC")

        with pytest.warns(FutureWarning):
            self.assertEqual(dummy_c_deprecated("A", b="B", c="C"), "ABC")

        with pytest.warns(FutureWarning):
            self.assertEqual(dummy_b_c_deprecated("A", b="B"), "ABc")

        with pytest.warns(FutureWarning):
            self.assertEqual(dummy_b_c_deprecated("A", c="C"), "AbC")

        with pytest.warns(FutureWarning):
            self.assertEqual(dummy_b_c_deprecated("A", b="B", c="C"), "ABC")
