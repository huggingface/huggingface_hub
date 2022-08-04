import re
import unittest
import warnings

import pytest

from huggingface_hub.utils._deprecation import (
    _deprecate_arguments,
    _deprecate_positional_args,
)


def dummy(a, b="b", c="c"):
    pass


def dummy_kwonly(a, *, b="b", c="c"):
    pass


class TestDeprecationUtils(unittest.TestCase):
    def test_deprecate_positional_args(self):
        """Test warnings are triggered when using deprecated positional args."""
        dummy_position_deprecated = _deprecate_positional_args(
            dummy_kwonly, version="xxx"
        )

        with warnings.catch_warnings():
            # Assert no warnings when used correctly.
            # Taken from https://docs.pytest.org/en/latest/how-to/capture-warnings.html#additional-use-cases-of-warnings-in-tests
            warnings.simplefilter("error")
            dummy_position_deprecated(a="A", b="B", c="C")
            dummy_position_deprecated("A", b="B", c="C")

        with pytest.warns(FutureWarning):
            dummy_position_deprecated("A", "B", c="C")

        with pytest.warns(FutureWarning):
            dummy_position_deprecated("A", "B", "C")

    def test_deprecate_arguments(self):
        """Test warnings are triggered when using deprecated arguments."""
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
            dummy_c_deprecated("A")
            dummy_c_deprecated("A", "B")
            dummy_c_deprecated("A", b="B")

            dummy_b_c_deprecated("A")

        with pytest.warns(FutureWarning):
            dummy_c_deprecated("A", "B", "C")

        with pytest.warns(FutureWarning):
            dummy_c_deprecated("A", c="C")

        with pytest.warns(FutureWarning):
            dummy_c_deprecated("A", b="B", c="C")

        with pytest.warns(FutureWarning):
            dummy_b_c_deprecated("A", b="B")

        with pytest.warns(FutureWarning):
            dummy_b_c_deprecated("A", c="C")

        with pytest.warns(FutureWarning):
            dummy_b_c_deprecated("A", b="B", c="C")
