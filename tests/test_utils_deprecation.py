import unittest
import warnings

import pytest

from huggingface_hub.utils._deprecation import (
    _deprecate_arguments,
    _deprecate_list_output,
    _deprecate_method,
    _deprecate_positional_args,
)


class TestDeprecationUtils(unittest.TestCase):
    def test_deprecate_positional_args(self):
        """Test warnings are triggered when using deprecated positional args."""

        @_deprecate_positional_args(version="xxx")
        def dummy_position_deprecated(a, *, b="b", c="c"):
            pass

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

        @_deprecate_arguments(version="xxx", deprecated_args={"c"})
        def dummy_c_deprecated(a, b="b", c="c"):
            pass

        @_deprecate_arguments(version="xxx", deprecated_args={"b", "c"})
        def dummy_b_c_deprecated(a, b="b", c="c"):
            pass

        with warnings.catch_warnings():
            # Assert no warnings when used correctly.
            # Taken from https://docs.pytest.org/en/latest/how-to/capture-warnings.html#additional-use-cases-of-warnings-in-tests
            warnings.simplefilter("error")
            dummy_c_deprecated("A")
            dummy_c_deprecated("A", "B")
            dummy_c_deprecated("A", b="B")

            dummy_b_c_deprecated("A")

            dummy_b_c_deprecated("A", b="b")
            dummy_b_c_deprecated("A", b="b", c="c")

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

    def test_deprecate_arguments_with_default_warning_message(self) -> None:
        """Test default warning message when deprecating arguments."""

        @_deprecate_arguments(version="xxx", deprecated_args={"a"})
        def dummy_deprecated_default_message(a: str = "a") -> None:
            pass

        # Default message
        with pytest.warns(FutureWarning) as record:
            dummy_deprecated_default_message(a="A")
        self.assertEqual(len(record), 1)
        self.assertEqual(
            record[0].message.args[0],
            "Deprecated argument(s) used in 'dummy_deprecated_default_message': a."
            " Will not be supported from version 'xxx'.",
        )

    def test_deprecate_arguments_with_custom_warning_message(self) -> None:
        """Test custom warning message when deprecating arguments."""

        @_deprecate_arguments(
            version="xxx",
            deprecated_args={"a"},
            custom_message="This is a custom message.",
        )
        def dummy_deprecated_custom_message(a: str = "a") -> None:
            pass

        # Custom message
        with pytest.warns(FutureWarning) as record:
            dummy_deprecated_custom_message(a="A")
        self.assertEqual(len(record), 1)
        self.assertEqual(
            record[0].message.args[0],
            "Deprecated argument(s) used in 'dummy_deprecated_custom_message': a."
            " Will not be supported from version 'xxx'.\n\nThis is a custom"
            " message.",
        )

    def test_deprecated_method(self) -> None:
        """Test deprecate method throw warning."""

        @_deprecate_method(version="xxx", message="This is a custom message.")
        def dummy_deprecated() -> None:
            pass

        # Custom message
        with pytest.warns(FutureWarning) as record:
            dummy_deprecated()
        self.assertEqual(len(record), 1)
        self.assertEqual(
            record[0].message.args[0],
            "'dummy_deprecated' (from 'tests.test_utils_deprecation') is deprecated"
            " and will be removed from version 'xxx'. This is a custom message.",
        )

    def test_deprecate_list_output(self) -> None:
        """Test test_deprecate_list_output throw warnings."""

        @_deprecate_list_output(version="xxx")
        def dummy_deprecated() -> None:
            return [1, 2, 3]

        output = dummy_deprecated()

        # Still a list !
        self.assertIsInstance(output, list)

        # __getitem__
        with pytest.warns(FutureWarning) as record:
            self.assertEqual(output[0], 1)

        # (check real message once)
        self.assertEqual(
            record[0].message.args[0],
            "'dummy_deprecated' currently returns a list of objects but is planned to be a generator starting from"
            " version xxx in order to implement pagination. Please avoid to use"
            " `dummy_deprecated(...).__getitem__` or explicitly convert the output to a list first with `[item for"
            " item in dummy_deprecated(...)]`.",
        )

        # __setitem__
        with pytest.warns(FutureWarning):
            output[0] = 10

        # __delitem__
        with pytest.warns(FutureWarning):
            del output[1]

        # `.append` deprecated (as .index, .pop, .insert, .remove, .sort,...)
        with pytest.warns(FutureWarning):
            output.append(5)

        # Magic __len__ deprecated (as __add__, __mul__, __contains__,...)
        with pytest.warns(FutureWarning):
            self.assertEqual(len(output), 3)

        with pytest.warns(FutureWarning):
            self.assertTrue(3 in output)

        # List has been modified
        # Iterating over items is NOT deprecated !
        for item, expected in zip(output, [10, 3, 5]):
            self.assertEqual(item, expected)

        # If output is converted to list, no warning is raised.
        output_as_list = list(iter(dummy_deprecated()))
        self.assertEqual(output_as_list[0], 1)
        del output_as_list[1]
        output_as_list.append(5)
        self.assertEqual(len(output_as_list), 3)
        self.assertTrue(3 in output_as_list)
