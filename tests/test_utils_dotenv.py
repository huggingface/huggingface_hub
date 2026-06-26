# AI-generated module (ChatGPT)
from huggingface_hub.utils._dotenv import load_dotenv


def test_basic_key_value():
    data = "KEY=value"
    assert load_dotenv(data) == {"KEY": "value"}


def test_whitespace_and_comments():
    data = """
    # This is a comment
    KEY = value    # inline comment
    EMPTY=
    """
    assert load_dotenv(data) == {"KEY": "value", "EMPTY": ""}


def test_quoted_values():
    data = """
    SINGLE='single quoted'
    DOUBLE="double quoted"
    ESCAPED="line\\nbreak"
    """
    assert load_dotenv(data) == {"SINGLE": "single quoted", "DOUBLE": "double quoted", "ESCAPED": "line\nbreak"}


def test_export_and_inline_comment():
    data = "export KEY=value # this is a comment"
    assert load_dotenv(data) == {"KEY": "value"}


def test_ignore_invalid_lines():
    data = """
    this is not valid
    KEY=value
    """
    assert load_dotenv(data) == {"KEY": "value"}


def test_complex_quotes():
    data = r"""
    QUOTED="some value with # not comment"
    ESCAPE="escaped \$dollar and \\backslash"
    """
    assert load_dotenv(data) == {
        "QUOTED": "some value with # not comment",
        "ESCAPE": "escaped $dollar and \\backslash",
    }


def test_escaped_backslash_before_escape_char():
    # An escaped backslash ("\\") collapses to a single backslash and the next character
    # stays literal, even when it is "n", "t" or a quote. This used to break: the trailing
    # character got merged into a newline/tab, so a Windows path like "C:\\new" came out as
    # "C:" + backslash + newline + "ew".
    data = r"""
    WIN="C:\\new"
    LITERAL="a\\nb"
    TAB="x\\t"
    """
    assert load_dotenv(data) == {
        "WIN": "C:\\new",
        "LITERAL": "a\\nb",
        "TAB": "x\\t",
    }


def test_no_value():
    data = "NOVALUE="
    assert load_dotenv(data) == {"NOVALUE": ""}


def test_multiple_lines():
    data = """
    A=1
    B="two"
    C='three'
    D=4
    """
    assert load_dotenv(data) == {"A": "1", "B": "two", "C": "three", "D": "4"}


def test_environ():
    data = """
    A=1
    B
    C=3
    MISSING
    EMPTY
    """
    environ = {"A": "one", "B": "two", "D": "four", "EMPTY": ""}
    assert load_dotenv(data, environ=environ) == {"A": "1", "B": "two", "C": "3", "EMPTY": ""}
