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


def test_multiline_quoted_value():
    # A quoted value may span several lines (PEM keys, JSON blobs). This used to stop at the
    # first newline and keep the opening quote, so the key came out as '"-----BEGIN...' and the
    # base64 body line was parsed as a separate entry of its own.
    data = 'KEY="-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQ==\n-----END PRIVATE KEY-----"\nOTHER=fine'
    assert load_dotenv(data) == {
        "KEY": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQ==\n-----END PRIVATE KEY-----",
        "OTHER": "fine",
    }


def test_multiline_single_quoted_value():
    # Single quotes span lines too, and a '#' inside the value is not an inline comment.
    data = 'CONFIG=\'{\n  "a": 1,  # not a comment\n  "b": 2\n}\'\nNEXT=ok'
    assert load_dotenv(data) == {"CONFIG": '{\n  "a": 1,  # not a comment\n  "b": 2\n}', "NEXT": "ok"}


def test_multiline_value_with_trailing_comment():
    data = 'A="line1\nline2"  # trailing comment\nB=2'
    assert load_dotenv(data) == {"A": "line1\nline2", "B": "2"}


def test_unterminated_quote_keeps_line_by_line_parsing():
    # A quote that is never closed must not swallow the rest of the file.
    data = 'A="oops\nB=2'
    assert load_dotenv(data) == {"A": '"oops', "B": "2"}
