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


def test_hash_without_leading_whitespace_is_part_of_value():
    # An inline comment starts at " #". A "#" that is not preceded by whitespace is an
    # ordinary character, so it must not truncate the value. Secrets routinely contain
    # one, and truncating silently produced a wrong value rather than an error.
    assert load_dotenv("KEY=abc#def") == {"KEY": "abc#def"}
    assert load_dotenv("HF_TOKEN=hf_AbC#123xyz") == {"HF_TOKEN": "hf_AbC#123xyz"}
    assert load_dotenv("PASSWORD=p@ss#word") == {"PASSWORD": "p@ss#word"}


def test_hash_at_start_of_value_is_kept():
    # No whitespace between '=' and '#', so the '#' belongs to the value.
    assert load_dotenv("KEY=#value") == {"KEY": "#value"}


def test_comment_right_after_equals_leaves_value_empty():
    # With whitespace after '=', the '#' opens a comment and the value stays empty.
    # Guards against the comment text itself being passed through as an env/secret value.
    assert load_dotenv("KEY= # comment") == {"KEY": ""}
    assert load_dotenv("KEY=  #comment") == {"KEY": ""}
    assert load_dotenv("SECRET= # note") == {"SECRET": ""}


def test_inline_comment_still_stripped_after_hash_fix():
    # Guards the other direction: " #" must still begin a comment.
    assert load_dotenv("KEY=value # comment") == {"KEY": "value"}
    assert load_dotenv("KEY=value\t# comment") == {"KEY": "value"}
    assert load_dotenv("KEY=a#b # comment") == {"KEY": "a#b"}


def test_unquoted_value_keeps_inner_whitespace():
    assert load_dotenv("KEY=a b c") == {"KEY": "a b c"}
