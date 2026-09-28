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


def test_single_quoted_values_are_literal():
    # Single-quoted values are kept verbatim: escape sequences such as "\n" and "\t"
    # are NOT expanded (unlike double-quoted values).
    data = r"""
    NEWLINE='line1\nline2'
    TAB='a\tb'
    ESCAPED_QUOTE='a\"b'
    """
    assert load_dotenv(data) == {
        "NEWLINE": r"line1\nline2",
        "TAB": r"a\tb",
        "ESCAPED_QUOTE": r"a\"b",
    }
    assert load_dotenv(r'DQ="line1\nline2"') == {"DQ": "line1\nline2"}


def test_hash_in_unquoted_value_is_kept():
    # A "#" only starts an inline comment when preceded by whitespace. A "#" that is part of an
    # unquoted value (e.g. in a password, token or URL fragment) must be preserved, not truncated.
    data = """
    PASSWORD=p@ss#word
    TOKEN=abc#123
    URL=http://example.com/x#frag
    LEADING=#notacomment
    COMMENTED=value  # actual comment
    """
    assert load_dotenv(data) == {
        "PASSWORD": "p@ss#word",
        "TOKEN": "abc#123",
        "URL": "http://example.com/x#frag",
        "LEADING": "#notacomment",
        "COMMENTED": "value",
    }


def test_empty_value_with_inline_comment():
    # Whitespace right after "=" also separates an inline comment: the value is empty, not the
    # comment text. Otherwise a comment would leak into env vars/secrets (e.g. `hf jobs --env-file`).
    data = """
    EMPTY= # comment
    EMPTY_MULTI_SPACE=   # comment
    EMPTY_NO_COMMENT=
    LEADING=#notacomment
    """
    assert load_dotenv(data) == {
        "EMPTY": "",
        "EMPTY_MULTI_SPACE": "",
        "EMPTY_NO_COMMENT": "",
        "LEADING": "#notacomment",
    }


def test_comment_attached_to_closing_quote():
    # After a closing quote, a "#" starts a comment even without preceding whitespace.
    data = """
    DQ="value"# comment
    SQ='value'#comment
    SPACED="value" # comment
    HASH_INSIDE="a#b"
    """
    assert load_dotenv(data) == {
        "DQ": "value",
        "SQ": "value",
        "SPACED": "value",
        "HASH_INSIDE": "a#b",
    }


def test_bare_key_with_inline_comment():
    # A bare key (no "=") is resolved from the environment. A trailing comment must not prevent the
    # line from matching, otherwise the key is silently dropped by `--env-file` / `--secrets-file`.
    data = """
    BARE # comment
    BARE_NO_SPACE#comment
    BARE_PLAIN
    """
    environ = {"BARE": "1", "BARE_NO_SPACE": "2", "BARE_PLAIN": "3"}
    assert load_dotenv(data, environ=environ) == {"BARE": "1", "BARE_NO_SPACE": "2", "BARE_PLAIN": "3"}


def test_invalid_line_does_not_import_from_environ():
    # A key followed by arbitrary text is not a valid line: it must be ignored rather than treated
    # as a bare key, which would pull the host value in and clobber an explicit assignment above.
    data = """
    SECRET=explicit_value
    SECRET is documented above
    OTHER not an assignment
    """
    assert load_dotenv(data, environ={"SECRET": "HOST_ENV", "OTHER": "HOST_ENV"}) == {"SECRET": "explicit_value"}
