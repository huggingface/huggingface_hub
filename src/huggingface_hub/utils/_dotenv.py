# AI-generated module (ChatGPT)
import re
from collections.abc import Iterator


# Escape sequences expanded inside quoted values. Double-quoted values additionally
# expand "\$" to "$"; single-quoted values keep it verbatim.
_ESCAPES = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}
_DOUBLE_QUOTE_ESCAPES = {**_ESCAPES, "$": "$"}

# A line whose value starts with a quote. Only the opening quote is captured; whether it is
# closed on the same line is decided by `_quote_is_closed`.
_QUOTED_VALUE_START = re.compile(
    r"""
    ^[^\S\n]*
    (?:export[^\S\n]+)?               # optional export
    [A-Za-z_][A-Za-z0-9_]*            # key
    [^\S\n]*=[^\S\n]*
    (['"])                            # opening quote
""",
    re.VERBOSE,
)


def _unescape(value: str, escapes: dict[str, str]) -> str:
    r"""Expand backslash escapes in a single left-to-right pass.

    Processing in one pass (rather than chained `str.replace` calls) ensures an escaped
    backslash (`\\`) is consumed as a unit and cannot merge with the following character,
    e.g. `\\n` is a backslash followed by `n`, not a newline. Unknown escapes are kept as-is.
    """
    return re.sub(r"\\(.)", lambda match: escapes.get(match.group(1), match.group(0)), value)


def _quote_is_closed(text: str, quote: str, start: int) -> bool:
    """Whether the quote opened at `start - 1` is closed within `text`.

    Backslash escapes are skipped as a unit so that an escaped quote does not end the value,
    matching how `_unescape` reads them afterwards.
    """
    index = start
    while index < len(text):
        if text[index] == "\\":
            index += 2
            continue
        if text[index] == quote:
            return True
        index += 1
    return False


def _logical_lines(dotenv_str: str) -> Iterator[str]:
    r"""Split into logical lines, joining physical lines while a quoted value stays open.

    A quoted value may span several lines — a PEM key or a JSON blob being the usual cases.
    Splitting on newlines alone would end the value at the first newline while leaving the
    opening quote in it, so `KEY="-----BEGIN X-----\n...\n-----END X-----"` would yield
    `'"-----BEGIN X-----'` and then parse the remaining lines as separate entries.

    A quote that is never closed is left alone: those lines are yielded one by one, so
    malformed input degrades the same way it did before rather than swallowing the rest of
    the file.
    """
    lines = dotenv_str.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1

        match = _QUOTED_VALUE_START.match(line)
        if match is None:
            yield line
            continue

        quote, value_start = match.group(1), match.end(1)
        joined, lookahead = line, index
        while not _quote_is_closed(joined, quote, value_start) and lookahead < len(lines):
            joined += "\n" + lines[lookahead]
            lookahead += 1

        if _quote_is_closed(joined, quote, value_start):
            index = lookahead
            yield joined
        else:
            yield line


def load_dotenv(dotenv_str: str, environ: dict[str, str] | None = None) -> dict[str, str]:
    """
    Parse a DOTENV-format string and return a dictionary of key-value pairs.
    Handles quoted values, comments, export keyword, and blank lines.
    """
    env: dict[str, str] = {}
    line_pattern = re.compile(
        r"""
        ^\s*
        (?:export[^\S\n]+)?               # optional export
        ([A-Za-z_][A-Za-z0-9_]*)          # key
        [^\S\n]*(=)?[^\S\n]*
        (                                 # value group
            (?:
                '(?:\\'|[^'])*'           # single-quoted value
                | \"(?:\\\"|[^\"])*\"     # double-quoted value
                | [^#\n\r]+?              # unquoted value
            )
        )?
        [^\S\n]*(?:\#.*)?$                # optional inline comment
    """,
        re.VERBOSE,
    )

    for line in _logical_lines(dotenv_str):
        line = line.strip()
        if not line or line.startswith("#"):
            continue  # Skip comments and empty lines

        match = line_pattern.match(line)
        if match:
            key = match.group(1)
            val = None
            if match.group(2):  # if there is '='
                raw_val = match.group(3) or ""
                val = raw_val.strip()
                # Remove surrounding quotes if quoted
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    escapes = _DOUBLE_QUOTE_ESCAPES if raw_val.startswith('"') else _ESCAPES
                    val = _unescape(val[1:-1], escapes)
            elif environ is not None:
                # Get it from the current environment
                val = environ.get(key)

            if val is not None:
                env[key] = val

    return env
