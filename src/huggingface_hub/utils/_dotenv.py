# AI-generated module (ChatGPT)
import re


# Escape sequences expanded inside quoted values. Double-quoted values additionally
# expand "\$" to "$"; single-quoted values keep it verbatim.
_ESCAPES = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}
_DOUBLE_QUOTE_ESCAPES = {**_ESCAPES, "$": "$"}


def _unescape(value: str, escapes: dict[str, str]) -> str:
    r"""Expand backslash escapes in a single left-to-right pass.

    Processing in one pass (rather than chained `str.replace` calls) ensures an escaped
    backslash (`\\`) is consumed as a unit and cannot merge with the following character,
    e.g. `\\n` is a backslash followed by `n`, not a newline. Unknown escapes are kept as-is.
    """
    return re.sub(r"\\(.)", lambda match: escapes.get(match.group(1), match.group(0)), value)


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
        (?:
            [^\S\n]*
            (=)                           # equal sign
            (?:
                [^\S\n]*
                (                         # quoted value
                    '(?:\\'|[^'])*'       # single-quoted
                    | \"(?:\\\"|[^\"])*\" # double-quoted
                )
                [^\S\n]*(?:\#[^\n\r]*)?   # inline comment (needs no preceding whitespace after a quote)
                |
                (?:[^\S\n]+(?!\#))?       # whitespace after '=' also separates an inline comment
                (                         # unquoted value (may contain '#')
                    [^\n\r]*?
                )
                (?:[^\S\n]+\#[^\n\r]*)?   # inline comment (must be preceded by whitespace)
            )
            |
            [^\S\n]*(?:\#[^\n\r]*)?       # bare key (no '='), with an optional inline comment
        )$
    """,
        re.VERBOSE,
    )

    for line in dotenv_str.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue  # Skip comments and empty lines

        match = line_pattern.match(line)
        if match:
            key = match.group(1)
            val = None
            if match.group(2):  # if there is '='
                raw_val = match.group(3) or match.group(4) or ""
                val = raw_val.strip()
                # Remove surrounding quotes if quoted
                if val.startswith('"') and val.endswith('"'):
                    # Double-quoted values expand escape sequences (\n, \t, \", \\, \$).
                    val = _unescape(val[1:-1], _DOUBLE_QUOTE_ESCAPES)
                elif val.startswith("'") and val.endswith("'"):
                    # Single-quoted values are kept verbatim: no escape expansion.
                    val = val[1:-1]
            elif environ is not None:
                # Get it from the current environment
                val = environ.get(key)

            if val is not None:
                env[key] = val

    return env
