import os
from typing import Union

def tabulate(rows: list[list[Union[str, int]]], headers: list[str]) -> str:
    """
    Inspired by:

    - stackoverflow.com/a/8356620/593036
    - stackoverflow.com/questions/9535954/printing-lists-as-tabular-data
    """
    col_widths = [max(len(str(x)) for x in col) for col in zip(*rows, headers)]
    terminal_width = max(os.get_terminal_size().columns, len(headers) * 12)
    while len(headers) + sum(col_widths) > terminal_width:
        col_to_minimize = col_widths.index(max(col_widths))
        col_widths[col_to_minimize] //= 2
        if len(headers) + sum(col_widths) <= terminal_width:
            col_widths[col_to_minimize] = terminal_width - sum(col_widths) - len(headers) + col_widths[col_to_minimize]
    row_format = ("{{:{}}} " * len(headers)).format(*col_widths)
    lines = []
    lines.append(row_format.format(*headers))
    lines.append(row_format.format(*["-" * w for w in col_widths]))
    for row in rows:
        row = [x[:col_width - 3] + "..." if len(str(x)) > col_width else x for x, col_width in zip(row, col_widths)]
        lines.append(row_format.format(*row))
    return "\n".join(lines)
