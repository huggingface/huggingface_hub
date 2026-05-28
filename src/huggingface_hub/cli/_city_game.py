"""Interactive isometric city explorer — easter egg for `hf repos ls --explore`."""

from __future__ import annotations

import dataclasses
import os
import random
import select
import shutil
import sys
import time

from huggingface_hub.hf_api import RepoStorageInfo

from ._city_view import (
    _DX,
    _DY,
    CityData,
    Color,
    TileInfo,
    _build_summary,
    _fill_poly,
    _pixels_to_lines,
    _strip_ansi,
    _visible_len,
    prepare_city_data,
    render_base_buffer,
)
from ._file_listing import format_size


# Cursor sprite — pixel art arrow pointer
_OUTLINE: Color = (30, 30, 30)
_FILL: Color = (255, 255, 255)

_CURSOR_GRID = [
    "  X  ",
    " XWX ",
    "XWWWX",
    " XWX ",
    "  X  ",
]
_CURSOR_PALETTE: dict[str, Color] = {
    "X": _OUTLINE,
    "W": _FILL,
}
_CURSOR_H = len(_CURSOR_GRID)


def _build_cursor() -> list[tuple[int, int, Color]]:
    pixels: list[tuple[int, int, Color]] = []
    for ri, row in enumerate(_CURSOR_GRID):
        for ci, ch in enumerate(row):
            if ch in _CURSOR_PALETTE:
                pixels.append((ci - len(row) // 2, ri - _CURSOR_H + 1, _CURSOR_PALETTE[ch]))
    return pixels


_CURSOR_PIXELS = _build_cursor()

_MOVE_FRAMES = 8
_MOVE_DELAY = 0.03
_CURSOR_PAD = _CURSOR_H + 16
_GAP = 3
_MIN_TERM_W = 100
_MIN_TERM_H = 24
_SUMMARY_W = 24


def run_city_game(repos: list[RepoStorageInfo]) -> None:
    """Launch the interactive city explorer."""
    if not repos:
        print("No repositories found.")
        return

    try:
        import termios
        import tty
    except ImportError:
        print("Interactive mode requires a Unix-like terminal (Linux/macOS).")
        return

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print("Interactive mode requires a terminal.")
        return

    term = shutil.get_terminal_size()
    if term.columns < _MIN_TERM_W or term.lines < _MIN_TERM_H:
        print(f"Your terminal is {term.columns}×{term.lines} characters.")
        print(f"Please resize to at least {_MIN_TERM_W}×{_MIN_TERM_H} to explore the city!")
        return

    city = prepare_city_data(repos)

    tiles_with_repos = [t for t in city.tiles if t.repo is not None]
    start_tile = random.choice(tiles_with_repos) if tiles_with_repos else city.tiles[0]

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        sys.stdout.write("\033[?1049h\033[?25l\033[2J")
        sys.stdout.flush()
        _game_loop(city, start_tile.grid_row, start_tile.grid_col)
    finally:
        sys.stdout.write("\033[?25h\033[?1049l")
        sys.stdout.flush()
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _game_loop(city: CityData, cur_row: int, cur_col: int) -> None:
    """Main game loop."""
    tile_map: dict[tuple[int, int], TileInfo] = {(t.grid_row, t.grid_col): t for t in city.tiles}
    city = dataclasses.replace(city, buf_h=city.buf_h + _CURSOR_PAD, y_off=city.y_off + _CURSOR_PAD)
    base_buf = render_base_buffer(city, tile_grid_only=True)

    summary = _build_summary(city.all_repos, city.total_storage, city.extra_count, no_color=False)

    # Intro: cursor drops onto starting tile
    tx, ty = _tile_top_center(city, cur_row, cur_col, tile_map)
    for i in range(1, _MOVE_FRAMES + 1):
        t = i / _MOVE_FRAMES
        t = t * t * (3 - 2 * t)
        drop_y = ty - 16 * (1 - t)
        frame = _copy_buf(base_buf)
        _highlight_tile(frame, city, tile_map[(cur_row, cur_col)])
        _draw_cursor(frame, round(tx), round(drop_y))
        _present(city, frame, tile_map.get((cur_row, cur_col)), summary)
        time.sleep(_MOVE_DELAY)

    while True:
        cx, cy = _tile_top_center(city, cur_row, cur_col, tile_map)
        frame = _copy_buf(base_buf)
        _highlight_tile(frame, city, tile_map[(cur_row, cur_col)])
        _draw_cursor(frame, round(cx), round(cy))
        _present(city, frame, tile_map.get((cur_row, cur_col)), summary)

        key = _read_key()
        if key in ("q", "Q", "esc", "\x03"):
            return

        dr, dc = _key_to_direction(key)
        if dr == 0 and dc == 0:
            continue

        nr, nc = cur_row + dr, cur_col + dc
        if (nr, nc) not in tile_map:
            continue

        sx, sy = cx, cy
        ex, ey = _tile_top_center(city, nr, nc, tile_map)
        for i in range(1, _MOVE_FRAMES + 1):
            t = i / _MOVE_FRAMES
            t = t * t * (3 - 2 * t)
            bx = sx + (ex - sx) * t
            by = sy + (ey - sy) * t
            frame = _copy_buf(base_buf)
            _highlight_tile(frame, city, tile_map[(nr, nc)])
            _draw_cursor(frame, round(bx), round(by))
            _present(city, frame, tile_map.get((nr, nc)), summary)
            time.sleep(_MOVE_DELAY)

        cur_row, cur_col = nr, nc


def _tile_top_center(
    city: CityData, row: int, col: int, tile_map: dict[tuple[int, int], TileInfo]
) -> tuple[float, float]:
    tile = tile_map.get((row, col))
    h = tile.height if tile else 1
    cx = city.x_off + (col - row) * _DX
    cy = city.y_off + (col + row) * _DY
    return float(cx), float(cy + _DY - h)


def _key_to_direction(key: str) -> tuple[int, int]:
    match key:
        case "w" | "W" | "\x1b[A":
            return -1, 0
        case "s" | "S" | "\x1b[B":
            return 1, 0
        case "a" | "A" | "\x1b[D":
            return 0, -1
        case "d" | "D" | "\x1b[C":
            return 0, 1
        case _:
            return 0, 0


def _draw_cursor(buf: list[list[Color | None]], cx: int, cy: int) -> None:
    """Draw the cursor sprite at (cx, cy)."""
    bh = len(buf)
    bw = len(buf[0]) if buf else 0
    for dx, dy, color in _CURSOR_PIXELS:
        px, py = cx + dx, cy + dy
        if 0 <= py < bh and 0 <= px < bw:
            buf[py][px] = color


def _highlight_tile(buf: list[list[Color | None]], city: CityData, tile: TileInfo) -> None:
    cx = city.x_off + (tile.grid_col - tile.grid_row) * _DX
    cy = city.y_off + (tile.grid_col + tile.grid_row) * _DY
    h = tile.height
    _fill_poly(
        buf,
        [(cx, cy - h), (cx + _DX, cy + _DY - h), (cx, cy + 2 * _DY - h), (cx - _DX, cy + _DY - h)],
        _brighten(tile.top, 35),
    )


def _brighten(color: Color, amount: int) -> Color:
    return (min(255, color[0] + amount), min(255, color[1] + amount), min(255, color[2] + amount))


def _present(
    city: CityData,
    buf: list[list[Color | None]],
    tile: TileInfo | None,
    summary: list[str],
) -> None:
    city_lines = _pixels_to_lines(buf, no_color=False)
    while city_lines and not _strip_ansi(city_lines[0]).strip():
        city_lines.pop(0)
    while city_lines and not _strip_ansi(city_lines[-1]).strip():
        city_lines.pop()

    city_w = max((_visible_len(line) for line in city_lines), default=0)
    term = shutil.get_terminal_size()
    panel_max_w = max(20, term.columns - city_w - _SUMMARY_W - 2 * _GAP)

    info = _build_info_panel(tile, city, panel_max_w)

    # Three-panel layout: summary (left) | city (center) | details (right)
    n = max(len(summary), len(city_lines), len(info))
    summary_lo = max(0, (n - len(summary)) // 2)
    info_lo = max(0, (n - len(info)) // 2)

    lines: list[str] = []
    for i in range(n):
        # Left: summary panel
        si = i - summary_lo
        lt = summary[si] if 0 <= si < len(summary) else ""
        lpad = max(0, _SUMMARY_W - _visible_len(lt))

        # Center: city
        ct = city_lines[i] if i < len(city_lines) else ""
        cpad = max(0, city_w - _visible_len(ct))

        # Right: info panel
        ri = i - info_lo
        rt = info[ri] if 0 <= ri < len(info) else ""

        lines.append(lt + " " * lpad + " " * _GAP + ct + " " * cpad + " " * _GAP + rt)

    lines.append("")
    lines.append("  \033[90mWASD/Arrows: move · Q/ESC: quit\033[0m")

    while len(lines) < term.lines - 1:
        lines.append("")

    output = "\033[H"
    for line in lines[: term.lines - 1]:
        output += line + "\033[K\r\n"
    sys.stdout.write(output)
    sys.stdout.flush()


def _build_info_panel(tile: TileInfo | None, city: CityData, max_w: int) -> list[str]:
    reset = "\033[0m"
    gray = "\033[90m"
    bold = "\033[1m"
    indent = "  "
    content_w = max_w - len(indent)

    lines: list[str] = [""]
    lines.append(f"{indent}{bold}City Explorer{reset}")
    lines.append(indent + "─" * min(22, content_w))
    lines.append("")

    if tile is None:
        lines.append(f"{indent}{gray}Move to a tile")
        lines.append(f"{indent}to see details.{reset}")
        return lines

    if tile.repo is None:
        lines.append(f"{indent}{gray}+{city.extra_count} more repos{reset}")
        lines.append(f"{indent}{gray}{format_size(city.extra_storage, human_readable=True)} combined{reset}")
        return lines

    repo = tile.repo
    name = repo.id
    if len(name) > content_w:
        name = name[: content_w - 3] + "..."
    lines.append(f"{indent}{bold}{name}{reset}")
    lines.append("")

    type_ansi = {
        "model": "\033[38;2;175;148;240m",
        "dataset": "\033[38;2;245;128;128m",
        "space": "\033[38;2;245;175;85m",
        "bucket": "\033[38;2;112;185;242m",
    }
    tc = type_ansi.get(repo.type, "")

    lines.append(f"{indent}Type       {tc}{repo.type}{reset}")
    lines.append(f"{indent}Visibility {repo.visibility}")
    lines.append(f"{indent}Storage    {format_size(repo.storage, human_readable=True)}")
    lines.append(f"{indent}Usage      {repo.storage_percent:.1f}%")
    lines.append("")

    bar_w = min(18, content_w)
    filled = max(0, min(bar_w, round(repo.storage_percent / 100 * bar_w)))
    lines.append(f"{indent}{tc}{'█' * filled}{gray}{'░' * (bar_w - filled)}{reset}")

    return lines


def _copy_buf(buf: list[list[Color | None]]) -> list[list[Color | None]]:
    return [row[:] for row in buf]


def _read_key() -> str:
    fd = sys.stdin.fileno()
    ch = os.read(fd, 1)
    if ch == b"\x1b":
        if _has_input(fd, 0.05):
            ch2 = os.read(fd, 1)
            if ch2 == b"[" and _has_input(fd, 0.05):
                ch3 = os.read(fd, 1)
                return f"\x1b[{ch3.decode()}"
        return "esc"
    return ch.decode("utf-8", errors="replace")


def _has_input(fd: int, timeout: float) -> bool:
    r, _, _ = select.select([fd], [], [], timeout)
    return bool(r)
