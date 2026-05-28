"""3D isometric city view for repository storage visualization.

Renders an isometric city where each tile represents a repository.
Tile height is proportional to storage usage, color indicates repo type.
Uses a pixel buffer with half-block character rendering for smooth output.
"""

from __future__ import annotations

import dataclasses
import math
import os
import re

from huggingface_hub.hf_api import RepoStorageInfo

from ._file_listing import format_size


Color = tuple[int, int, int]

# (top_face, left_face, right_face) — lighter to darker for 3D effect
_TYPE_COLORS: dict[str, tuple[Color, Color, Color]] = {
    "model": ((175, 148, 240), (138, 112, 208), (105, 80, 180)),
    "dataset": ((245, 128, 128), (222, 92, 92), (190, 60, 60)),
    "space": ((245, 175, 85), (218, 140, 55), (185, 110, 30)),
    "bucket": ((112, 185, 242), (70, 150, 220), (40, 118, 192)),
}
_EXTRA_COLORS: tuple[Color, Color, Color] = ((168, 176, 188), (128, 136, 148), (90, 98, 110))
_GRID_COLOR: Color = (178, 182, 190)

_DX = 4  # isometric half-width (pixels)
_DY = 2  # isometric half-height (pixels)
_MAX_H = 16  # tallest tile (pixels)
_MIN_H = 1
_COLS = 6
_EXT = 1  # grid extension beyond tiles
_SUMMARY_W = 24
_GAP = 2
_MAX_TILES = 30


@dataclasses.dataclass
class TileInfo:
    grid_row: int
    grid_col: int
    height: int
    top: Color
    left: Color
    right: Color
    repo: RepoStorageInfo | None


@dataclasses.dataclass
class CityData:
    tiles: list[TileInfo]
    rows: int
    cols: int
    x_off: int
    y_off: int
    buf_w: int
    buf_h: int
    total_storage: int
    extra_count: int
    extra_storage: int
    all_repos: list[RepoStorageInfo]


def prepare_city_data(repos: list[RepoStorageInfo]) -> CityData:
    """Prepare all city layout data from a list of repos."""
    sorted_repos = sorted(repos, key=lambda r: r.storage, reverse=True)
    display = sorted_repos[:_MAX_TILES]
    extra_count = max(0, len(sorted_repos) - _MAX_TILES)
    extra_storage = sum(r.storage for r in sorted_repos[_MAX_TILES:])
    total_storage = sum(r.storage for r in repos)
    max_storage = max(1, display[0].storage)

    n = len(display) + (1 if extra_count > 0 else 0)
    cols = min(n, _COLS)
    rows = math.ceil(n / cols) if cols > 0 else 1

    tiles: list[TileInfo] = []
    for i, repo in enumerate(display):
        r, c = divmod(i, cols)
        h = max(_MIN_H, round(math.sqrt(repo.storage / max_storage) * _MAX_H))
        top, left, right = _TYPE_COLORS.get(repo.type, _EXTRA_COLORS)
        tiles.append(TileInfo(r, c, h, top, left, right, repo))
    if extra_count > 0:
        r, c = divmod(len(display), cols)
        h = max(_MIN_H, round(math.sqrt(extra_storage / max_storage) * _MAX_H))
        tiles.append(TileInfo(r, c, h, *_EXTRA_COLORS, None))

    r_lo, r_hi = -_EXT, rows - 1 + _EXT
    c_lo, c_hi = -_EXT, cols - 1 + _EXT

    xs: list[int] = []
    ys: list[int] = []
    for rr in range(r_lo, r_hi + 1):
        for cc in range(c_lo, c_hi + 1):
            cx, cy = (cc - rr) * _DX, (cc + rr) * _DY
            xs.extend([cx - _DX, cx + _DX])
            ys.extend([cy, cy + 2 * _DY])
    for tile in tiles:
        ys.append((tile.grid_col + tile.grid_row) * _DY - tile.height)

    x_off = -min(xs)
    y_off = -min(ys)
    buf_w = max(xs) - min(xs) + 1
    buf_h = max(ys) - min(ys) + 1
    if buf_h % 2:
        buf_h += 1

    return CityData(
        tiles=tiles,
        rows=rows,
        cols=cols,
        x_off=x_off,
        y_off=y_off,
        buf_w=buf_w,
        buf_h=buf_h,
        total_storage=total_storage,
        extra_count=extra_count,
        extra_storage=extra_storage,
        all_repos=repos,
    )


def render_base_buffer(city: CityData, *, tile_grid_only: bool = False) -> list[list[Color | None]]:
    """Render the city grid and tiles into a fresh pixel buffer."""
    buf: list[list[Color | None]] = [[None] * city.buf_w for _ in range(city.buf_h)]

    if tile_grid_only:
        for tile in city.tiles:
            cx = city.x_off + (tile.grid_col - tile.grid_row) * _DX
            cy = city.y_off + (tile.grid_col + tile.grid_row) * _DY
            _draw_diamond_outline(buf, cx, cy)
    else:
        r_lo, r_hi = -_EXT, city.rows - 1 + _EXT
        c_lo, c_hi = -_EXT, city.cols - 1 + _EXT
        for rr in range(r_lo, r_hi + 1):
            for cc in range(c_lo, c_hi + 1):
                cx = city.x_off + (cc - rr) * _DX
                cy = city.y_off + (cc + rr) * _DY
                _draw_diamond_outline(buf, cx, cy)

    sorted_tiles = sorted(city.tiles, key=lambda t: (t.grid_row + t.grid_col, t.grid_col))
    for tile in sorted_tiles:
        cx = city.x_off + (tile.grid_col - tile.grid_row) * _DX
        cy = city.y_off + (tile.grid_col + tile.grid_row) * _DY
        _draw_block(buf, cx, cy, tile.height, tile.top, tile.left, tile.right)

    return buf


def render_city_view(repos: list[RepoStorageInfo]) -> str:
    if not repos:
        return "No repositories found."

    no_color = bool(os.environ.get("NO_COLOR"))
    city = prepare_city_data(repos)
    buf = render_base_buffer(city)
    city_lines = _pixels_to_lines(buf, no_color)

    while city_lines and not _strip_ansi(city_lines[0]).strip():
        city_lines.pop(0)
    while city_lines and not _strip_ansi(city_lines[-1]).strip():
        city_lines.pop()

    summary = _build_summary(city.all_repos, city.total_storage, city.extra_count, no_color)
    return _merge(summary, city_lines)


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------


def _draw_diamond_outline(buf: list[list[Color | None]], cx: int, cy: int) -> None:
    t = (cx, cy)
    r = (cx + _DX, cy + _DY)
    b = (cx, cy + 2 * _DY)
    ll = (cx - _DX, cy + _DY)
    _draw_line(buf, *t, *r, _GRID_COLOR)
    _draw_line(buf, *r, *b, _GRID_COLOR)
    _draw_line(buf, *b, *ll, _GRID_COLOR)
    _draw_line(buf, *ll, *t, _GRID_COLOR)


def _draw_block(
    buf: list[list[Color | None]],
    cx: int,
    cy: int,
    h: int,
    top: Color,
    left: Color,
    right: Color,
) -> None:
    _fill_poly(
        buf,
        [(cx - _DX, cy + _DY - h), (cx, cy + 2 * _DY - h), (cx, cy + 2 * _DY), (cx - _DX, cy + _DY)],
        left,
    )
    _fill_poly(
        buf,
        [(cx, cy + 2 * _DY - h), (cx + _DX, cy + _DY - h), (cx + _DX, cy + _DY), (cx, cy + 2 * _DY)],
        right,
    )
    _fill_poly(
        buf,
        [(cx, cy - h), (cx + _DX, cy + _DY - h), (cx, cy + 2 * _DY - h), (cx - _DX, cy + _DY - h)],
        top,
    )


def _fill_poly(buf: list[list[Color | None]], verts: list[tuple[int, int]], color: Color) -> None:
    bh = len(buf)
    bw = len(buf[0]) if buf else 0
    all_y = [v[1] for v in verts]
    y0 = max(0, min(all_y))
    y1 = min(bh - 1, max(all_y))
    n = len(verts)
    for y in range(y0, y1 + 1):
        xl: float = float("inf")
        xr: float = float("-inf")
        for i in range(n):
            ax, ay = verts[i]
            bx, by = verts[(i + 1) % n]
            if ay == by:
                if y == ay:
                    xl = min(xl, float(min(ax, bx)))
                    xr = max(xr, float(max(ax, bx)))
                continue
            if not (min(ay, by) <= y <= max(ay, by)):
                continue
            t = (y - ay) / (by - ay)
            ix = ax + t * (bx - ax)
            xl = min(xl, ix)
            xr = max(xr, ix)
        if xl <= xr:
            for x in range(max(0, round(xl)), min(bw, round(xr) + 1)):
                buf[y][x] = color


def _draw_line(buf: list[list[Color | None]], x0: int, y0: int, x1: int, y1: int, color: Color) -> None:
    bh = len(buf)
    bw = len(buf[0]) if buf else 0
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    steps = max(dx, dy)
    if steps == 0:
        if 0 <= y0 < bh and 0 <= x0 < bw:
            buf[y0][x0] = color
        return
    xi = (x1 - x0) / steps
    yi = (y1 - y0) / steps
    fx, fy = float(x0), float(y0)
    for _ in range(steps + 1):
        px, py = round(fx), round(fy)
        if 0 <= py < bh and 0 <= px < bw:
            buf[py][px] = color
        fx += xi
        fy += yi


# ---------------------------------------------------------------------------
# Pixel buffer → terminal
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


def _visible_len(s: str) -> int:
    return len(_strip_ansi(s))


def _pixels_to_lines(buf: list[list[Color | None]], no_color: bool) -> list[str]:
    height = len(buf)
    width = len(buf[0]) if buf else 0
    lines: list[str] = []
    for row in range(0, height, 2):
        last = -1
        for col in range(width - 1, -1, -1):
            top = buf[row][col]
            bot = buf[row + 1][col] if row + 1 < height else None
            if top or bot:
                last = col
                break
        if last < 0:
            lines.append("")
            continue

        parts: list[str] = []
        cfg: Color | None = None
        cbg: Color | None = None

        for col in range(last + 1):
            top = buf[row][col]
            bot = buf[row + 1][col] if row + 1 < height else None

            if no_color:
                parts.append("█" if top and bot else "▀" if top else "▄" if bot else " ")
                continue

            if not top and not bot:
                if cfg is not None or cbg is not None:
                    parts.append("\033[0m")
                    cfg = cbg = None
                parts.append(" ")
                continue

            if top and bot and top == bot:
                nfg, nbg, ch = top, None, "█"
            elif top and bot:
                nfg, nbg, ch = bot, top, "▄"
            elif top:
                nfg, nbg, ch = top, None, "▀"
            else:
                nfg, nbg, ch = bot, None, "▄"

            esc = ""
            if nfg != cfg:
                esc += f"\033[38;2;{nfg[0]};{nfg[1]};{nfg[2]}m"
                cfg = nfg
            if nbg != cbg:
                esc += "\033[49m" if nbg is None else f"\033[48;2;{nbg[0]};{nbg[1]};{nbg[2]}m"
                cbg = nbg
            parts.append(esc + ch)

        if cfg is not None or cbg is not None:
            parts.append("\033[0m")
        lines.append("".join(parts))
    return lines


# ---------------------------------------------------------------------------
# Summary panel
# ---------------------------------------------------------------------------


def _colored_square(color: Color, no_color: bool) -> str:
    if no_color:
        return "■"
    return f"\033[38;2;{color[0]};{color[1]};{color[2]}m■\033[0m"


def _build_summary(
    repos: list[RepoStorageInfo],
    total_storage: int,
    extra_count: int,
    no_color: bool,
) -> list[str]:
    lines: list[str] = [""]
    lines.append("  Storage Overview")
    lines.append("  " + "─" * 16)
    lines.append(f"  {format_size(total_storage, human_readable=True)} total")
    lines.append("")

    order = ["model", "dataset", "space", "bucket"]
    labels = {"model": "Models", "dataset": "Datasets", "space": "Spaces", "bucket": "Buckets"}
    for rtype in order:
        group = [r for r in repos if r.type == rtype]
        if not group:
            continue
        storage = sum(r.storage for r in group)
        sq = _colored_square(_TYPE_COLORS[rtype][0], no_color)
        lines.append(f"  {sq} {labels[rtype]}")
        lines.append(f"    {len(group)} repos · {format_size(storage, human_readable=True)}")
        lines.append("")

    if extra_count > 0:
        sq = _colored_square(_EXTRA_COLORS[0], no_color)
        lines.append(f"  {sq} +{extra_count} more repos")

    return lines


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def _merge(left: list[str], right: list[str]) -> str:
    n = max(len(left), len(right))
    lo = max(0, (n - len(left)) // 2)
    out: list[str] = []
    for i in range(n):
        li = i - lo
        lt = left[li] if 0 <= li < len(left) else ""
        rt = right[i] if i < len(right) else ""
        pad = max(0, _SUMMARY_W - _visible_len(lt))
        out.append(lt + " " * pad + " " * _GAP + rt)
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out)
