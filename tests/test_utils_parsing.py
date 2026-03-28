import time

import pytest

from huggingface_hub.utils._parsing import format_timesince, parse_duration, parse_size


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("10", 10),
        ("10k", 10_000),
        ("5M", 5_000_000),
        ("2G", 2_000_000_000),
        ("1T", 1_000_000_000_000),
        ("0", 0),
    ],
)
def test_parse_size_valid(value, expected):
    assert parse_size(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "1.5G",
        "3KB",
        "-5M",
        "10X",
        "abc",
        "",
        "123abc456",
        " 10 K",
    ],
)
def test_parse_size_invalid(value):
    with pytest.raises(ValueError):
        parse_size(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("10s", 10),
        ("5m", 300),
        ("2h", 7_200),
        ("1d", 86_400),
        ("1w", 604_800),
        ("1mo", 2_592_000),
        ("1y", 31_536_000),
        ("0", 0),
    ],
)
def test_parse_duration_valid(value, expected):
    assert parse_duration(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "1.5h",
        "3month",
        "-5m",
        "10X",
        "abc",
        "",
        "123abc456",
        " 10 m",
    ],
)
def test_parse_duration_invalid(value):
    with pytest.raises(ValueError):
        parse_duration(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, "a few seconds ago"),
        (15, "a few seconds ago"),
        (25, "25 seconds ago"),
        (80, "1 minute ago"),
        (1000, "17 minutes ago"),
        (4000, "1 hour ago"),
        (8000, "2 hours ago"),
    ],
)
def test_format_timesince(value, expected):
    assert format_timesince(time.time() - value) == expected
