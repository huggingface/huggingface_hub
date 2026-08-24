import time

import pytest

from huggingface_hub.utils._parsing import format_duration, format_timesince, parse_duration, parse_size


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("10", 10),
        ("10k", 10_000),
        ("5M", 5_000_000),
        ("2G", 2_000_000_000),
        ("1T", 1_000_000_000_000),
        ("3KB", 3_000),
        ("10MB", 10_000_000),
        ("1GB", 1_000_000_000),
        ("2TB", 2_000_000_000_000),
        ("0", 0),
    ],
)
def test_parse_size_valid(value, expected):
    assert parse_size(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "1.5G",
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


@pytest.mark.parametrize(
    ("secs", "expected"),
    [
        (None, "--"),
        (0, "0s"),
        (1, "1s"),
        (45, "45s"),
        (59, "59s"),
        (60, "1m 0s"),
        (61, "1m 1s"),
        (199, "3m 19s"),
        (3599, "59m 59s"),
        (3600, "1h 0m"),
        (8100, "2h 15m"),
    ],
)
def test_format_duration(secs, expected):
    assert format_duration(secs) == expected
