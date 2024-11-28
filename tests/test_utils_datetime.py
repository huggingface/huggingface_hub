import unittest
from datetime import datetime, timezone

import pytest

from huggingface_hub.utils import parse_datetime


class TestDatetimeUtils(unittest.TestCase):
    def test_parse_datetime(self):
        """Test `parse_datetime` works correctly on datetimes returned by server."""
        self.assertEqual(
            parse_datetime("2022-08-19T07:19:38.123Z"),
            datetime(2022, 8, 19, 7, 19, 38, 123000, tzinfo=timezone.utc),
        )

        # Test nanoseconds precision (should be truncated to microseconds)
        self.assertEqual(
            parse_datetime("2022-08-19T07:19:38.123456789Z"),
            datetime(2022, 8, 19, 7, 19, 38, 123456, tzinfo=timezone.utc),
        )

        # Test without milliseconds (should add .000)
        self.assertEqual(
            parse_datetime("2024-11-16T00:27:02Z"),
            datetime(2024, 11, 16, 0, 27, 2, 0, tzinfo=timezone.utc),
        )

        with pytest.raises(ValueError, match=r".*Cannot parse '2022-08-19T07:19:38' as a datetime.*"):
            parse_datetime("2022-08-19T07:19:38")

        with pytest.raises(
            ValueError,
            match=r".*Cannot parse '2022-08-19T07:19:38.123' as a datetime.*",
        ):
            parse_datetime("2022-08-19T07:19:38.123")

        with pytest.raises(
            ValueError,
            match=r".*Cannot parse '2022-08-19 07:19:38.123Z\+6:00' as a datetime.*",
        ):
            parse_datetime("2022-08-19 07:19:38.123Z+6:00")
