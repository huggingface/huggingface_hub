import unittest
from datetime import datetime
from unittest.mock import patch

import pytest

from dateutil.tz import tzoffset
from huggingface_hub.utils import parse_datetime


class TestDatetimeUtils(unittest.TestCase):
    def test_parse_datetime_using_dateutil(self):
        self.assertEqual(
            parse_datetime("2022-08-19T07:19:38"),
            datetime(2022, 8, 19, 7, 19, 38),
        )

        self.assertEqual(
            parse_datetime("2022-08-19T07:19:38.123"),
            datetime(2022, 8, 19, 7, 19, 38, 123000),
        )

        self.assertEqual(
            parse_datetime("2022-08-19T07:19:38.123Z"),
            datetime(2022, 8, 19, 7, 19, 38, 123000, tzinfo=datetime.timezone.utc),
        )

        # Case that cannot be parsed without dateutil.
        self.assertEqual(
            parse_datetime("2022-08-19 07:19:38.123Z+6:00"),
            datetime(2022, 8, 19, 7, 19, 38, 123000, tzinfo=tzoffset(None, -21600)),
        )

    @patch("huggingface_hub.utils._datetime._dateutil_available", False)
    def test_parse_datetime_without_dateutil(self):
        self.assertEqual(
            parse_datetime("2022-08-19T07:19:38"),
            datetime(2022, 8, 19, 7, 19, 38),
        )

        self.assertEqual(
            parse_datetime("2022-08-19T07:19:38.123"),
            datetime(2022, 8, 19, 7, 19, 38, 123000),
        )

        self.assertEqual(
            parse_datetime("2022-08-19T07:19:38.123Z"),
            datetime(2022, 8, 19, 7, 19, 38, 123000, tzinfo=datetime.timezone.utc),
        )

        with pytest.raises(
            ValueError, match=r"`.*pip install huggingface_hub\[dateutil\]`.*"
        ):
            parse_datetime("2022-08-19 07:19:38.123Z+6:00")
