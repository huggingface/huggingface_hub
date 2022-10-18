import unittest

from huggingface_hub.utils._chunk_utils import chunk_iterable


class TestUtilsCommon(unittest.TestCase):
    def test_chunk_iterable_non_truncated(self):
        # Can iterable over any iterable (iterator, list, tuple,...)
        for iterable in (range(12), list(range(12)), tuple(range(12))):
            # 12 is a multiple of 4 -> last chunk is not truncated
            for chunk, expected_chunk in zip(
                chunk_iterable(iterable, chunk_size=4),
                [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
            ):
                self.assertListEqual(list(chunk), expected_chunk)

    def test_chunk_iterable_last_chunk_truncated(self):
        # Can iterable over any iterable (iterator, list, tuple,...)
        for iterable in (range(12), list(range(12)), tuple(range(12))):
            # 12 is NOT a multiple of 5 -> last chunk is truncated
            for chunk, expected_chunk in zip(
                chunk_iterable(iterable, chunk_size=5),
                [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11]],
            ):
                self.assertListEqual(list(chunk), expected_chunk)

    def test_chunk_iterable_validation(self):
        with self.assertRaises(ValueError):
            next(chunk_iterable(range(128), 0))

        with self.assertRaises(ValueError):
            next(chunk_iterable(range(128), -1))
