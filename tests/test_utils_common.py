import unittest

from huggingface_hub.utils.common import chunk_iterable


class TestUtilsCommon(unittest.TestCase):
    def test_chunk_iterable(self):
        iterable = range(128)
        chunked_iterable = chunk_iterable(iterable, chunk_size=8)
        for idx, chunk in enumerate(chunked_iterable):
            self.assertEqual(chunk, list(range(8 * idx, 8 * (idx + 1))))

        iterable = range(12)
        chunked_iterable = chunk_iterable(iterable, 5)
        self.assertListEqual(
            [chunk for chunk in chunked_iterable],
            [list(range(5)), list(range(5, 10)), list(range(10, 12))],
        )

    def test_chunk_iterable_validation(self):
        iterable = range(128)
        with self.assertRaises(ValueError):
            next(chunk_iterable(iterable, 0))

        iterable = range(128)
        with self.assertRaises(ValueError):
            next(chunk_iterable(iterable, -1))
