import unittest

from huggingface_hub.utils._chunk_utils import chunk_iterable


class TestUtilsCommon(unittest.TestCase):
    def test_chunk_iterable_range(self):
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

    def test_chunk_iterable_list(self):
        lst = [elem for elem in range(128)]
        chunked_iterable = chunk_iterable(lst, chunk_size=8)
        for idx, chunk in enumerate(chunked_iterable):
            self.assertEqual(chunk, list(range(8 * idx, 8 * (idx + 1))))

        lst = [elem for elem in range(12)]
        chunked_iterable = chunk_iterable(lst, 5)
        self.assertListEqual(
            [chunk for chunk in chunked_iterable],
            [list(range(5)), list(range(5, 10)), list(range(10, 12))],
        )

    def test_chunk_iterable_tuple(self):
        tup = tuple(elem for elem in range(128))
        chunked_iterable = chunk_iterable(tup, chunk_size=8)
        for idx, chunk in enumerate(chunked_iterable):
            self.assertEqual(chunk, list(range(8 * idx, 8 * (idx + 1))))

        tup = tuple(elem for elem in range(12))
        chunked_iterable = chunk_iterable(tup, 5)
        self.assertListEqual(
            [chunk for chunk in chunked_iterable],
            [list(range(5)), list(range(5, 10)), list(range(10, 12))],
        )

    def test_chunk_iterable_generator(self):
        gen = (elem for elem in range(128))
        chunked_iterable = chunk_iterable(gen, chunk_size=8)
        for idx, chunk in enumerate(chunked_iterable):
            self.assertEqual(chunk, list(range(8 * idx, 8 * (idx + 1))))

        gen = (elem for elem in range(12))
        chunked_iterable = chunk_iterable(gen, 5)
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
