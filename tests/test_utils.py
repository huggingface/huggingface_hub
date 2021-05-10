import unittest

from huggingface_hub.utils import normalize


class UtilsTests(unittest.TestCase):
    def test_normalize(self):
        path_in_repo = "./hello/World.bin"
        expected = "hello/World.bin"
        self.assertEqual(normalize(path_in_repo), expected)

        path_in_repo = "./subdir/with space/file.pth"
        expected = "subdir/with space/file.pth"
        self.assertEqual(normalize(path_in_repo), expected)

        path_in_repo = ".\\subdir\\on Windows\\file.pth"
        expected = "subdir/on Windows/file.pth"
        self.assertEqual(normalize(path_in_repo), expected)

        path_in_repo = "/mnt/dvxa1/absolute/../path.bin"
        expected = "mnt/dvxa1/path.bin"
        self.assertEqual(normalize(path_in_repo), expected)

        path_in_repo = "C:\\Complex mix/..\\of everything//./file.ht5"
        expected = "C:/of everything/file.ht5"
        self.assertEqual(normalize(path_in_repo), expected)
