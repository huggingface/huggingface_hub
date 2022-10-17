import unittest

from huggingface_hub.utils._runtime import is_google_colab, is_notebook


class TestRuntimeUtils(unittest.TestCase):
    def test_is_notebook(self) -> None:
        """Test `is_notebook`."""
        self.assertFalse(is_notebook())

    def test_is_google_colab(self) -> None:
        """Test `is_google_colab`."""
        self.assertFalse(is_google_colab())
