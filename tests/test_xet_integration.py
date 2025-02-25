import os
from unittest import TestCase, skip
from io import BytesIO

from huggingface_hub import HfApi

from huggingface_hub.utils import is_xet_available

from .testing_constants import ENDPOINT_STAGING, TOKEN 
from .testing_utils import repo_name

WORKING_REPO_SUBDIR = f"fixtures/working_repo_{__name__.split('.')[-1]}"
WORKING_REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), WORKING_REPO_SUBDIR)

def require_hf_xet(test_case):
    """
    Decorator marking a test that requires hf_xet.
    These tests are skipped when hf_xet is not installed.
    """
    if not is_xet_available():
        return skip("Test requires hf_xet")(test_case)
    else:
        return test_case

@require_hf_xet
class TestHfXet(TestCase): 
    def setUp(self) -> None:
        self.content = b"RandOm Xet ConTEnT" * 1024  
        self._token = TOKEN
        self._api = HfApi(endpoint=ENDPOINT_STAGING, token=self._token)

        self._repo_id = self._api.create_repo(repo_name()).repo_id

    def tearDown(self) -> None:
        self._api.delete_repo(repo_id=self._repo_id)

    def test__xet_available(self):
        # renaming to this runs first
        self.assertTrue(is_xet_available())

