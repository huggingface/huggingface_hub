import os
from typing import Dict, Optional, Union
from unittest import TestCase, skip

from huggingface_hub import HfApi
from huggingface_hub.constants import REPO_TYPE_MODEL, REPO_TYPES
from huggingface_hub.utils import get_session, is_xet_available

from .testing_constants import ENDPOINT_STAGING, TOKEN
from .testing_utils import repo_name


WORKING_REPO_SUBDIR = f"fixtures/working_repo_{__name__.split('.')[-1]}"
WORKING_REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), WORKING_REPO_SUBDIR)


def set_xet_enabled(
    api: HfApi,
    repo_id: str,
    token: Union[str, bool, None] = None,
    repo_type: Optional[str] = None,
) -> Dict[str, bool]:
    """
    Enables XET on a repo.

    This is a test utility and will go away when xet is enabled for all repos by default.
    """
    if repo_type not in REPO_TYPES:
        raise ValueError(f"Invalid repo type, must be one of {REPO_TYPES}")
    if repo_type is None:
        repo_type = REPO_TYPE_MODEL  # default repo type

    r = get_session().put(
        url=f"{api.endpoint}/api/{repo_type}s/{repo_id}/settings",
        headers=api._build_hf_headers(token=token),
        json={"xetEnabled": True},
    )
    return r.json()


def is_xet_enabled(
    api: HfApi,
    repo_id: str,
    token: Union[str, bool, None] = None,
    repo_type: Optional[str] = None,
) -> bool:
    """
    Checks if XET is enabled on a repo.

    This is a test utility and will go away when xet is enabled for all repos by default.
    """
    if repo_type not in REPO_TYPES:
        raise ValueError(f"Invalid repo type, must be one of {REPO_TYPES}")
    if repo_type is None:
        repo_type = REPO_TYPE_MODEL  # default repo type

    r = get_session().get(
        url=f"{api.endpoint}/api/{repo_type}s/{repo_id}/xet-read-repo/main",
        headers=api._build_hf_headers(token=token),
    )
    return r.status_code != 501 and r.text != "XET is not enabled in this version of Hub."


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
        set_xet_enabled(api=self._api, repo_id=self._repo_id, token=self._token)

    def tearDown(self) -> None:
        self._api.delete_repo(repo_id=self._repo_id)

    def test__xet_enabled_on_repo(self):
        self.assertTrue(is_xet_enabled(api=self._api, repo_id=self._repo_id, token=self._token))

    def test__xet_available(self):
        self.assertTrue(is_xet_available())
