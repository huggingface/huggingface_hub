import base64
import json
from huggingface_hub.constants import HUGGINGFACE_HEADER_X_XET_ACCESS_TOKEN

from huggingface_hub.file_download import (
    hf_hub_url,
    _request_wrapper,
    build_hf_headers,
)

from .testing_utils import (
    DUMMY_XET_FILE,
    DUMMY_XET_MODEL_ID,
    with_production_testing,
)

@with_production_testing
class TestXetPrivateLinks:
    def test_check_private_link_token_item_for_prod(self):
        url = hf_hub_url(DUMMY_XET_MODEL_ID, filename=DUMMY_XET_FILE)

        headers = build_hf_headers()

        r = _request_wrapper(method="HEAD", url=url, headers=headers, allow_redirects=False, follow_relative_redirects=True)

        assert r.status_code == 302 

        xet_token = r.headers[HUGGINGFACE_HEADER_X_XET_ACCESS_TOKEN]

        parts = xet_token.split(".")
        assert len(parts) == 3
        json_string = base64.b64decode(parts[1] + '==')

        token_info = json.loads(json_string)
        assert token_info["userId"] == "public"
        assert token_info["usePrivateLink"] == True 


def test_check_private_link_token_item_for_hub_ci():
    url = hf_hub_url("__DUMMY_TRANSFORMERS_USER__/test-list-xet-files", filename="lfs_file.bin")

    headers = build_hf_headers()

    r = _request_wrapper(method="HEAD", url=url, headers=headers, allow_redirects=False, follow_relative_redirects=True)

    assert r.status_code == 302 
    xet_token = r.headers[HUGGINGFACE_HEADER_X_XET_ACCESS_TOKEN]

    parts = xet_token.split(".")
    assert len(parts) == 3
    json_string = base64.b64decode(parts[1] + '==')

    token_info = json.loads(json_string)
    assert token_info["userId"] == "public"
    assert token_info["usePrivateLink"] == True 
