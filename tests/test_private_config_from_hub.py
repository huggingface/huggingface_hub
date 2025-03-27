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

        assert r.status_code == 200
        assert r.headers[HUGGINGFACE_HEADER_X_XET_ACCESS_TOKEN] == ""

def test_check_private_link_token_item_for_hub_ci(self):
    url = hf_hub_url("__DUMMY_TRANSFORMERS_USER__/test-list-xet-files", filename="lfs_file.bin")

    headers = build_hf_headers()

    r = _request_wrapper(method="HEAD", url=url, headers=headers, allow_redirects=False, follow_relative_redirects=True)

    assert r.status_code == 200
    assert r.headers[HUGGINGFACE_HEADER_X_XET_ACCESS_TOKEN] == ""
