import os
from unittest.mock import patch

import pytest

from huggingface_hub.file_download import get_hf_file_metadata, hf_hub_download, hf_hub_url, try_to_load_from_cache

from .testing_utils import (
    DUMMY_XET_MODEL_ID,
    requires,
    with_production_testing,
)


@requires("hf_xet")
@with_production_testing
class TestXetDownload:
    file_name = "dummy.txt"
    xet_file_name = "dummy.safetensors"

    def test_basic_download(self, tmp_path):
        filepath = hf_hub_download(
            DUMMY_XET_MODEL_ID,
            filename=self.xet_file_name,
            cache_dir=tmp_path,
        )

        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 0

    @pytest.mark.skip(reason="temporarily skipping this test")
    def test_get_xet_file_metadata_basic(self) -> None:
        """Test getting metadata from a file on the Hub."""
        url = hf_hub_url(
            repo_id=DUMMY_XET_MODEL_ID,
            filename=self.xet_file_name,
        )
        metadata = get_hf_file_metadata(url)
        xet_metadata = metadata.xet_metadata
        assert xet_metadata is not None
        assert xet_metadata.endpoint is not None
        assert xet_metadata.access_token is not None
        assert isinstance(xet_metadata.expiration_unix_epoch, int)

    def test_try_to_load_from_cache(self, tmp_path):
        cached_path = try_to_load_from_cache(DUMMY_XET_MODEL_ID, filename=self.xet_file_name, cache_dir=tmp_path)
        assert cached_path is None

        downloaded_path = hf_hub_download(
            DUMMY_XET_MODEL_ID,
            filename=self.xet_file_name,
            cache_dir=tmp_path,
        )

        # Now should find it in cache
        cached_path = try_to_load_from_cache(DUMMY_XET_MODEL_ID, filename=self.xet_file_name, cache_dir=tmp_path)
        assert cached_path == downloaded_path

    def test_cache_reuse(self, tmp_path):
        path1 = hf_hub_download(
            DUMMY_XET_MODEL_ID,
            filename=self.xet_file_name,
            cache_dir=tmp_path,
        )

        assert os.path.exists(path1)

        with patch("huggingface_hub.file_download._download_to_tmp_and_move") as mock:
            # Second download should use cache
            path2 = hf_hub_download(
                DUMMY_XET_MODEL_ID,
                filename=self.xet_file_name,
                cache_dir=tmp_path,
            )

            assert path1 == path2
            mock.assert_not_called()
