import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from huggingface_hub.hf_assets import (
    _get_brand_assets,
    _pick_filename,
    list_brand_assets,
    get_brand_asset_url,
    download_brand_asset,
    ASSET_REPO,
)


class HfAssetsListUnitTest(unittest.TestCase):
    @patch("huggingface_hub.hf_api.HfApi")
    def test_list_brand_assets_basic(self, mock_hf_api_class):
        mock_api = Mock()
        mock_hf_api_class.return_value = mock_api
        mock_api.list_repo_files.return_value = [
            "hf-logo.svg",
            "hf-logo.png",
            "hf-logo.ai",
            "another-asset.svg",
            ".hidden_file",
            "folder/file.txt",
            "invalid.ext",
        ]

        assets = list_brand_assets()

        assert isinstance(assets, dict)
        assert set(assets.keys()) == {"hf-logo", "another-asset"}
        assert assets["hf-logo"] == {"svg": "hf-logo.svg", "png": "hf-logo.png", "ai": "hf-logo.ai"}
        assert assets["another-asset"] == {"svg": "another-asset.svg"}

        mock_hf_api_class.assert_called_once()
        mock_api.list_repo_files.assert_called_once_with(ASSET_REPO, repo_type="dataset")


class HfAssetsGetBrandAssetsUnitTest(unittest.TestCase):
    @patch("huggingface_hub.hf_api.HfApi")
    def test_get_brand_assets_filters_and_merges(self, mock_hf_api_class):
        mock_api = Mock()
        mock_hf_api_class.return_value = mock_api
        mock_api.list_repo_files.return_value = [
            "hf-logo.svg",
            "hf-logo.png",
            "hf-logo.ai",
            "duplicate.svg",
            "duplicate.png",
            ".ignored.svg",
            "dir/file.svg",
            "not-an-asset.txt",
        ]

        assets = _get_brand_assets()

        assert "hf-logo" in assets
        assert assets["hf-logo"] == {"svg": "hf-logo.svg", "png": "hf-logo.png", "ai": "hf-logo.ai"}
        assert ".ignored" not in assets
        assert "dir" not in assets
        assert "not-an-asset" not in assets
        assert "duplicate" in assets
        assert set(assets["duplicate"].keys()) == {"svg", "png"}


class HfAssetsPickFilenameUnitTest(unittest.TestCase):
    def test_explicit_filename_passthrough(self):
        assert _pick_filename("hf-logo.svg", None) == "hf-logo.svg"
        assert _pick_filename("hf-logo.png", None) == "hf-logo.png"
        assert _pick_filename("hf-logo.ai", None) == "hf-logo.ai"

    @patch("huggingface_hub.hf_assets._get_brand_assets")
    def test_asset_name_default_preference_and_type(self, mock_get_assets):
        mock_get_assets.return_value = {
            "hf-logo": {"png": "hf-logo.png", "svg": "hf-logo.svg"},
            "wordmark": {"ai": "wordmark.ai"},
            "icon": {"png": "icon.png"},
        }

        assert _pick_filename("hf-logo", None) == "hf-logo.svg"
        assert _pick_filename("icon", None) == "icon.png"
        assert _pick_filename("wordmark", None) == "wordmark.ai"
        assert _pick_filename("hf-logo", "png") == "hf-logo.png"
        assert _pick_filename("hf-logo", "SVG") == "hf-logo.svg"

        with self.assertRaises(ValueError):
            _pick_filename("unknown", None)


class HfAssetsUrlUnitTest(unittest.TestCase):
    @patch("huggingface_hub.hf_assets.get_hf_file_metadata")
    @patch("huggingface_hub.hf_assets.hf_hub_url")
    @patch("huggingface_hub.hf_assets._pick_filename")
    def test_get_brand_asset_url_resolves_redirect(self, mock_pick, mock_hub_url, mock_meta):
        mock_pick.return_value = "hf-logo.svg"
        mock_hub_url.return_value = "https://example.com/resolve/main/hf-logo.svg"
        fake_meta = Mock()
        fake_meta.location = "https://cdn.example.com/hf-logo.svg"
        mock_meta.return_value = fake_meta

        url = get_brand_asset_url(
            "hf-logo",
            file_type="svg",
            revision="main",
            token="fake_token",
            endpoint="https://hub-ci.huggingface.co",
        )

        assert url == "https://cdn.example.com/hf-logo.svg"
        mock_pick.assert_called_once_with("hf-logo", "svg")
        mock_hub_url.assert_called_once_with(
            repo_id=ASSET_REPO,
            filename="hf-logo.svg",
            repo_type="dataset",
            revision="main",
            endpoint="https://hub-ci.huggingface.co",
        )
        mock_meta.assert_called_once_with(
            "https://example.com/resolve/main/hf-logo.svg",
            token="fake_token",
            endpoint="https://hub-ci.huggingface.co",
        )


class HfAssetsDownloadUnitTest(unittest.TestCase):
    @patch("huggingface_hub.hf_assets.hf_hub_download")
    @patch("huggingface_hub.hf_assets._pick_filename")
    def test_download_brand_asset_delegates_to_hf_hub_download(self, mock_pick, mock_download):
        mock_pick.return_value = "hf-logo.png"
        mock_download.return_value = "/tmp/cache/hf-logo.png"

        path = download_brand_asset(
            "hf-logo",
            file_type="png",
            revision="v1",
            token="token",
            repo_type="dataset",
            cache_dir="/tmp/cache",
            force_download=True,
        )

        assert path == "/tmp/cache/hf-logo.png"
        mock_pick.assert_called_once_with("hf-logo", "png")
        mock_download.assert_called_once_with(
            repo_id=ASSET_REPO,
            filename="hf-logo.png",
            repo_type="dataset",
            revision="v1",
            token="token",
            force_download=True,
            local_dir="/tmp/cache",
        )


class HfAssetsProductionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pass

    def test_list_brand_assets_overview(self) -> None:
        assets = list_brand_assets()
        assert isinstance(assets, dict)
        assert "hf-logo" in assets
        assert "svg" in assets["hf-logo"]
        assert assets["hf-logo"]["svg"].endswith(".svg")

    def test_get_asset_url_svg(self) -> None:
        url = get_brand_asset_url("hf-logo", file_type="svg")
        assert isinstance(url, str)
        assert url.startswith("http")
        assert "hf-logo.svg" in url

    def test_download_png(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = download_brand_asset("hf-logo", file_type="png", cache_dir=d)
            p = Path(path)
            assert p.exists()
            assert p.stat().st_size > 0
            assert p.name.endswith(".png")
            assert str(p).startswith(d)


if __name__ == "__main__":
    unittest.main()
