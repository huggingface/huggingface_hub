import re
from typing import Dict, Optional

from .constants import ASSET_REPO
from .file_download import get_hf_file_metadata, hf_hub_url, hf_hub_download
from .utils import validate_hf_hub_args


def _get_brand_assets() -> Dict[str, Dict[str, str]]:
    """Fetch and parse the list of brand assets from the huggingface/brand-assets repo.

    Returns a dict where keys are asset names (e.g. "hf-logo") and values are dicts
    mapping file extensions to filenames.
    """
    from .hf_api import HfApi

    api = HfApi()
    files = api.list_repo_files(ASSET_REPO, repo_type="dataset")

    assets = {}
    for filename in files:
        if not filename or filename.startswith(".") or "/" in filename:
            continue  # Skip hidden files, directories, etc.
        match = re.match(r"^(.+)\.([a-z]+)$", filename)
        if match:
            name, ext = match.groups()
            if ext in ("svg", "png", "ai"):
                assets.setdefault(name, {})[ext] = filename
    return assets


@validate_hf_hub_args
def list_brand_assets() -> Dict[str, Dict[str, str]]:
    """List all available brand assets from the huggingface/brand-assets dataset.

    Returns:
        Dict[str, Dict[str, str]]: A dictionary where keys are asset names and values
        are dictionaries mapping file extensions to filenames.

    Example:
        >>> from huggingface_hub import list_brand_assets
        >>> assets = list_brand_assets()
        >>> print(assets["hf-logo"])
        {'svg': 'hf-logo.svg', 'png': 'hf-logo.png', 'ai': 'hf-logo.ai'}
    """
    return _get_brand_assets()


def _pick_filename(name_or_filename: str, file_type: Optional[str]) -> str:
    """Resolve an asset logical name and optional type to a filename in the asset repo.

    Accepts either a logical asset key (e.g. "hf-logo") or a direct filename
    (e.g. "hf-logo.svg"). If a type is provided it will be used when available.
    """
    # If the user passed an explicit filename, use it as-is
    if "." in name_or_filename and name_or_filename.split(".")[-1] in ("svg", "png", "ai"):
        return name_or_filename

    assets = _get_brand_assets()
    if name_or_filename not in assets:
        raise ValueError(f"Unknown asset '{name_or_filename}'. Known keys: {sorted(assets.keys())}")

    formats = assets[name_or_filename]
    if file_type:
        file_type = file_type.lower()
        if file_type in formats:
            return formats[file_type]

    # Default preference order: svg, png, ai
    for preferred in ("svg", "png", "ai"):
        if preferred in formats:
            return formats[preferred]

    # As a last resort, return the first available file
    return next(iter(formats.values()))


def get_brand_asset_url(
    name_or_filename: str, *, file_type: Optional[str] = None, revision: str = "main", token: Optional[str] = None, endpoint: Optional[str] = None
) -> str:
    """Return a final (possibly CDN-signed) URL for an asset hosted in the
    `huggingface/brand-assets` dataset.

    This builds the hub `resolve` URL and then performs a HEAD request to
    retrieve the final `Location` (or the request URL) so callers get the
    direct link they can use in HTML/CSS or pass to other tools.
    """
    filename = _pick_filename(name_or_filename, file_type)
    url = hf_hub_url(repo_id=ASSET_REPO, filename=filename, repo_type="dataset", revision=revision, endpoint=endpoint)
    meta = get_hf_file_metadata(url, token=token, endpoint=endpoint)
    return meta.location


def download_brand_asset(
    name_or_filename: str,
    *,
    file_type: Optional[str] = None,
    revision: str = "main",
    token: Optional[str] = None,
    repo_type: str = "dataset",
    cache_dir: Optional[str] = None,
    force_download: bool = False,
):
    """Download the requested brand asset into the local cache and return
    the local path. This delegates to the existing `hf_hub_download` helper to
    preserve the library's caching and authentication behavior.
    """
    filename = _pick_filename(name_or_filename, file_type)
    # Delegate to hf_hub_download which implements caching and xet handling.
    return hf_hub_download(
        repo_id=ASSET_REPO,
        filename=filename,
        repo_type=repo_type,
        revision=revision,
        token=token,
        force_download=force_download,
        local_dir=cache_dir,
    )


__all__ = ["ASSET_REPO", "get_brand_asset_url", "download_brand_asset"]
