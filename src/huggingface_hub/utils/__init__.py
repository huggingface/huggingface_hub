# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Public utility helpers, loaded on first use.

Most library modules import a small subset of this package. Keeping the facade lazy
prevents every such import from also loading unrelated cache, terminal, subprocess,
and optional-dependency helpers.
"""

# ruff: noqa: F401, F403

import importlib
import sys
import types
from typing import TYPE_CHECKING


_SUBMODULES = {
    "_auth",
    "_detect_agent",
    "_http",
    "_tqdm",
    "logging",
}

_SUBMOD_ATTRS = {
    "_auth": [
        "get_stored_tokens",
        "get_token",
    ],
    "_cache_assets": [
        "cached_assets_path",
    ],
    "_cache_manager": [
        "CachedFileInfo",
        "CachedIncompleteFileInfo",
        "CachedRepoInfo",
        "CachedRevisionInfo",
        "DeleteCacheStrategy",
        "HFCacheInfo",
        "_format_size",
        "scan_cache_dir",
    ],
    "_chunk_utils": [
        "chunk_iterable",
    ],
    "_datetime": [
        "parse_datetime",
    ],
    "_detect_agent": [
        "detect_agent",
        "is_agent",
    ],
    "_experimental": [
        "experimental",
    ],
    "_fixes": [
        "SoftTemporaryDirectory",
        "WeakFileLock",
        "yaml_dump",
    ],
    "_git_credential": [
        "list_credential_helpers",
        "set_git_credential",
        "unset_git_credential",
    ],
    "_headers": [
        "build_hf_headers",
        "get_token_to_send",
    ],
    "_hf_uris": [
        "HfMount",
        "HfUri",
        "is_hf_uri",
        "parse_hf_mount",
        "parse_hf_uri",
    ],
    "_http": [
        "ASYNC_CLIENT_FACTORY_T",
        "CLIENT_FACTORY_T",
        "RateLimitInfo",
        "close_session",
        "fix_hf_endpoint_in_url",
        "get_async_session",
        "get_session",
        "hf_raise_for_status",
        "http_backoff",
        "http_stream_backoff",
        "parse_ratelimit_headers",
        "set_async_client_factory",
        "set_client_factory",
    ],
    "_pagination": [
        "paginate",
    ],
    "_paths": [
        "DEFAULT_IGNORE_PATTERNS",
        "FORBIDDEN_FOLDERS",
        "filter_repo_objects",
    ],
    "_runtime": [
        "dump_environment_info",
        "get_aiohttp_version",
        "get_fastai_version",
        "get_fastapi_version",
        "get_fastcore_version",
        "get_gradio_version",
        "get_graphviz_version",
        "get_hf_hub_version",
        "get_jinja_version",
        "get_numpy_version",
        "get_pillow_version",
        "get_pydantic_version",
        "get_pydot_version",
        "get_python_version",
        "get_tensorboard_version",
        "get_tf_version",
        "get_torch_version",
        "installation_method",
        "is_aiohttp_available",
        "is_colab_enterprise",
        "is_fastai_available",
        "is_fastapi_available",
        "is_fastcore_available",
        "is_google_colab",
        "is_gradio_available",
        "is_graphviz_available",
        "is_jinja_available",
        "is_notebook",
        "is_numpy_available",
        "is_package_available",
        "is_pillow_available",
        "is_pydantic_available",
        "is_pydot_available",
        "is_safetensors_available",
        "is_tensorboard_available",
        "is_tf_available",
        "is_torch_available",
    ],
    "_safetensors": [
        "SafetensorsFileMetadata",
        "SafetensorsRepoMetadata",
        "TensorInfo",
    ],
    "_subprocess": [
        "capture_output",
        "run_interactive_subprocess",
        "run_subprocess",
    ],
    "_telemetry": [
        "send_telemetry",
    ],
    "_terminal": [
        "ANSI",
        "StatusLine",
        "select_choice",
        "tabulate",
    ],
    "_typing": [
        "is_jsonable",
        "is_simple_optional_type",
        "unwrap_simple_optional_type",
    ],
    "_validators": [
        "validate_hf_hub_args",
        "validate_repo_id",
    ],
    "_xet": [
        "XetFileData",
        "XetTokenType",
        "parse_xet_file_data_from_response",
    ],
    "huggingface_hub.errors": [
        "BadRequestError",
        "BucketNotFoundError",
        "CacheNotFound",
        "CorruptedCacheException",
        "DisabledRepoError",
        "EntryNotFoundError",
        "FileMetadataError",
        "GatedRepoError",
        "HFValidationError",
        "HfHubHTTPError",
        "JobNotFoundError",
        "LocalEntryNotFoundError",
        "LocalTokenNotFoundError",
        "NotASafetensorsRepoError",
        "OfflineModeIsEnabled",
        "RepositoryNotFoundError",
        "RevisionNotFoundError",
        "SafetensorsParsingError",
    ],
    "tqdm": [
        "are_progress_bars_disabled",
        "disable_progress_bars",
        "enable_progress_bars",
        "hf_thread_map",
        "is_tqdm_disabled",
        "silent_tqdm",
        "tqdm",
        "tqdm_stream_file",
    ],
}

__all__ = sorted(["httpx", *(attr for attrs in _SUBMOD_ATTRS.values() for attr in attrs if not attr.startswith("_"))])

_ATTR_TO_MODULE = {attr: module for module, attrs in _SUBMOD_ATTRS.items() for attr in attrs}


def __getattr__(name: str):
    if name == "httpx":  # Forward compatibility: this will be httpx2 in huggingface_hub v2.x.
        return importlib.import_module("httpx")
    if name in _SUBMODULES:
        module_name = "tqdm" if name == "_tqdm" else name
        return importlib.import_module(f"{__name__}.{module_name}")
    if attr_module_name := _ATTR_TO_MODULE.get(name):
        if attr_module_name.startswith("huggingface_hub."):
            module = importlib.import_module(attr_module_name)
        else:
            module = importlib.import_module(f"{__name__}.{attr_module_name}")
        return getattr(module, name)
    raise AttributeError(f"No {__name__} attribute {name}")


def __dir__() -> list[str]:
    return [*__all__, "httpx"]


class _LazyUtilsModule(types.ModuleType):
    def __getattribute__(self, name: str):
        # Import machinery stores the `tqdm` submodule on this package. The public
        # `utils.tqdm` attribute has historically referred to the class instead.
        if name == "tqdm":
            return importlib.import_module(f"{__name__}.tqdm").tqdm
        return super().__getattribute__(name)


sys.modules[__name__].__class__ = _LazyUtilsModule


if TYPE_CHECKING:  # pragma: no cover
    import httpx as httpx  # noqa: F401

    from huggingface_hub.errors import (
        BadRequestError,  # noqa: F401
        BucketNotFoundError,  # noqa: F401
        CacheNotFound,  # noqa: F401
        CorruptedCacheException,  # noqa: F401
        DisabledRepoError,  # noqa: F401
        EntryNotFoundError,  # noqa: F401
        FileMetadataError,  # noqa: F401
        GatedRepoError,  # noqa: F401
        HfHubHTTPError,  # noqa: F401
        HFValidationError,  # noqa: F401
        JobNotFoundError,  # noqa: F401
        LocalEntryNotFoundError,  # noqa: F401
        LocalTokenNotFoundError,  # noqa: F401
        NotASafetensorsRepoError,  # noqa: F401
        OfflineModeIsEnabled,  # noqa: F401
        RepositoryNotFoundError,  # noqa: F401
        RevisionNotFoundError,  # noqa: F401
        SafetensorsParsingError,  # noqa: F401
    )

    from . import _auth as _auth  # noqa: F401
    from . import _detect_agent as _detect_agent  # noqa: F401
    from . import _http as _http  # noqa: F401
    from . import logging as logging  # noqa: F401
    from . import tqdm as _tqdm  # noqa: F401
    from ._auth import (
        get_stored_tokens,  # noqa: F401
        get_token,  # noqa: F401
    )
    from ._cache_assets import cached_assets_path  # noqa: F401
    from ._cache_manager import (
        CachedFileInfo,  # noqa: F401
        CachedIncompleteFileInfo,  # noqa: F401
        CachedRepoInfo,  # noqa: F401
        CachedRevisionInfo,  # noqa: F401
        DeleteCacheStrategy,  # noqa: F401
        HFCacheInfo,  # noqa: F401
        _format_size,  # noqa: F401
        scan_cache_dir,  # noqa: F401
    )
    from ._chunk_utils import chunk_iterable  # noqa: F401
    from ._datetime import parse_datetime  # noqa: F401
    from ._detect_agent import (
        detect_agent,  # noqa: F401
        is_agent,  # noqa: F401
    )
    from ._experimental import experimental  # noqa: F401
    from ._fixes import (
        SoftTemporaryDirectory,  # noqa: F401
        WeakFileLock,  # noqa: F401
        yaml_dump,  # noqa: F401
    )
    from ._git_credential import (
        list_credential_helpers,  # noqa: F401
        set_git_credential,  # noqa: F401
        unset_git_credential,  # noqa: F401
    )
    from ._headers import (
        build_hf_headers,  # noqa: F401
        get_token_to_send,  # noqa: F401
    )
    from ._hf_uris import (
        HfMount,  # noqa: F401
        HfUri,  # noqa: F401
        is_hf_uri,  # noqa: F401
        parse_hf_mount,  # noqa: F401
        parse_hf_uri,  # noqa: F401
    )
    from ._http import (
        ASYNC_CLIENT_FACTORY_T,  # noqa: F401
        CLIENT_FACTORY_T,  # noqa: F401
        RateLimitInfo,  # noqa: F401
        close_session,  # noqa: F401
        fix_hf_endpoint_in_url,  # noqa: F401
        get_async_session,  # noqa: F401
        get_session,  # noqa: F401
        hf_raise_for_status,  # noqa: F401
        http_backoff,  # noqa: F401
        http_stream_backoff,  # noqa: F401
        parse_ratelimit_headers,  # noqa: F401
        set_async_client_factory,  # noqa: F401
        set_client_factory,  # noqa: F401
    )
    from ._pagination import paginate  # noqa: F401
    from ._paths import (
        DEFAULT_IGNORE_PATTERNS,  # noqa: F401
        FORBIDDEN_FOLDERS,  # noqa: F401
        filter_repo_objects,  # noqa: F401
    )
    from ._runtime import (
        dump_environment_info,  # noqa: F401
        get_aiohttp_version,  # noqa: F401
        get_fastai_version,  # noqa: F401
        get_fastapi_version,  # noqa: F401
        get_fastcore_version,  # noqa: F401
        get_gradio_version,  # noqa: F401
        get_graphviz_version,  # noqa: F401
        get_hf_hub_version,  # noqa: F401
        get_jinja_version,  # noqa: F401
        get_numpy_version,  # noqa: F401
        get_pillow_version,  # noqa: F401
        get_pydantic_version,  # noqa: F401
        get_pydot_version,  # noqa: F401
        get_python_version,  # noqa: F401
        get_tensorboard_version,  # noqa: F401
        get_tf_version,  # noqa: F401
        get_torch_version,  # noqa: F401
        installation_method,  # noqa: F401
        is_aiohttp_available,  # noqa: F401
        is_colab_enterprise,  # noqa: F401
        is_fastai_available,  # noqa: F401
        is_fastapi_available,  # noqa: F401
        is_fastcore_available,  # noqa: F401
        is_google_colab,  # noqa: F401
        is_gradio_available,  # noqa: F401
        is_graphviz_available,  # noqa: F401
        is_jinja_available,  # noqa: F401
        is_notebook,  # noqa: F401
        is_numpy_available,  # noqa: F401
        is_package_available,  # noqa: F401
        is_pillow_available,  # noqa: F401
        is_pydantic_available,  # noqa: F401
        is_pydot_available,  # noqa: F401
        is_safetensors_available,  # noqa: F401
        is_tensorboard_available,  # noqa: F401
        is_tf_available,  # noqa: F401
        is_torch_available,  # noqa: F401
    )
    from ._safetensors import (
        SafetensorsFileMetadata,  # noqa: F401
        SafetensorsRepoMetadata,  # noqa: F401
        TensorInfo,  # noqa: F401
    )
    from ._subprocess import (
        capture_output,  # noqa: F401
        run_interactive_subprocess,  # noqa: F401
        run_subprocess,  # noqa: F401
    )
    from ._telemetry import send_telemetry  # noqa: F401
    from ._terminal import (
        ANSI,  # noqa: F401
        StatusLine,  # noqa: F401
        select_choice,  # noqa: F401
        tabulate,  # noqa: F401
    )
    from ._typing import (
        is_jsonable,  # noqa: F401
        is_simple_optional_type,  # noqa: F401
        unwrap_simple_optional_type,  # noqa: F401
    )
    from ._validators import (
        validate_hf_hub_args,  # noqa: F401
        validate_repo_id,  # noqa: F401
    )
    from ._xet import (
        XetFileData,  # noqa: F401
        XetTokenType,  # noqa: F401
        parse_xet_file_data_from_response,  # noqa: F401
    )
    from .tqdm import (
        are_progress_bars_disabled,  # noqa: F401
        disable_progress_bars,  # noqa: F401
        enable_progress_bars,  # noqa: F401
        hf_thread_map,  # noqa: F401
        is_tqdm_disabled,  # noqa: F401
        silent_tqdm,  # noqa: F401
        tqdm,  # noqa: F401
        tqdm_stream_file,  # noqa: F401
    )
