# coding=utf-8
# Copyright 2022-present, the HuggingFace Inc. team.
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
"""Check presence of installed packages at runtime."""
import sys
from typing import Optional

import packaging.version

from .. import __version__


_PY_VERSION: str = sys.version.split()[0].rstrip("+")

if packaging.version.Version(_PY_VERSION) < packaging.version.Version("3.8.0"):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


_package_versions = {}

_CANDIDATES = {
    "torch": {"torch"},
    "pydot": {"pydot"},
    "graphviz": {"graphviz"},
    "tensorflow": (
        "tensorflow",
        "tensorflow-cpu",
        "tensorflow-gpu",
        "tf-nightly",
        "tf-nightly-cpu",
        "tf-nightly-gpu",
        "intel-tensorflow",
        "intel-tensorflow-avx512",
        "tensorflow-rocm",
        "tensorflow-macos",
    ),
    "fastai": {"fastai"},
    "fastcore": {"fastcore"},
    "jinja": {"Jinja2"},
}

# Check once at runtime
for candidate_name, package_names in _CANDIDATES.items():
    _package_versions[candidate_name] = "N/A"
    for name in package_names:
        try:
            _package_versions[candidate_name] = importlib_metadata.version(name)
            break
        except importlib_metadata.PackageNotFoundError:
            pass


def _get_version(package_name: str) -> Optional[str]:
    return _package_versions.get(package_name, "N/A")


def _is_available(package_name: str) -> bool:
    return _get_version(package_name) != "N/A"


# Python
def get_python_version() -> str:
    return _PY_VERSION


# Huggingface Hub
def get_hf_hub_version() -> str:
    return __version__


# FastAI
def is_fastai_available() -> bool:
    return _is_available("fastai")


def get_fastai_version() -> str:
    return _get_version("fastai")


# Fastcore
def is_fastcore_available() -> bool:
    return _is_available("fastcore")


def get_fastcore_version() -> str:
    return _get_version("fastcore")


# Graphviz
def is_graphviz_available() -> bool:
    return _is_available("graphviz")


def get_graphviz_version() -> str:
    return _get_version("graphviz")


# Jinja
def is_jinja_available() -> bool:
    return _is_available("jinja")


def get_jinja_version() -> str:
    return _get_version("jinja")


# Pydot
def is_pydot_available() -> bool:
    return _is_available("pydot")


def get_pydot_version() -> str:
    return _get_version("pydot")


# Tensorflow
def is_tf_available() -> bool:
    return _is_available("tensorflow")


def get_tf_version() -> str:
    return _get_version("tensorflow")


# Torch
def is_torch_available() -> bool:
    return _is_available("torch")


def get_torch_version() -> str:
    return _get_version("torch")
