# Copyright 2022 The HuggingFace Team. All rights reserved.
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
"""Contains commands to print information about the environment and version.

Usage:
    hf env
    hf version
"""

from huggingface_hub import __version__

from ..utils import dump_environment_info


def env() -> None:
    """Print information about the environment."""
    dump_environment_info()


def version() -> None:
    """Print CLI version."""
    print(f"huggingface_hub version: {__version__}")
