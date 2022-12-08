# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
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
from dataclasses import dataclass
from typing import List


@dataclass
class UserLikes:
    """
    Contains information about a user likes on the Hub.

    Args:
        user (`str`):
            Name of the user for which we fetched the likes.
        total (`int`):
            Total number of likes (models, datasets and Spaces combined).
        datasets (`List[str]`):
            List of datasets liked by the user (as repo_ids).
        models (`List[str]`):
            List of models liked by the user (as repo_ids).
        spaces (`List[str]`):
            List of spaces liked by the user (as repo_ids).
    """

    # Metadata
    user: str
    total: int

    # User likes
    datasets: List[str]
    models: List[str]
    spaces: List[str]
