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
"""Contain helper class to retrieve/store token from/to local cache."""
import os
from pathlib import Path
from typing import Optional


class HfFolder:
    path_token = Path("~/.huggingface/token").expanduser()

    @classmethod
    def save_token(cls, token: str) -> None:
        """
        Save token, creating folder as needed.

        Args:
            token (`str`):
                The token to save to the [`HfFolder`]
        """
        cls.path_token.parent.mkdir(exist_ok=True)
        with cls.path_token.open("w+") as f:
            f.write(token)

    @classmethod
    def get_token(cls) -> Optional[str]:
        """
        Get token or None if not existent.

        Note that a token can be also provided using the
        `HUGGING_FACE_HUB_TOKEN` environment variable.

        Returns:
            `str` or `None`: The token, `None` if it doesn't exist.
        """
        token: Optional[str] = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token is None:
            try:
                return cls.path_token.read_text()
            except FileNotFoundError:
                pass
        return token

    @classmethod
    def delete_token(cls) -> None:
        """
        Deletes the token from storage. Does not fail if token does not exist.
        """
        try:
            cls.path_token.unlink()
        except FileNotFoundError:
            pass
