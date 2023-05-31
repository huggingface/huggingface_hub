# Copyright 2023 The HuggingFace Team. All rights reserved.
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
"""Contains a logger to push training logs to the Hub, using Tensorboard."""
import os
import warnings
from concurrent.futures import Future
from typing import TYPE_CHECKING, List, Optional, Union

from .hf_api import create_repo, upload_folder
from .utils import is_tensorboard_available


if is_tensorboard_available():
    from tensorboardX import SummaryWriter

    # TODO: clarify: should we import from torch.utils.tensorboard ?
else:
    SummaryWriter = object  # Dummy class to avoid failing at import. Will raise on instance creation.

if TYPE_CHECKING:
    from tensorboardX import SummaryWriter


class HFSummaryWriter(SummaryWriter):
    """
    Wrapper around the tensorboard's `SummaryWriter` to push training logs to the Hub.

    Data is logged locally and then pushed to the Hub asynchronously. Pushing data to the Hub is done in a separate
    thread to avoid blocking the training script. In particular, if the upload fails for any reason (e.g. a connection
    issue), the main script will not be interrupted.

    Args:
        repo_id (`str`):
            The id of the repo to which the logs will be pushed.
        logdir (`str`, *optional*):
            The directory where the logs will be written. If not specified, a local directory will be created by the
            underlying `SummaryWriter` object.
        repo_type (`str`, *optional*):
            The type of the repo to which the logs will be pushed. Defaults to "model".
        repo_revision (`str`, *optional*):
            The revision of the repo to which the logs will be pushed. Defaults to "main".
        repo_private (`bool`, *optional*):
            Whether to create a private repo or not. Defaults to False. This argument is ignored if the repo already
            exists.
        path_in_repo (`str`, *optional*):
            The path to the folder in the repo where the logs will be pushed. Defaults to "tensorboard/".
        repo_allow_patterns (`List[str]` or `str`, *optional*):
            A list of patterns to include in the upload. Defaults to `"*.tfevents.*"`. Check out the
            [upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload#upload-a-folder) for more details.
        repo_ignore_patterns (`List[str]` or `str`, *optional*):
            A list of patterns to exclude in the upload. Check out the
            [upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload#upload-a-folder) for more details.
        token (`str`, *optional*):
            Authentication token. Will default to the stored token. See https://huggingface.co/settings/token for more
            details
        **kwargs:
            Additional keyword arguments passed to `SummaryWriter`.

    Example:
    ```py
    from huggingface_hub import HFSummaryWriter

    logger = HFSummaryWriter(repo_id="test_hf_logger")
    logger.log_hyperparams({"a": 1, "b": 2})
    logger.push_to_hub()
    ```
    """

    def __new__(cls, *args, **kwargs) -> "HFSummaryWriter":
        if not is_tensorboard_available():
            raise ImportError(
                "You must have `tensorboard` installed to use `HFSummaryWriter`. Please run `pip install --upgrade"
                " tensorboardX` first."
            )
        return super().__new__(cls, *args, **kwargs)

    def __init__(
        self,
        repo_id: str,
        *,
        logdir: Optional[str] = None,
        repo_type: Optional[str] = None,
        repo_revision: Optional[str] = None,
        repo_private: bool = False,
        path_in_repo: Optional[str] = "tensorboard",
        repo_allow_patterns: Optional[Union[List[str], str]] = "*.tfevents.*",
        repo_ignore_patterns: Optional[Union[List[str], str]] = None,
        token: Optional[str] = None,
        **kwargs,
    ):
        # Initialize SummaryWriter
        super().__init__(logdir=logdir, **kwargs)

        # Create repo if doesn't exist
        repo_url = create_repo(repo_id=repo_id, repo_type=repo_type, token=token, exist_ok=True, private=repo_private)
        self.repo_id = repo_url.repo_id
        print(f"Logs will be pushed to {repo_url}")

        # Set Hub-related attributes
        self.repo_type = repo_type
        self.repo_revision = repo_revision
        self.path_in_repo = path_in_repo
        self.token = token
        self.repo_allow_patterns = repo_allow_patterns
        self.repo_ignore_patterns = repo_ignore_patterns

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Push to hub in a non-blocking way when exiting the logger's context manager."""
        super().__exit__(exc_type, exc_val, exc_tb)
        future = self.push_to_hub(commit_message="Closing Tensorboard logger.")
        future.result()

    def push_to_hub(
        self, commit_message: Optional[str] = None, commit_description: Optional[str] = None
    ) -> Optional[Future[str]]:
        """
        Push the logs to the Hub asynchronously.

        Args:
            commit_message (`str`, *optional*):
                The summary / title / first line of the pushed commit. Defaults to "Upload training logs using HFSummaryWriter.".
            commit_description (`str`, *optional*):
                The description of the pushed commit.

        Returns:
            `Future[str]`: A future object that will yield the commit url when the upload is complete. Can be used to
            check the status of the upload. Returns None if `self.logdir` is an empty directory.
        """
        if not os.path.isdir(self.logdir):
            warnings.warn(f"Cannot push log to hub: {self.logdir} is not a directory.")
            return None

        return upload_folder(
            repo_id=self.repo_id,
            folder_path=self.logdir,
            path_in_repo=self.path_in_repo,
            commit_message=commit_message or "Upload training logs using HFSummaryWriter.",
            commit_description=commit_description,
            token=self.token,
            repo_type=self.repo_type,
            revision=self.repo_revision,
            allow_patterns=self.repo_allow_patterns,
            ignore_patterns=self.repo_ignore_patterns,
            run_as_future=True,
        )
