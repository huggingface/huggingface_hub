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
"""Contains a Mixin to push training logs to the Hub."""
import os
from concurrent.futures import Future
from typing import List, Optional, Union

from .hf_api import create_repo, upload_folder


class HFLoggerMixin:
    """
    Mixin class to add `push_to_hub` method to a logger class.

    The idea is to log data locally and then push it to the Hub only when requested (e.g. once every few epochs).
    This Mixin class is not specific to any logger class as long as it implements a `log_dir` property to return the
    current experiment directory. In practice, it is compatible with lightning loggers (e.g. TensorBoardLogger) when
    data is saved locally.

    The `push_to_hub` method is asynchronous, meaning the upload action will be executed in a separate thread.
    Warning: you must be careful not to modify the log files while they are being pushed. Make sure to push to the Hub
    only when your script is not logging data anymore (e.g. at the end of an epoch).

    Example:
        ```py
        from tensorboardX import SummaryWriter

        class HFTensorBoardLogger(HFLoggerMixin, SummaryWriter):
            @property
            def log_dir(self) -> str:
                return self.logdir

        logger = HFTensorBoardLogger("training/results", repo_id="test_hf_logger", path_in_repo="tensorboard")
        logger.log_hyperparams({"a": 1, "b": 2})
        logger.push_to_hub()
        ```

        The order of the parent classes is important!
        ```py
        # Invalid order: will fail at initialization!
        class HFLogger(TensorBoardLogger, HFLoggerMixin):
            pass

        # Valid order: both parent classes are initialized correctly
        class HFLogger(HFLoggerMixin, TensorBoardLogger):
            pass
        ```
    """

    _hf_logger_mixin_initialized = False

    def __init__(
        self,
        *args,
        repo_id: str,
        repo_type: Optional[str] = None,
        repo_revision: Optional[str] = None,
        repo_private: bool = False,
        path_in_repo: Optional[str] = None,
        hf_token: Optional[str] = None,
        repo_allow_patterns: Optional[Union[List[str], str]] = None,
        repo_ignore_patterns: Optional[Union[List[str], str]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Check that the base logger class implements a `log_dir` property
        # In practice, it's compatible with Pytorch lightning loggers.
        if not hasattr(self, "log_dir"):
            raise NotImplementedError("HFLoggerMixin must be used with a Logger class that implements `log_dir`.")

        # Fail early if log_dir data is not saved locally
        log_dir = getattr(self, "log_dir", None)
        if not isinstance(log_dir, str):
            raise ValueError(
                f"Expected `log_dir` to be a string, but got {type(log_dir)} ({log_dir}). This can be the case if the"
                " logger doesn't save data locally."
            )

        # Create repo if doesn't exist
        repo_url = create_repo(
            repo_id=repo_id, repo_type=repo_type, token=hf_token, exist_ok=True, private=repo_private
        )
        self.repo_id = repo_url.repo_id
        print(f"Logs will be pushed to {repo_url}")

        # Set attributes for `push_to_hub`
        self.repo_type = repo_type
        self.repo_revision = repo_revision
        self.path_in_repo = path_in_repo
        self.hf_token = hf_token
        self.repo_allow_patterns = repo_allow_patterns
        self.repo_ignore_patterns = repo_ignore_patterns

        # Set flag that this mixin has been initialized
        # Will be False if the logger class is badly defined (e.g. HFLoggerMixin.__init__ is not called)
        self._hf_logger_mixin_initialized = True

    def push_to_hub(
        self, commit_message: Optional[str] = None, commit_description: Optional[str] = None
    ) -> Future[str]:
        if not self._hf_logger_mixin_initialized:
            raise RuntimeError(
                "`HFLoggerMixin` must be initialized before calling `push_to_hub` but is not. This is due to"
                " `HFLoggerMixin.__init__` not being called. You must define a logger class that inherits from"
                " `HFLoggerMixin` first (e.g. `class HFLogger(HFLoggerMixin, TensorBoardLogger)`) or call"
                " `super().__init__(*args, **kwargs)`."
            )

        log_dir = getattr(self, "log_dir", None)
        if not isinstance(log_dir, str):
            raise ValueError(
                f"Expected `logger.log_dir` to be a string, but got {type(log_dir)} ({log_dir}). This can be the case"
                " if the logger doesn't save data locally."
            )
        if not os.path.isdir(log_dir):
            raise ValueError(
                f"Expected `logger.log_dir` ({log_dir}) to be a path to a local dir but is not. Data cannot be pushed"
                " to the hub."
            )

        return upload_folder(
            repo_id=self.repo_id,
            folder_path=log_dir,
            path_in_repo=self.path_in_repo,
            commit_message=commit_message,
            commit_description=commit_description,
            token=self.hf_token,
            repo_type=self.repo_type,
            revision=self.repo_revision,
            allow_patterns=self.repo_allow_patterns,
            ignore_patterns=self.repo_ignore_patterns,
            run_as_future=True,
        )
