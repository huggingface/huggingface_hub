#!/usr/bin/env python
# coding=utf-8
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
# limitations under the License

import subprocess

from .logging import get_logger


logger = get_logger(__name__)


def run_subprocess(command, folder, check=True) -> subprocess.CompletedProcess:
    if isinstance(command, str):
        logger.error("`run_subprocess` should be called with a list of strings.")
        command = command.split()

    return subprocess.run(
        command,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        check=check,
        encoding="utf-8",
        cwd=folder,
    )
