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

import os
import subprocess
import sys


LFS_MULTIPART_UPLOAD_COMMAND = "lfs-multipart-upload"


def install_lfs_in_userspace():
    """
    If in Linux, installs git-lfs in userspace
    (sometimes useful if you can't `sudo apt install` or equivalent).
    """
    if sys.platform != "linux":
        raise ValueError("Only implemented for Linux right now")
    GIT_LFS_TARBALL = "https://github.com/git-lfs/git-lfs/releases/download/v2.13.1/git-lfs-linux-amd64-v2.13.1.tar.gz"
    CWD = os.path.join(os.getcwd(), "install_lfs")
    os.makedirs(CWD, exist_ok=True)
    subprocess.run(
        ["wget", "-O", "tarball.tar.gz", GIT_LFS_TARBALL], check=True, cwd=CWD
    )
    subprocess.run(["tar", "-xvzf", "tarball.tar.gz"], check=True, cwd=CWD)
    subprocess.run(["bash", "install.sh"], check=True, cwd=CWD)
