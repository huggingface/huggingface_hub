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
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from huggingface_hub import constants
from huggingface_hub._space_api import SpaceHardware
from huggingface_hub.utils._datetime import parse_datetime


class JobStage(str, Enum):
    """
    Enumeration of possible stage of a Job on the Hub.

    Value can be compared to a string:
    ```py
    assert JobStage.COMPLETED == "COMPLETED"
    ```

    Taken from https://github.com/huggingface/moon-landing/blob/main/server/job_types/JobInfo.ts#L61 (private url).
    """

    # Copied from moon-landing > server > lib > Job.ts
    COMPLETED = "COMPLETED"
    CANCELED = "CANCELED"
    ERROR = "ERROR"
    DELETED = "DELETED"
    RUNNING = "RUNNING"


@dataclass
class JobStatus:
    stage: JobStage
    message: Optional[str]

    def __init__(self, **kwargs) -> None:
        self.stage = kwargs["stage"]
        self.message = kwargs.get("message")


@dataclass
class JobOwner:
    id: str
    name: str


@dataclass
class JobInfo:
    id: str
    created_at: Optional[datetime]
    docker_image: Optional[str]
    space_id: Optional[str]
    command: Optional[List[str]]
    arguments: Optional[List[str]]
    environment: Optional[Dict[str, Any]]
    secrets: Optional[Dict[str, Any]]
    flavor: Optional[SpaceHardware]
    status: Optional[JobStatus]
    owner: Optional[JobOwner]

    # Inferred fields
    endpoint: str
    url: str

    def __init__(self, **kwargs) -> None:
        self.id = kwargs["id"]
        created_at = kwargs.get("createdAt") or kwargs.get("created_at")
        self.created_at = parse_datetime(created_at) if created_at else None
        self.docker_image = kwargs.get("dockerImage") or kwargs.get("docker_image")
        self.space_id = kwargs.get("spaceId") or kwargs.get("space_id")
        self.owner = JobOwner(**(kwargs["owner"] if isinstance(kwargs.get("owner"), dict) else {}))
        self.command = kwargs.get("command")
        self.arguments = kwargs.get("arguments")
        self.environment = kwargs.get("environment")
        self.secrets = kwargs.get("secrets")
        self.flavor = kwargs.get("flavor")
        self.status = JobStatus(**(kwargs["status"] if isinstance(kwargs.get("status"), dict) else {}))

        # Inferred fields
        self.endpoint = kwargs.get("endpoint", constants.ENDPOINT)
        self.url = f"{self.endpoint}/jobs/{self.owner.id}/{self.id}"
