"""
Hot-reloading API types
"""

from dataclasses import dataclass
from typing import Literal, Optional, Union


@dataclass
class ReloadRegion:
    startLine: int
    startCol: int
    endLine: int
    endCol: int


@dataclass
class ReloadOperationObject:
    kind: Literal["add", "update", "delete"]
    region: ReloadRegion
    objectType: str
    objectName: str


@dataclass
class ReloadOperationRun:
    kind: Literal["run"]
    region: ReloadRegion
    codeLines: str
    stdout: Optional[str] = None
    stderr: Optional[str] = None


@dataclass
class ReloadOperationException:
    kind: Literal["exception"]
    region: ReloadRegion
    traceback: str


@dataclass
class ReloadOperationError:
    kind: Literal["error"]
    traceback: str


@dataclass
class ReloadOperationUI:
    kind: Literal["ui"]
    updated: bool


@dataclass
class ApiCreateReloadRequest:
    filepath: str
    contents: str
    reloadId: Optional[str] = None


@dataclass
class ApiCreateReloadResponseSuccess:
    status: Literal["created"]
    reloadId: str


@dataclass
class ApiCreateReloadResponseError:
    status: Literal["alreadyReloading", "fileNotFound"]


@dataclass
class ApiCreateReloadResponse:
    res: Union[ApiCreateReloadResponseError, ApiCreateReloadResponseSuccess]


@dataclass
class ApiGetReloadRequest:
    reloadId: str


@dataclass
class ApiGetReloadEventSourceData:
    data: Union[
        ReloadOperationError,
        ReloadOperationException,
        ReloadOperationObject,
        ReloadOperationRun,
        ReloadOperationUI,
    ]


@dataclass
class ApiGetStatusRequest:
    revision: str


@dataclass
class ApiGetStatusResponse:
    reloading: bool
    uncommited: list[str]


@dataclass
class ApiFetchContentsRequest:
    filepath: str


@dataclass
class ApiFetchContentsResponseError:
    status: Literal["fileNotFound"]


@dataclass
class ApiFetchContentsResponseSuccess:
    status: Literal["ok"]
    contents: str


@dataclass
class ApiFetchContentsResponse:
    res: Union[ApiFetchContentsResponseError, ApiFetchContentsResponseSuccess]
