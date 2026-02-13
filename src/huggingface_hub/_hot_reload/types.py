"""
Hot-reloading API types
"""

from typing import Literal, TypedDict, Union

from typing_extensions import NotRequired


class ReloadRegion(TypedDict):
    startLine: int
    startCol: int
    endLine: int
    endCol: int


class ReloadOperationObject(TypedDict):
    kind: Literal["add", "update", "delete"]
    region: ReloadRegion
    objectType: str
    objectName: str


class ReloadOperationRun(TypedDict):
    kind: Literal["run"]
    region: ReloadRegion
    codeLines: str
    stdout: NotRequired[str]
    stderr: NotRequired[str]


class ReloadOperationException(TypedDict):
    kind: Literal["exception"]
    region: ReloadRegion
    traceback: str


class ReloadOperationError(TypedDict):
    kind: Literal["error"]
    traceback: str


class ReloadOperationUI(TypedDict):
    kind: Literal["ui"]
    updated: bool


class ApiCreateReloadRequest(TypedDict):
    filepath: str
    contents: str
    reloadId: NotRequired[str]


class ApiCreateReloadResponseSuccess(TypedDict):
    status: Literal["created"]
    reloadId: str


class ApiCreateReloadResponseError(TypedDict):
    status: Literal["alreadyReloading", "fileNotFound"]


class ApiCreateReloadResponse(TypedDict):
    res: Union[ApiCreateReloadResponseError, ApiCreateReloadResponseSuccess]


class ApiGetReloadRequest(TypedDict):
    reloadId: str


class ApiGetReloadEventSourceData(TypedDict):
    data: Union[
        ReloadOperationError,
        ReloadOperationException,
        ReloadOperationObject,
        ReloadOperationRun,
        ReloadOperationUI,
    ]


class ApiGetStatusRequest(TypedDict):
    revision: str


class ApiGetStatusResponse(TypedDict):
    reloading: bool
    uncommited: list[str]


class ApiFetchContentsRequest(TypedDict):
    filepath: str


class ApiFetchContentsResponseError(TypedDict):
    status: Literal["fileNotFound"]


class ApiFetchContentsResponseSuccess(TypedDict):
    status: Literal["ok"]
    contents: str


class ApiFetchContentsResponse(TypedDict):
    res: Union[ApiFetchContentsResponseError, ApiFetchContentsResponseSuccess]
