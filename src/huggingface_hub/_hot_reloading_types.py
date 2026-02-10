"""
Hot-reloading API types
"""

from typing import Literal, Optional, Union

from pydantic import BaseModel


class ReloadRegion(BaseModel):
    startLine: int
    startCol: int
    endLine: int
    endCol: int


class ReloadOperationObject(BaseModel):
    kind: Literal["add", "update", "delete"]
    region: ReloadRegion
    objectType: str
    objectName: str


class ReloadOperationRun(BaseModel):
    kind: Literal["run"]
    region: ReloadRegion
    codeLines: str
    stdout: Optional[str] = None
    stderr: Optional[str] = None


class ReloadOperationException(BaseModel):
    kind: Literal["exception"]
    region: ReloadRegion
    traceback: str


class ReloadOperationError(BaseModel):
    kind: Literal["error"]
    traceback: str


class ReloadOperationUI(BaseModel):
    kind: Literal["ui"]
    updated: bool


class ApiCreateReloadRequest(BaseModel):
    filepath: str
    contents: str
    reloadId: Optional[str] = None


class ApiCreateReloadResponseSuccess(BaseModel):
    status: Literal["created"]
    reloadId: str


class ApiCreateReloadResponseError(BaseModel):
    status: Literal["alreadyReloading", "fileNotFound"]


class ApiCreateReloadResponse(BaseModel):
    res: Union[ApiCreateReloadResponseError, ApiCreateReloadResponseSuccess]


class ApiGetReloadRequest(BaseModel):
    reloadId: str


class ApiGetReloadEventSourceData(BaseModel):
    data: Union[
        ReloadOperationError,
        ReloadOperationException,
        ReloadOperationObject,
        ReloadOperationRun,
        ReloadOperationUI,
    ]


class ApiGetStatusRequest(BaseModel):
    revision: str


class ApiGetStatusResponse(BaseModel):
    reloading: bool
    uncommited: list[str]


class ApiFetchContentsRequest(BaseModel):
    filepath: str


class ApiFetchContentsResponseError(BaseModel):
    status: Literal["fileNotFound"]


class ApiFetchContentsResponseSuccess(BaseModel):
    status: Literal["ok"]
    contents: str


class ApiFetchContentsResponse(BaseModel):
    res: Union[ApiFetchContentsResponseError, ApiFetchContentsResponseSuccess]
