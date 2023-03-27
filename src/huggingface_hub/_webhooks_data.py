from typing import Optional

from pydantic import BaseModel

from .utils._typing import Literal


class WebhookPayloadEvent(BaseModel):
    action: Literal["create", "update", "delete"]
    scope: str


class WebhookPayloadRepo(BaseModel):
    type: Literal["dataset", "model", "space"]
    name: str
    private: bool


class WebhookPayloadDiscussion(BaseModel):
    num: int
    isPullRequest: bool
    status: Literal["open", "closed", "merged"]


class WebhookPayload(BaseModel):
    event: WebhookPayloadEvent
    repo: WebhookPayloadRepo
    discussion: Optional[WebhookPayloadDiscussion]
