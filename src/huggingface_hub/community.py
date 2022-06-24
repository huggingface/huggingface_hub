from dataclasses import dataclass
from datetime import datetime
from hashlib import new
from typing import List, Optional

from dateutil.parser import parse as parse_datetime


@dataclass
class Discussion:
    title: str
    status: str
    num: int
    repo_id: str
    repo_type: str
    author: str
    is_pull_request: bool
    created_at: datetime


@dataclass
class DiscussionWithDetails(Discussion):
    events: List["DiscussionEvent"]
    conflicting_files: Optional[List[str]]
    target_branch: Optional[str]
    merge_commit_oid: Optional[str]


@dataclass
class DiscussionEvent:
    id: str
    type: str
    created_at: datetime
    author: str


@dataclass
class DiscussionComment(DiscussionEvent):
    edited: bool
    hidden: bool
    content: str


@dataclass
class DiscussionStatusChange(DiscussionEvent):
    new_status: str


@dataclass
class DiscussionCommit(DiscussionEvent):
    summary: str
    oid: str


@dataclass
class DiscussionTitleChange(DiscussionEvent):
    old_title: str
    new_title: str


def deserialize_event(event: dict) -> DiscussionEvent:
    event_id: str = event["id"]
    event_type: str = event["type"]
    created_at = parse_datetime(event["createdAt"])

    common_args = dict(
        id=event_id,
        type=event_type,
        created_at=created_at,
        author=event.get("author", {}).get("name", "deleted"),
    )

    if event_type == "comment":
        return DiscussionComment(
            **common_args,
            edited=event["data"]["edited"],
            hidden=event["data"]["hidden"],
            content=event["data"]["latest"]["raw"],
        )
    if event_type == "status-change":
        return DiscussionStatusChange(
            **common_args,
            new_status=event["data"]["status"],
        )
    if event_type == "commit":
        return DiscussionCommit(
            **common_args,
            summary=event["data"]["subject"],
            oid=event["data"]["oid"],
        )
    if event_type == "title-change":
        return DiscussionTitleChange(
            **common_args,
            old_title=event["data"]["from"],
            new_title=event["data"]["to"],
        )

    return DiscussionEvent(**common_args)
