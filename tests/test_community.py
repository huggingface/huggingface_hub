# Copyright 2026 The HuggingFace Team. All rights reserved.
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
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from huggingface_hub import HfApi
from huggingface_hub.community import (
    Discussion,
    DiscussionComment,
    DiscussionCommit,
    DiscussionEvent,
    DiscussionStatusChange,
    DiscussionTitleChange,
    DiscussionWithDetails,
    deserialize_event,
)


class TestCommunityAuthorNullDeserialization(unittest.TestCase):
    """Test suite verifying defensive deserialization when author is null/None in discussions and events."""

    def test_deserialize_comment_with_null_author(self) -> None:
        """Verify that deserialize_event correctly handles author=None when a user account was deleted."""
        raw_event = {
            "id": "645d00000000000000000001",
            "type": "comment",
            "createdAt": "2026-09-12T07:00:00.000Z",
            "author": None,
            "data": {
                "edited": False,
                "hidden": False,
                "latest": {
                    "raw": "Test comment with deleted author",
                    "html": "<p>Test comment with deleted author</p>",
                    "author": None,
                    "updatedAt": "2026-09-12T07:00:00.000Z",
                },
                "history": [],
            },
        }
        event = deserialize_event(raw_event)
        self.assertIsInstance(event, DiscussionComment)
        assert isinstance(event, DiscussionComment)
        self.assertEqual(event.author, "deleted")
        self.assertEqual(event.content, "Test comment with deleted author")
        self.assertEqual(event.rendered, "<p>Test comment with deleted author</p>")
        self.assertEqual(event.last_edited_by, "deleted")
        self.assertEqual(event.number_of_edits, 0)

    def test_comment_last_edited_by_null_author(self) -> None:
        """Verify last_edited_by returns 'deleted' when latest.author is None even if top-level author is present."""
        raw_event = {
            "id": "645d00000000000000000002",
            "type": "comment",
            "createdAt": "2026-09-12T07:00:00.000Z",
            "author": {"name": "active_user"},
            "data": {
                "edited": True,
                "hidden": False,
                "latest": {
                    "raw": "Edited comment",
                    "html": "<p>Edited comment</p>",
                    "author": None,
                    "updatedAt": "2026-09-12T07:05:00.000Z",
                },
                "history": [],
            },
        }
        event = deserialize_event(raw_event)
        self.assertIsInstance(event, DiscussionComment)
        assert isinstance(event, DiscussionComment)
        self.assertEqual(event.author, "active_user")
        self.assertEqual(event.last_edited_by, "deleted")

    def test_deserialize_status_change_with_null_author(self) -> None:
        """Verify deserialize_event on status-change with null author."""
        raw_event = {
            "id": "645d00000000000000000003",
            "type": "status-change",
            "createdAt": "2026-09-12T07:00:00.000Z",
            "author": None,
            "data": {"status": "closed"},
        }
        event = deserialize_event(raw_event)
        self.assertIsInstance(event, DiscussionStatusChange)
        assert isinstance(event, DiscussionStatusChange)
        self.assertEqual(event.author, "deleted")
        self.assertEqual(event.new_status, "closed")

    def test_deserialize_commit_with_null_author(self) -> None:
        """Verify deserialize_event on commit with null author."""
        raw_event = {
            "id": "645d00000000000000000004",
            "type": "commit",
            "createdAt": "2026-09-12T07:00:00.000Z",
            "author": None,
            "data": {
                "subject": "Initial commit",
                "oid": "abcdef0123456789abcdef0123456789abcdef01",
            },
        }
        event = deserialize_event(raw_event)
        self.assertIsInstance(event, DiscussionCommit)
        assert isinstance(event, DiscussionCommit)
        self.assertEqual(event.author, "deleted")
        self.assertEqual(event.summary, "Initial commit")
        self.assertEqual(event.oid, "abcdef0123456789abcdef0123456789abcdef01")

    def test_deserialize_title_change_with_null_author(self) -> None:
        """Verify deserialize_event on title-change with null author."""
        raw_event = {
            "id": "645d00000000000000000005",
            "type": "title-change",
            "createdAt": "2026-09-12T07:00:00.000Z",
            "author": None,
            "data": {
                "from": "Old title",
                "to": "New title",
            },
        }
        event = deserialize_event(raw_event)
        self.assertIsInstance(event, DiscussionTitleChange)
        assert isinstance(event, DiscussionTitleChange)
        self.assertEqual(event.author, "deleted")
        self.assertEqual(event.old_title, "Old title")
        self.assertEqual(event.new_title, "New title")

    def test_deserialize_generic_event_with_null_author(self) -> None:
        """Verify deserialize_event on generic unknown event type with null author."""
        raw_event = {
            "id": "645d00000000000000000006",
            "type": "custom-unhandled-event",
            "createdAt": "2026-09-12T07:00:00.000Z",
            "author": None,
        }
        event = deserialize_event(raw_event)
        self.assertIsInstance(event, DiscussionEvent)
        self.assertEqual(event.author, "deleted")

    def test_get_repo_discussions_with_null_author(self) -> None:
        """Verify get_repo_discussions deserializes discussions when author is None."""
        api = HfApi()
        mock_response = {
            "count": 1,
            "start": 0,
            "discussions": [
                {
                    "title": "Deleted user discussion",
                    "num": 42,
                    "author": None,
                    "createdAt": "2026-09-12T07:00:00.000Z",
                    "status": "open",
                    "repo": {"name": "user/repo", "type": "model"},
                    "isPullRequest": False,
                }
            ],
        }

        with patch("huggingface_hub.hf_api.get_session") as mock_get_session:
            mock_session = MagicMock()
            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_response
            mock_session.get.return_value = mock_resp
            mock_get_session.return_value = mock_session

            discussions = list(api.get_repo_discussions(repo_id="user/repo"))
            self.assertEqual(len(discussions), 1)
            disc = discussions[0]
            self.assertEqual(disc.author, "deleted")
            self.assertEqual(disc.num, 42)
            self.assertEqual(disc.title, "Deleted user discussion")
            self.assertEqual(disc.url, "https://huggingface.co/user/repo/discussions/42")

    def test_get_discussion_details_with_null_author(self) -> None:
        """Verify get_discussion_details deserializes discussion and nested events when author is None."""
        api = HfApi()
        mock_details = {
            "title": "Discussion with null author",
            "num": 1,
            "author": None,
            "createdAt": "2026-09-12T07:00:00.000Z",
            "status": "open",
            "repo": {"name": "user/repo", "type": "model"},
            "isPullRequest": False,
            "events": [
                {
                    "id": "645d00000000000000000001",
                    "type": "comment",
                    "createdAt": "2026-09-12T07:00:00.000Z",
                    "author": None,
                    "data": {
                        "edited": False,
                        "hidden": False,
                        "latest": {
                            "raw": "Hello world",
                            "html": "<p>Hello world</p>",
                            "author": None,
                            "updatedAt": "2026-09-12T07:00:00.000Z",
                        },
                        "history": [],
                    },
                }
            ],
        }

        with patch("huggingface_hub.hf_api.get_session") as mock_get_session:
            mock_session = MagicMock()
            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_details
            mock_session.get.return_value = mock_resp
            mock_get_session.return_value = mock_session

            details = api.get_discussion_details(repo_id="user/repo", discussion_num=1)
            self.assertIsInstance(details, DiscussionWithDetails)
            self.assertEqual(details.author, "deleted")
            self.assertEqual(len(details.events), 1)
            self.assertEqual(details.events[0].author, "deleted")


class TestCommunityNormalAuthor(unittest.TestCase):
    """Test standard behavior with normal (non-null) authors to prevent regressions."""

    def test_deserialize_comment_normal_author(self) -> None:
        raw_event = {
            "id": "645d00000000000000000010",
            "type": "comment",
            "createdAt": "2026-09-12T07:00:00.000Z",
            "author": {"name": "octocat"},
            "data": {
                "edited": False,
                "hidden": False,
                "latest": {
                    "raw": "Normal comment",
                    "html": "<p>Normal comment</p>",
                    "author": {"name": "octocat"},
                    "updatedAt": "2026-09-12T07:00:00.000Z",
                },
                "history": [],
            },
        }
        event = deserialize_event(raw_event)
        self.assertIsInstance(event, DiscussionComment)
        assert isinstance(event, DiscussionComment)
        self.assertEqual(event.author, "octocat")
        self.assertEqual(event.last_edited_by, "octocat")

    def test_discussion_dataclass_properties(self) -> None:
        created = datetime(2026, 9, 12, 7, 0, 0, tzinfo=timezone.utc)
        disc = Discussion(
            title="PR Title",
            status="open",
            num=123,
            repo_id="user/repo",
            repo_type="model",
            author="contributor",
            is_pull_request=True,
            created_at=created,
            endpoint="https://huggingface.co",
        )
        self.assertEqual(disc.git_reference, "refs/pr/123")
        self.assertEqual(disc.url, "https://huggingface.co/user/repo/discussions/123")

    def test_deserialize_event_missing_author_key(self) -> None:
        raw_event = {
            "id": "645d00000000000000000011",
            "type": "comment",
            "createdAt": "2026-09-12T07:00:00.000Z",
            "data": {
                "edited": False,
                "hidden": False,
                "latest": {
                    "raw": "Comment without author key",
                    "html": "<p>Comment without author key</p>",
                    "updatedAt": "2026-09-12T07:00:00.000Z",
                },
                "history": [],
            },
        }
        event = deserialize_event(raw_event)
        self.assertIsInstance(event, DiscussionComment)
        assert isinstance(event, DiscussionComment)
        self.assertEqual(event.author, "deleted")
        self.assertEqual(event.last_edited_by, "deleted")

    def test_deserialize_event_empty_author_dict(self) -> None:
        raw_event = {
            "id": "645d00000000000000000012",
            "type": "comment",
            "createdAt": "2026-09-12T07:00:00.000Z",
            "author": {},
            "data": {
                "edited": False,
                "hidden": False,
                "latest": {
                    "raw": "Comment with empty author dict",
                    "html": "<p>Comment with empty author dict</p>",
                    "author": {},
                    "updatedAt": "2026-09-12T07:00:00.000Z",
                },
                "history": [],
            },
        }
        event = deserialize_event(raw_event)
        self.assertIsInstance(event, DiscussionComment)
        assert isinstance(event, DiscussionComment)
        self.assertEqual(event.author, "deleted")
        self.assertEqual(event.last_edited_by, "deleted")

    def test_list_repo_commits_null_formatted(self) -> None:
        api = HfApi()
        mock_commit_item = {
            "id": "abcdef0123456789abcdef0123456789abcdef01",
            "authors": [{"user": "octocat"}],
            "date": "2026-09-12T07:00:00.000Z",
            "title": "Commit Title",
            "message": "Commit Message",
            "formatted": None,
        }

        with patch("huggingface_hub.hf_api.paginate") as mock_paginate:
            mock_paginate.return_value = iter([mock_commit_item])
            commits = api.list_repo_commits(repo_id="user/repo", formatted=True)
            self.assertEqual(len(commits), 1)
            commit = commits[0]
            self.assertEqual(commit.commit_id, "abcdef0123456789abcdef0123456789abcdef01")
            self.assertIsNone(commit.formatted_title)
            self.assertIsNone(commit.formatted_message)
