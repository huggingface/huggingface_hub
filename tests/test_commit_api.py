import json

import httpx2
import pytest

from huggingface_hub import HfApi
from huggingface_hub._commit_api import (
    CommitOperationAdd,
    CommitOperationCopy,
    CommitOperationDelete,
    _warn_on_overwriting_operations,
)
from huggingface_hub.utils import set_client_factory
from huggingface_hub.utils._http import default_client_factory
from huggingface_hub.utils.sha import git_hash


class TestCommitOperationDelete:
    def test_implicit_file(self):
        assert not CommitOperationDelete(path_in_repo="path/to/file").is_folder
        assert not CommitOperationDelete(path_in_repo="path/to/file.md").is_folder

    def test_implicit_folder(self):
        assert CommitOperationDelete(path_in_repo="path/to/folder/").is_folder
        assert CommitOperationDelete(path_in_repo="path/to/folder.md/").is_folder

    def test_explicit_file(self):
        # Weird case: if user explicitly set as file (`is_folder`=False) but path has a
        # trailing "/" => user input has priority
        assert not CommitOperationDelete(path_in_repo="path/to/folder/", is_folder=False).is_folder
        assert not CommitOperationDelete(path_in_repo="path/to/folder.md/", is_folder=False).is_folder

    def test_explicit_folder(self):
        # No need for the trailing "/" is `is_folder` explicitly passed
        assert CommitOperationDelete(path_in_repo="path/to/folder", is_folder=True).is_folder
        assert CommitOperationDelete(path_in_repo="path/to/folder.md", is_folder=True).is_folder

    def test_is_folder_wrong_value(self):
        with pytest.raises(ValueError):
            CommitOperationDelete(path_in_repo="path/to/folder", is_folder="any value")


class TestCommitOperationPathInRepo:
    valid_values = {  # key is input, value is expected validated output
        "file.txt": "file.txt",
        ".file.txt": ".file.txt",
        "/file.txt": "file.txt",
        "./file.txt": "file.txt",
    }
    invalid_values = [
        ".",
        "..",
        "../file.txt",
        # a ".." segment anywhere in the path is rejected, whether or not it would
        # resolve back inside the repo root
        "a/../file.txt",
        "a/b/../../c/file.txt",
        "a/../../file.txt",
        "a/b/../../../file.txt",
    ]

    def test_path_in_repo_valid(self) -> None:
        for input, expected in self.valid_values.items():
            assert CommitOperationAdd(path_in_repo=input, path_or_fileobj=b"").path_in_repo == expected
            assert CommitOperationDelete(path_in_repo=input).path_in_repo == expected

    def test_path_in_repo_invalid(self) -> None:
        for input in self.invalid_values:
            with pytest.raises(ValueError):
                CommitOperationAdd(path_in_repo=input, path_or_fileobj=b"")
            with pytest.raises(ValueError):
                CommitOperationDelete(path_in_repo=input)


class TestCommitOperationForbiddenPathInRepo:
    """Commit operations must throw an error on files in the .git/ or .cache/huggingface/ folders.

    Server would error anyway so it's best to prevent early.
    """

    INVALID_PATHS_IN_REPO = {
        ".git",
        ".git/path/to/file",
        "./.git/path/to/file",
        "subfolder/path/.git/to/file",
        "./subfolder/path/.git/to/file",
        ".cache/huggingface",
        "./.cache/huggingface/path/to/file",
        "./subfolder/path/.cache/huggingface/to/file",
    }

    VALID_PATHS_IN_REPO = {
        ".gitignore",
        "path/to/.gitignore",
        "path/to/something.git",
        "path/to/something.git/more",
        "path/to/something.huggingface/more",
        "huggingface",
        ".huggingface",
        "./.huggingface/path/to/file",
        "./subfolder/path/huggingface/to/file",
        "./subfolder/path/.huggingface/to/file",
    }

    def test_cannot_update_file_in_git_folder(self):
        for path in self.INVALID_PATHS_IN_REPO:
            with pytest.raises(ValueError):
                CommitOperationAdd(path_in_repo=path, path_or_fileobj=b"content")
            with pytest.raises(ValueError):
                CommitOperationDelete(path_in_repo=path)

    def test_valid_path_in_repo_containing_git(self):
        for path in self.VALID_PATHS_IN_REPO:
            CommitOperationAdd(path_in_repo=path, path_or_fileobj=b"content")
            CommitOperationDelete(path_in_repo=path)


class TestWarnOnOverwritingOperations:
    add_file_ab = CommitOperationAdd(path_in_repo="a/b.txt", path_or_fileobj=b"data")
    add_file_abc = CommitOperationAdd(path_in_repo="a/b/c.md", path_or_fileobj=b"data")
    add_file_abd = CommitOperationAdd(path_in_repo="a/b/d.md", path_or_fileobj=b"data")
    update_file_abc = CommitOperationAdd(path_in_repo="a/b/c.md", path_or_fileobj=b"updated data")
    delete_file_abc = CommitOperationDelete(path_in_repo="a/b/c.md")
    delete_folder_a = CommitOperationDelete(path_in_repo="a/")
    delete_folder_e = CommitOperationDelete(path_in_repo="e/")

    def test_no_overwrite(self) -> None:
        _warn_on_overwriting_operations(
            [
                self.add_file_ab,
                self.add_file_abc,
                self.add_file_abd,
                self.delete_folder_e,
            ]
        )

    def test_add_then_update_file(self) -> None:
        with pytest.warns(UserWarning):
            _warn_on_overwriting_operations([self.add_file_abc, self.update_file_abc])

    def test_add_then_delete_file(self) -> None:
        with pytest.warns(UserWarning):
            _warn_on_overwriting_operations([self.add_file_abc, self.delete_file_abc])

    def test_add_then_delete_folder(self) -> None:
        with pytest.warns(UserWarning):
            _warn_on_overwriting_operations([self.add_file_abc, self.delete_folder_a])

        with pytest.warns(UserWarning):
            _warn_on_overwriting_operations([self.add_file_ab, self.delete_folder_a])

    def test_delete_file_then_add(self) -> None:
        _warn_on_overwriting_operations([self.delete_file_abc, self.add_file_abc])

    def test_delete_folder_then_add(self) -> None:
        _warn_on_overwriting_operations([self.delete_folder_a, self.add_file_ab, self.add_file_abc])


class TestCreateCommitNoOpFilter:
    def test_readd_unchanged_file_after_delete_not_dropped(self) -> None:
        content = b"keep me\n"
        sent_commits = []

        def handler(request: httpx2.Request) -> httpx2.Response:
            path = request.url.path
            if "/paths-info/" in path:
                return httpx2.Response(
                    200,
                    json=[
                        {"path": "src.txt", "type": "file", "oid": git_hash(content), "size": len(content)},
                        {"path": "x/copy.txt", "type": "file", "oid": git_hash(content), "size": len(content)},
                        {"path": "unrelated_copy.txt", "type": "file", "oid": git_hash(content), "size": len(content)},
                    ],
                )
            if "/preupload/" in path:
                files = json.loads(request.content)["files"]
                return httpx2.Response(
                    200,
                    json={
                        "files": [
                            {
                                "path": f["path"],
                                "uploadMode": "regular",
                                "shouldIgnore": False,
                                "oid": git_hash(content),
                            }
                            for f in files
                        ]
                    },
                )
            if "/resolve/" in path:
                return httpx2.Response(200, content=content)
            if "/commit/" in path:
                lines = [json.loads(line) for line in request.content.decode().splitlines() if line]
                sent_commits.append([line for line in lines if line["key"] != "header"])
                return httpx2.Response(
                    200,
                    json={"commitUrl": "https://huggingface.co/u/r/commit/" + "a" * 40, "commitOid": "a" * 40},
                )
            return httpx2.Response(404, json={"error": f"unexpected {request.method} {path}"})

        client = httpx2.Client(transport=httpx2.MockTransport(handler))
        set_client_factory(lambda: client)
        try:
            api = HfApi(endpoint="https://huggingface.co", token="hf_fake")

            # Case 1: Folder delete + unchanged add under that folder
            sent_commits.clear()
            api.create_commit(
                "u/r",
                operations=[
                    CommitOperationDelete("x/", is_folder=True),
                    CommitOperationAdd("x/a.txt", content),
                ],
                commit_message="folder delete + add",
            )
            sent = [(op["key"], op["value"]["path"]) for op in sent_commits[0]]
            assert sent == [("deletedFolder", "x/"), ("file", "x/a.txt")]

            # Case 2: File delete + unchanged add of same file
            sent_commits.clear()
            api.create_commit(
                "u/r",
                operations=[
                    CommitOperationDelete("x/a.txt"),
                    CommitOperationAdd("x/a.txt", content),
                ],
                commit_message="file delete + add",
            )
            sent = [(op["key"], op["value"]["path"]) for op in sent_commits[0]]
            assert sent == [("deletedFile", "x/a.txt"), ("file", "x/a.txt")]

            # Case 3: Folder delete + unchanged copy into that folder
            sent_commits.clear()
            api.create_commit(
                "u/r",
                operations=[
                    CommitOperationDelete("x/", is_folder=True),
                    CommitOperationCopy("src.txt", "x/copy.txt"),
                    CommitOperationCopy("src.txt", "unrelated_copy.txt"),
                ],
                commit_message="folder delete + copy",
            )
            sent = [(op["key"], op["value"]["path"]) for op in sent_commits[0]]
            assert sent == [("deletedFolder", "x/"), ("file", "x/copy.txt")]

            # Case 4: Unchanged add without delete is still dropped as no-op
            sent_commits.clear()
            api.create_commit(
                "u/r",
                operations=[
                    CommitOperationDelete("x/", is_folder=True),
                    CommitOperationAdd("x/a.txt", content),
                    CommitOperationAdd("unrelated.txt", content),
                ],
                commit_message="mixed",
            )
            sent = [(op["key"], op["value"]["path"]) for op in sent_commits[0]]
            assert sent == [("deletedFolder", "x/"), ("file", "x/a.txt")]
        finally:
            set_client_factory(default_client_factory)
