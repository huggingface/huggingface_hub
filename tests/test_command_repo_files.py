import unittest
from argparse import ArgumentParser
from unittest.mock import Mock, patch

from huggingface_hub.commands.repo_files import DeleteFilesSubCommand, RepoFilesCommand

from .testing_utils import DUMMY_MODEL_ID


class TestRepoFilesCommand(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up CLI as in `src/huggingface_hub/commands/huggingface_cli.py`.
        """
        self.parser = ArgumentParser("huggingface-cli", usage="huggingface-cli <command> [<args>]")
        commands_parser = self.parser.add_subparsers()
        RepoFilesCommand.register_subcommand(commands_parser)

    @patch("huggingface_hub.commands.repo_files.HfApi.delete_files_r")
    def test_delete(self, delete_files_r_mock: Mock) -> None:
        fixtures = [
            {
                "input_args": [
                    "repo-files",
                    DUMMY_MODEL_ID,
                    "delete",
                    "*",
                ],
                "delete_files_r_args": {
                    "patterns": [
                        "*",
                    ],
                    "repo_id": DUMMY_MODEL_ID,
                    "repo_type": "model",
                    "revision": None,
                },
            },
            {
                "input_args": [
                    "repo-files",
                    DUMMY_MODEL_ID,
                    "delete",
                    "file.txt",
                ],
                "delete_files_r_args": {
                    "patterns": [
                        "file.txt",
                    ],
                    "repo_id": DUMMY_MODEL_ID,
                    "repo_type": "model",
                    "revision": None,
                },
            },
            {
                "input_args": [
                    "repo-files",
                    DUMMY_MODEL_ID,
                    "delete",
                    "folder/",
                ],
                "delete_files_r_args": {
                    "patterns": [
                        "folder/",
                    ],
                    "repo_id": DUMMY_MODEL_ID,
                    "repo_type": "model",
                    "revision": None,
                },
            },
            {
                "input_args": [
                    "repo-files",
                    DUMMY_MODEL_ID,
                    "delete",
                    "file1.txt",
                    "folder/",
                    "file2.txt",
                ],
                "delete_files_r_args": {
                    "patterns": [
                        "file1.txt",
                        "folder/",
                        "file2.txt",
                    ],
                    "repo_id": DUMMY_MODEL_ID,
                    "repo_type": "model",
                    "revision": None,
                },
            },
            {
                "input_args": [
                    "repo-files",
                    DUMMY_MODEL_ID,
                    "delete",
                    "file.txt *",
                    "*.json",
                    "folder/*.parquet",
                ],
                "delete_files_r_args": {
                    "patterns": [
                        "file.txt *",
                        "*.json",
                        "folder/*.parquet",
                    ],
                    "repo_id": DUMMY_MODEL_ID,
                    "repo_type": "model",
                    "revision": None,
                },
            },
            {
                "input_args": [
                    "repo-files",
                    DUMMY_MODEL_ID,
                    "delete",
                    "file.txt *",
                    "--revision",
                    "test_revision",
                    "--repo-type",
                    "dataset",
                ],
                "delete_files_r_args": {
                    "patterns": [
                        "file.txt *",
                    ],
                    "repo_id": DUMMY_MODEL_ID,
                    "repo_type": "dataset",
                    "revision": "test_revision",
                },
            },
        ]

        for expected in fixtures:
            # subTest is similar to pytest.mark.parametrize, but using the unittest
            # framework
            with self.subTest(expected):
                delete_files_r_args = expected["delete_files_r_args"]

                cmd = DeleteFilesSubCommand(self.parser.parse_args(expected["input_args"]))
                cmd.run()

                if delete_files_r_args is None:
                    assert delete_files_r_mock.call_count == 0
                else:
                    assert delete_files_r_mock.call_count == 1
                    # Inspect the captured calls
                    _, kwargs = delete_files_r_mock.call_args_list[0]
                    assert kwargs == delete_files_r_args

                delete_files_r_mock.reset_mock()
