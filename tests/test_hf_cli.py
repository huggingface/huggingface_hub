import unittest
from argparse import ArgumentParser
from unittest.mock import Mock, patch

from huggingface_hub.cli.auth import AuthCommand
from huggingface_hub.cli.cache import CacheCommand
from huggingface_hub.cli.files import FilesCommand
from huggingface_hub.cli.hf_cli import main
from huggingface_hub.cli.repo import RepoCommand
from huggingface_hub.cli.utils import UtilsCommand


class TestHfCLI(unittest.TestCase):
    def setUp(self) -> None:
        """Set up the hf CLI parser as in hf_cli.py."""
        self.parser = ArgumentParser("hf", usage="hf <command> [<args>]")
        commands_parser = self.parser.add_subparsers(help="hf command helpers")
        
        # Register commands
        AuthCommand.register_subcommand(commands_parser)
        RepoCommand.register_subcommand(commands_parser)
        FilesCommand.register_subcommand(commands_parser)
        CacheCommand.register_subcommand(commands_parser)
        UtilsCommand.register_subcommand(commands_parser)

    def test_auth_login_parsing(self):
        """Test parsing of hf auth login command."""
        args = self.parser.parse_args(["auth", "login"])
        self.assertEqual(args.token, None)
        self.assertEqual(args.add_to_git_credential, False)

    def test_auth_login_with_token(self):
        """Test parsing of hf auth login with token."""
        args = self.parser.parse_args(["auth", "login", "--token", "hf_test", "--add-to-git-credential"])
        self.assertEqual(args.token, "hf_test")
        self.assertEqual(args.add_to_git_credential, True)

    def test_auth_logout_parsing(self):
        """Test parsing of hf auth logout command."""
        args = self.parser.parse_args(["auth", "logout"])
        self.assertEqual(args.token_name, None)

    def test_auth_logout_with_token_name(self):
        """Test parsing of hf auth logout with token name."""
        args = self.parser.parse_args(["auth", "logout", "--token-name", "test_token"])
        self.assertEqual(args.token_name, "test_token")

    def test_auth_switch_parsing(self):
        """Test parsing of hf auth switch command."""
        args = self.parser.parse_args(["auth", "switch"])
        self.assertEqual(args.token_name, None)
        self.assertEqual(args.add_to_git_credential, False)

    def test_auth_list_parsing(self):
        """Test parsing of hf auth list command."""
        args = self.parser.parse_args(["auth", "list"])
        self.assertTrue(hasattr(args, "func"))

    def test_auth_whoami_parsing(self):
        """Test parsing of hf auth whoami command."""
        args = self.parser.parse_args(["auth", "whoami"])
        self.assertTrue(hasattr(args, "func"))

    def test_files_download_parsing(self):
        """Test parsing of hf files download command."""
        args = self.parser.parse_args(["files", "download", "gpt2", "config.json"])
        self.assertEqual(args.repo_id, "gpt2")
        self.assertEqual(args.filename, "config.json")
        self.assertEqual(args.repo_type, "model")

    def test_files_upload_parsing(self):
        """Test parsing of hf files upload command."""
        args = self.parser.parse_args(["files", "upload", "my-model", "./config.json"])
        self.assertEqual(args.repo_id, "my-model")
        self.assertEqual(args.local_path, "./config.json")
        self.assertEqual(args.repo_type, "model")

    def test_files_delete_parsing(self):
        """Test parsing of hf files delete command."""
        args = self.parser.parse_args(["files", "delete", "my-model", "config.json"])
        self.assertEqual(args.repo_id, "my-model")
        self.assertEqual(args.path_in_repo, "config.json")

    def test_cache_scan_parsing(self):
        """Test parsing of hf cache scan command."""
        args = self.parser.parse_args(["cache", "scan"])
        self.assertEqual(args.dir, None)
        self.assertEqual(args.verbose, 0)

    def test_cache_scan_verbose(self):
        """Test parsing of hf cache scan with verbose."""
        args = self.parser.parse_args(["cache", "scan", "-v"])
        self.assertEqual(args.verbose, 1)

    def test_cache_delete_parsing(self):
        """Test parsing of hf cache delete command."""
        args = self.parser.parse_args(["cache", "delete"])
        self.assertEqual(args.dir, None)
        self.assertEqual(args.verbose, 0)
        self.assertEqual(args.disable_tui, False)
        self.assertEqual(args.yes, False)

    def test_repo_create_parsing(self):
        """Test parsing of hf repo create command."""
        args = self.parser.parse_args(["repo", "create", "my-model"])
        self.assertEqual(args.name, "my-model")
        self.assertEqual(args.type, "model")
        self.assertEqual(args.private, False)

    def test_repo_tag_create_parsing(self):
        """Test parsing of hf repo tag create command."""
        args = self.parser.parse_args(["repo", "tag", "create", "my-model", "v1.0"])
        self.assertEqual(args.repo_id, "my-model")
        self.assertEqual(args.tag, "v1.0")
        self.assertEqual(args.repo_type, "model")

    def test_version_parsing(self):
        """Test parsing of hf version command."""
        args = self.parser.parse_args(["version"])
        self.assertTrue(hasattr(args, "func"))

    def test_env_parsing(self):
        """Test parsing of hf env command."""
        args = self.parser.parse_args(["env"])
        self.assertTrue(hasattr(args, "func"))

    def test_download_alias_parsing(self):
        """Test parsing of hf download alias."""
        args = self.parser.parse_args(["download", "gpt2", "config.json"])
        self.assertEqual(args.repo_id, "gpt2")
        self.assertEqual(args.filename, "config.json")

    def test_upload_alias_parsing(self):
        """Test parsing of hf upload alias."""
        args = self.parser.parse_args(["upload", "my-model", "./config.json"])
        self.assertEqual(args.repo_id, "my-model")
        self.assertEqual(args.local_path, "./config.json")


if __name__ == "__main__":
    unittest.main()