import os
import unittest
from pathlib import Path
from tempfile import mkstemp
from unittest.mock import Mock, patch

from InquirerPy.base.control import Choice
from InquirerPy.separator import Separator

from huggingface_hub.commands.delete_cache import (
    _CANCEL_DELETION_STR,
    DeleteCacheCommand,
    _ask_for_confirmation_no_tui,
    _get_expectations_str,
    _get_tui_choices_from_scan,
    _manual_review_no_tui,
    _read_manual_review_tmp_file,
)
from huggingface_hub.utils import SoftTemporaryDirectory, capture_output

from .testing_utils import handle_injection


class TestDeleteCacheHelpers(unittest.TestCase):
    def test_get_tui_choices_from_scan_empty(self) -> None:
        choices = _get_tui_choices_from_scan(repos={}, preselected=[], sort_by=None)
        assert len(choices) == 1
        assert isinstance(choices[0], Choice)
        assert choices[0].value == _CANCEL_DELETION_STR
        assert len(choices[0].name != 0)  # Something displayed to the user
        assert not (choices[0].enabled)

    def test_get_tui_choices_from_scan_with_preselection(self) -> None:
        choices = _get_tui_choices_from_scan(
            repos=_get_cache_mock().repos,
            preselected=[
                "dataset_revision_hash_id",  # dataset_1 is preselected
                "a_revision_id_that_does_not_exist",  # unknown but will not complain
                "older_hash_id",  # only the oldest revision from model_2
            ],
            sort_by=None,  # Don't sort to maintain original order
        )
        assert len(choices) == 8

        # Item to cancel everything
        assert isinstance(choices[0], Choice)
        assert choices[0].value == _CANCEL_DELETION_STR
        assert len(choices[0].name != 0)
        assert not (choices[0].enabled)

        # Dataset repo separator
        assert isinstance(choices[1], Separator)
        assert choices[1]._line == "\nDataset dummy_dataset (8M, used 2 weeks ago)"

        # Only revision of `dummy_dataset`
        assert isinstance(choices[2], Choice)
        assert choices[2].value == "dataset_revision_hash_id"
        self.assertEqual(
            choices[2].name,
            # truncated hash id + detached + last modified
            "dataset_: (detached) # modified 1 day ago",
        )
        assert choices[2].enabled  # preselected

        # Model `dummy_model` separator
        assert isinstance(choices[3], Separator)
        assert choices[3]._line == "\nModel dummy_model (1.4K, used 2 years ago)"

        # Recent revision of `dummy_model` (appears first due to sorting by last_modified)
        assert isinstance(choices[4], Choice)
        assert choices[4].value == "recent_hash_id"
        assert choices[4].name == "recent_h: main # modified 2 years ago"
        assert not (choices[4].enabled)

        # Oldest revision of `dummy_model`
        assert isinstance(choices[5], Choice)
        assert choices[5].value == "older_hash_id"
        assert choices[5].name == "older_ha: (detached) # modified 3 years ago"
        assert choices[5].enabled  # preselected

        # Model `gpt2` separator
        assert isinstance(choices[6], Separator)
        assert choices[6]._line == "\nModel gpt2 (3.6G, used 2 hours ago)"

        # Only revision of `gpt2`
        assert isinstance(choices[7], Choice)
        assert choices[7].value == "abcdef123456789"
        self.assertEqual(choices[7].name, "abcdef12: main, refs/pr/1 # modified 2 years ago")
        assert not (choices[7].enabled)

    def test_get_tui_choices_from_scan_with_sort_size(self) -> None:
        """Test sorting by size."""
        choices = _get_tui_choices_from_scan(repos=_get_cache_mock().repos, preselected=[], sort_by="size")

        # Verify repo order: gpt2 (3.6G) -> dummy_dataset (8M) -> dummy_model (1.4K)
        assert isinstance(choices[1], Separator)
        assert "gpt2" in choices[1]._line

        assert isinstance(choices[3], Separator)
        assert "dummy_dataset" in choices[3]._line

        assert isinstance(choices[5], Separator)
        assert "dummy_model" in choices[5]._line

    def test_get_expectations_str_on_no_deletion_item(self) -> None:
        """Test `_get_instructions` when `_CANCEL_DELETION_STR` is passed."""
        self.assertEqual(
            _get_expectations_str(
                hf_cache_info=Mock(),
                selected_hashes=["hash_1", _CANCEL_DELETION_STR, "hash_2"],
            ),
            "Nothing will be deleted.",
        )

    def test_get_expectations_str_with_selection(self) -> None:
        """Test `_get_instructions` with 2 revisions selected."""
        strategy_mock = Mock()
        strategy_mock.expected_freed_size_str = "5.1M"

        cache_mock = Mock()
        cache_mock.delete_revisions.return_value = strategy_mock

        self.assertEqual(
            _get_expectations_str(
                hf_cache_info=cache_mock,
                selected_hashes=["hash_1", "hash_2"],
            ),
            "2 revisions selected counting for 5.1M.",
        )
        cache_mock.delete_revisions.assert_called_once_with("hash_1", "hash_2")

    def test_read_manual_review_tmp_file(self) -> None:
        """Test `_read_manual_review_tmp_file`."""

        with SoftTemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "file.txt"

            with tmp_path.open("w") as f:
                f.writelines(
                    [
                        "# something commented out\n",
                        "###\n",
                        "\n\n\n\n",  # some empty lines
                        "    # Something commented out after spaces\n",
                        "a_revision_hash\n",
                        "a_revision_hash_with_a_comment # 2 years ago\n",
                        "    a_revision_hash_after_spaces\n",
                        "  a_revision_hash_with_a_comment_after_spaces # 2years ago\n",
                        "    # hash_commented_out #  2 years ago\n",
                        "a_revision_hash\n",  # Duplicate
                        "",  # empty line
                    ]
                )

            # Only non-commented lines are returned
            # Order is kept and lines are not de-duplicated
            self.assertListEqual(
                _read_manual_review_tmp_file(tmp_path),
                [
                    "a_revision_hash",
                    "a_revision_hash_with_a_comment",
                    "a_revision_hash_after_spaces",
                    "a_revision_hash_with_a_comment_after_spaces",
                    "a_revision_hash",
                ],
            )

    @patch("huggingface_hub.commands.delete_cache.input")
    @patch("huggingface_hub.commands.delete_cache.mkstemp")
    def test_manual_review_no_tui(self, mock_mkstemp: Mock, mock_input: Mock) -> None:
        # Mock file creation so that we know the file location in test
        fd, tmp_path = mkstemp()
        mock_mkstemp.return_value = fd, tmp_path

        # Mock cache
        cache_mock = _get_cache_mock()

        # Mock input from user
        def _input_answers():
            assert os.path.isfile(tmp_path)  # not deleted yet
            with open(tmp_path) as f:
                content = f.read()
            assert content.startswith("# INSTRUCTIONS")

            # older_hash_id is not commented
            assert "\n    older_hash_id # Refs: (detached)" in content
            # same for abcdef123456789
            assert "\n    abcdef123456789 # Refs: main in refs/pr/1", content
            # dataset revision is not preselected
            assert "#    dataset_revision_hash_id" in content
            # same for recent_hash_id
            assert "#    recent_hash_id" in content

            # Select dataset revision
            content = content.replace("#    dataset_revision_hash_id", "dataset_revision_hash_id")
            # Deselect abcdef123456789
            content = content.replace("abcdef123456789", "# abcdef123456789")
            with open(tmp_path, "w") as f:
                f.write(content)

            yield "no"  # User edited the file and want to see the strategy diff
            yield "y"  # User confirms

        mock_input.side_effect = _input_answers()

        # Run manual review
        with capture_output() as output:
            selected_hashes = _manual_review_no_tui(
                hf_cache_info=cache_mock, preselected=["abcdef123456789", "older_hash_id"], sort_by=None
            )

        # Tmp file has been created but is now deleted
        mock_mkstemp.assert_called_once_with(suffix=".txt")
        assert not (os.path.isfile(tmp_path))  # now deleted

        # User changed the selection
        self.assertListEqual(selected_hashes, ["dataset_revision_hash_id", "older_hash_id"])

        # Check printed instructions
        printed = output.getvalue()
        assert printed.startswith("TUI is disabled. In order to")  # ...
        assert tmp_path in printed

        # Check input called twice
        assert mock_input.call_count == 2

    @patch("huggingface_hub.commands.delete_cache.input")
    def test_ask_for_confirmation_no_tui(self, mock_input: Mock) -> None:
        """Test `_ask_for_confirmation_no_tui`."""
        # Answer yes
        mock_input.side_effect = ("y",)
        value = _ask_for_confirmation_no_tui("custom message 1", default=True)
        mock_input.assert_called_with("custom message 1 (Y/n) ")
        assert value

        # Answer no
        mock_input.side_effect = ("NO",)
        value = _ask_for_confirmation_no_tui("custom message 2", default=True)
        mock_input.assert_called_with("custom message 2 (Y/n) ")
        assert not (value)

        # Answer invalid, then default
        mock_input.side_effect = ("foo", "")
        with capture_output() as output:
            value = _ask_for_confirmation_no_tui("custom message 3", default=False)
        mock_input.assert_called_with("custom message 3 (y/N) ")
        assert not (value)
        self.assertEqual(
            output.getvalue(),
            "Invalid input. Must be one of ('y', 'yes', '1', 'n', 'no', '0', '')\n",
        )

    def test_get_tui_choices_from_scan_with_different_sorts(self) -> None:
        """Test different sorting modes."""
        cache_mock = _get_cache_mock()

        # Test size sorting (largest first) - order: gpt2 (3.6G) -> dummy_dataset (8M) -> dummy_model (1.4K)
        size_choices = _get_tui_choices_from_scan(cache_mock.repos, [], sort_by="size")
        # Separators at positions 1, 3, 5
        assert isinstance(size_choices[1], Separator)
        assert "gpt2" in size_choices[1]._line
        assert isinstance(size_choices[3], Separator)
        assert "dummy_dataset" in size_choices[3]._line
        assert isinstance(size_choices[5], Separator)
        assert "dummy_model" in size_choices[5]._line

        # Test alphabetical sorting - order: dummy_dataset -> dummy_model -> gpt2
        alpha_choices = _get_tui_choices_from_scan(cache_mock.repos, [], sort_by="alphabetical")
        # Separators at positions 1, 3, 6 (dummy_model has 2 revisions)
        assert isinstance(alpha_choices[1], Separator)
        assert "dummy_dataset" in alpha_choices[1]._line
        assert isinstance(alpha_choices[3], Separator)
        assert "dummy_model" in alpha_choices[3]._line
        assert isinstance(alpha_choices[6], Separator)
        assert "gpt2" in alpha_choices[6]._line

        # Test lastUpdated sorting - order: dummy_dataset (1 day) -> gpt2 (2 years) -> dummy_model (3 years)
        updated_choices = _get_tui_choices_from_scan(cache_mock.repos, [], sort_by="lastUpdated")
        # Separators at positions 1, 3, 5
        assert isinstance(updated_choices[1], Separator)
        assert "dummy_dataset" in updated_choices[1]._line
        assert isinstance(updated_choices[3], Separator)
        assert "gpt2" in updated_choices[3]._line
        assert isinstance(updated_choices[5], Separator)
        assert "dummy_model" in updated_choices[5]._line

        # Test lastUsed sorting - order: gpt2 (2h) -> dummy_dataset (2w) -> dummy_model (2y)
        used_choices = _get_tui_choices_from_scan(cache_mock.repos, [], sort_by="lastUsed")
        # Separators at positions 1, 3, 5
        assert isinstance(used_choices[1], Separator)
        assert "gpt2" in used_choices[1]._line
        assert isinstance(used_choices[3], Separator)
        assert "dummy_dataset" in used_choices[3]._line
        assert isinstance(used_choices[5], Separator)
        assert "dummy_model" in used_choices[5]._line


@patch("huggingface_hub.commands.delete_cache._ask_for_confirmation_no_tui")
@patch("huggingface_hub.commands.delete_cache._get_expectations_str")
@patch("huggingface_hub.commands.delete_cache.inquirer.confirm")
@patch("huggingface_hub.commands.delete_cache._manual_review_tui")
@patch("huggingface_hub.commands.delete_cache._manual_review_no_tui")
@patch("huggingface_hub.commands.delete_cache.scan_cache_dir")
@handle_injection
class TestMockedDeleteCacheCommand(unittest.TestCase):
    """Test case with a patched `DeleteCacheCommand` to test `.run()` without testing
    the manual review.
    """

    args: Mock
    command: DeleteCacheCommand

    def setUp(self) -> None:
        self.args = Mock()
        self.args.sort = None
        self.command = DeleteCacheCommand(self.args)

    def test_run_and_delete_with_tui(
        self,
        mock_scan_cache_dir: Mock,
        mock__manual_review_tui: Mock,
        mock__get_expectations_str: Mock,
        mock_confirm: Mock,
    ) -> None:
        """Test command run with a mocked manual review step."""
        # Mock return values
        mock__manual_review_tui.return_value = ["hash_1", "hash_2"]
        mock__get_expectations_str.return_value = "Will delete A and B."
        mock_confirm.return_value.execute.return_value = True
        mock_scan_cache_dir.return_value = _get_cache_mock()

        # Run
        self.command.disable_tui = False
        with capture_output() as output:
            self.command.run()

        # Step 1: scan
        mock_scan_cache_dir.assert_called_once_with(self.args.dir)
        cache_mock = mock_scan_cache_dir.return_value

        # Step 2: manual review
        mock__manual_review_tui.assert_called_once_with(cache_mock, preselected=[], sort_by=None)

        # Step 3: ask confirmation
        mock__get_expectations_str.assert_called_once_with(cache_mock, ["hash_1", "hash_2"])
        mock_confirm.assert_called_once_with("Will delete A and B. Confirm deletion ?", default=True)
        mock_confirm().execute.assert_called_once_with()

        # Step 4: delete
        cache_mock.delete_revisions.assert_called_once_with("hash_1", "hash_2")
        strategy_mock = cache_mock.delete_revisions.return_value
        strategy_mock.execute.assert_called_once_with()

        # Check output
        assert "Start deletion.\nDone. Deleted 0 repo(s) and 0 revision(s) for a total of 5.1M.\n" in output.getvalue()

    def test_run_nothing_selected_with_tui(self, mock__manual_review_tui: Mock) -> None:
        """Test command run but nothing is selected in manual review."""
        # Mock return value
        mock__manual_review_tui.return_value = []

        # Run
        self.command.disable_tui = False
        with capture_output() as output:
            self.command.run()

        # Check output
        assert "Deletion is cancelled. Do nothing.\n" in output.getvalue()

    def test_run_stuff_selected_but_cancel_item_as_well_with_tui(self, mock__manual_review_tui: Mock) -> None:
        """Test command run when some are selected but "cancel item" as well."""
        # Mock return value
        mock__manual_review_tui.return_value = [
            "hash_1",
            "hash_2",
            _CANCEL_DELETION_STR,
        ]

        # Run
        self.command.disable_tui = False
        with capture_output() as output:
            self.command.run()

        # Check output
        assert "Deletion is cancelled. Do nothing.\n" in output.getvalue()

    def test_run_and_delete_no_tui(
        self,
        mock_scan_cache_dir: Mock,
        mock__manual_review_no_tui: Mock,
        mock__get_expectations_str: Mock,
        mock__ask_for_confirmation_no_tui: Mock,
    ) -> None:
        """Test command run with a mocked manual review step."""
        # Mock return values
        mock__manual_review_no_tui.return_value = ["hash_1", "hash_2"]
        mock__get_expectations_str.return_value = "Will delete A and B."
        mock__ask_for_confirmation_no_tui.return_value.return_value = True
        mock_scan_cache_dir.return_value = _get_cache_mock()

        # Run
        self.command.disable_tui = True
        with capture_output() as output:
            self.command.run()

        # Step 1: scan
        mock_scan_cache_dir.assert_called_once_with(self.args.dir)
        cache_mock = mock_scan_cache_dir.return_value

        # Step 2: manual review
        mock__manual_review_no_tui.assert_called_once_with(cache_mock, preselected=[], sort_by=None)

        # Step 3: ask confirmation
        mock__get_expectations_str.assert_called_once_with(cache_mock, ["hash_1", "hash_2"])
        mock__ask_for_confirmation_no_tui.assert_called_once_with("Will delete A and B. Confirm deletion ?")

        # Step 4: delete
        cache_mock.delete_revisions.assert_called_once_with("hash_1", "hash_2")
        strategy_mock = cache_mock.delete_revisions.return_value
        strategy_mock.execute.assert_called_once_with()

        # Check output
        assert "Start deletion.\nDone. Deleted 0 repo(s) and 0 revision(s) for a total of 5.1M.\n" in output.getvalue()

    def test_run_with_sorting(self):
        """Test command run with sorting enabled."""
        self.args.sort = "size"
        self.command = DeleteCacheCommand(self.args)

        mock_scan_cache_dir = Mock()
        mock_scan_cache_dir.return_value = _get_cache_mock()

        with patch("huggingface_hub.commands.delete_cache.scan_cache_dir", mock_scan_cache_dir):
            with patch("huggingface_hub.commands.delete_cache._manual_review_tui") as mock_review:
                self.command.disable_tui = False
                self.command.run()

                mock_review.assert_called_once_with(mock_scan_cache_dir.return_value, preselected=[], sort_by="size")


def _get_cache_mock() -> Mock:
    # First model with 1 revision
    model_1 = Mock()
    model_1.repo_type = "model"
    model_1.repo_id = "gpt2"
    model_1.size_on_disk_str = "3.6G"
    model_1.last_accessed = 1660000000
    model_1.last_accessed_str = "2 hours ago"
    model_1.size_on_disk = 3.6 * 1024**3  # 3.6 GiB

    model_1_revision_1 = Mock()
    model_1_revision_1.commit_hash = "abcdef123456789"
    model_1_revision_1.refs = {"main", "refs/pr/1"}
    model_1_revision_1.last_modified = 123456789000  # 2 years ago
    model_1_revision_1.last_modified_str = "2 years ago"

    model_1.revisions = {model_1_revision_1}

    # Second model with 2 revisions
    model_2 = Mock()
    model_2.repo_type = "model"
    model_2.repo_id = "dummy_model"
    model_2.size_on_disk_str = "1.4K"
    model_2.last_accessed = 1550000000
    model_2.last_accessed_str = "2 years ago"
    model_2.size_on_disk = 1.4 * 1024  # 1.4K

    model_2_revision_1 = Mock()
    model_2_revision_1.commit_hash = "recent_hash_id"
    model_2_revision_1.refs = {"main"}
    model_2_revision_1.last_modified = 123456789  # 2 years ago
    model_2_revision_1.last_modified_str = "2 years ago"

    model_2_revision_2 = Mock()
    model_2_revision_2.commit_hash = "older_hash_id"
    model_2_revision_2.refs = {}
    model_2_revision_2.last_modified = 12345678000  # 3 years ago
    model_2_revision_2.last_modified_str = "3 years ago"

    model_2.revisions = {model_2_revision_1, model_2_revision_2}

    # And a dataset with 1 revision
    dataset_1 = Mock()
    dataset_1.repo_type = "dataset"
    dataset_1.repo_id = "dummy_dataset"
    dataset_1.size_on_disk_str = "8M"
    dataset_1.last_accessed = 1659000000
    dataset_1.last_accessed_str = "2 weeks ago"
    dataset_1.size_on_disk = 8 * 1024**2  # 8 MiB

    dataset_1_revision_1 = Mock()
    dataset_1_revision_1.commit_hash = "dataset_revision_hash_id"
    dataset_1_revision_1.refs = {}
    dataset_1_revision_1.last_modified = 1234567890000  # 1 day ago (newest)
    dataset_1_revision_1.last_modified_str = "1 day ago"

    dataset_1.revisions = {dataset_1_revision_1}

    # Fake cache
    strategy_mock = Mock()
    strategy_mock.repos = []
    strategy_mock.snapshots = []
    strategy_mock.expected_freed_size_str = "5.1M"

    cache_mock = Mock()
    cache_mock.repos = {model_1, model_2, dataset_1}
    cache_mock.delete_revisions.return_value = strategy_mock
    return cache_mock
