import unittest
from unittest.mock import Mock, patch

from huggingface_hub.commands.delete_cache import (
    _CANCEL_DELETION_STR,
    DeleteCacheCommand,
    _get_instructions_str,
    _get_tui_choices_from_scan,
)
from InquirerPy.base.control import Choice
from InquirerPy.separator import Separator

from .testing_utils import capture_output


class TestDeleteCacheHelpers(unittest.TestCase):
    def test_get_tui_choices_from_scan_empty(self) -> None:
        choices = _get_tui_choices_from_scan(repos={}, preselected=[])
        self.assertEqual(len(choices), 1)
        self.assertIsInstance(choices[0], Choice)
        self.assertEqual(choices[0].value, _CANCEL_DELETION_STR)
        self.assertTrue(len(choices[0].name) != 0)  # Something displayed to the user
        self.assertFalse(choices[0].enabled)

    def test_get_tui_choices_from_scan_with_preselection(self) -> None:
        # First model with 1 revision
        model_1 = Mock()
        model_1.repo_type = "model"
        model_1.repo_id = "gpt2"
        model_1.size_on_disk_str = "3.6G"
        model_1.last_accessed_str = "2 hours ago"

        model_1_revision_1 = Mock()
        model_1_revision_1.commit_hash = "abcdef123456789"
        model_1_revision_1.refs = {"main", "refs/pr/1"}
        # model_1_revision_1.last_modified = 123456789  # timestamp
        model_1_revision_1.last_modified_str = "2 years ago"

        model_1.revisions = {model_1_revision_1}

        # Second model with 2 revisions
        model_2 = Mock()
        model_2.repo_type = "model"
        model_2.repo_id = "dummy_model"
        model_2.size_on_disk_str = "1.4K"
        model_2.last_accessed_str = "2 years ago"

        model_2_revision_1 = Mock()
        model_2_revision_1.commit_hash = "recent_hash_id"
        model_2_revision_1.refs = {"main"}
        model_2_revision_1.last_modified = 123456789  # newer timestamp
        model_2_revision_1.last_modified_str = "2 years ago"

        model_2_revision_2 = Mock()
        model_2_revision_2.commit_hash = "older_hash_id"
        model_2_revision_2.refs = {}
        model_2_revision_2.last_modified = 12345678  # older timestamp
        model_2_revision_2.last_modified_str = "3 years ago"

        model_2.revisions = {model_2_revision_1, model_2_revision_2}

        # And a dataset with 1 revision
        dataset_1 = Mock()
        dataset_1.repo_type = "dataset"
        dataset_1.repo_id = "dummy_dataset"
        dataset_1.size_on_disk_str = "8M"
        dataset_1.last_accessed_str = "2 weeks ago"

        dataset_1_revision_1 = Mock()
        dataset_1_revision_1.commit_hash = "dataset_revision_hash_id"
        dataset_1_revision_1.refs = {}
        dataset_1_revision_1.last_modified_str = "1 day ago"

        dataset_1.revisions = {dataset_1_revision_1}

        choices = _get_tui_choices_from_scan(
            repos={model_1, model_2, dataset_1},
            preselected=[
                "dataset_revision_hash_id",  # dataset_1 is preselected
                "a_revision_id_that_does_not_exist",  # unknown but will not complain
                "older_hash_id",  # only the oldest revision from model_2
            ],
        )

        self.assertEqual(len(choices), 8)

        # Item to cancel everything
        self.assertIsInstance(choices[0], Choice)
        self.assertEqual(choices[0].value, _CANCEL_DELETION_STR)
        self.assertTrue(len(choices[0].name) != 0)
        self.assertFalse(choices[0].enabled)

        # Dataset repo separator
        self.assertIsInstance(choices[1], Separator)
        self.assertEqual(
            choices[1]._line, "\nDataset dummy_dataset (8M, used 2 weeks ago)"
        )

        # Only revision of `dummy_dataset`
        self.assertIsInstance(choices[2], Choice)
        self.assertEqual(choices[2].value, "dataset_revision_hash_id")
        self.assertEqual(
            choices[2].name,
            # truncated hash id + detached + last modified
            "dataset_: (detached) # modified 1 day ago",
        )
        self.assertTrue(choices[2].enabled)  # preselected

        # Model `dummy_model` separator
        self.assertIsInstance(choices[3], Separator)
        self.assertEqual(
            choices[3]._line, "\nModel dummy_model (1.4K, used 2 years ago)"
        )

        # Newest revision of `dummy_model`
        self.assertIsInstance(choices[4], Choice)
        self.assertEqual(choices[4].value, "recent_hash_id")
        self.assertEqual(choices[4].name, "recent_h: main # modified 2 years ago")
        self.assertFalse(choices[4].enabled)

        # Oldest revision of `dummy_model`
        self.assertIsInstance(choices[5], Choice)
        self.assertEqual(choices[5].value, "older_hash_id")
        self.assertEqual(choices[5].name, "older_ha: (detached) # modified 3 years ago")
        self.assertTrue(choices[5].enabled)  # preselected

        # Model `gpt2` separator
        self.assertIsInstance(choices[6], Separator)
        self.assertEqual(choices[6]._line, "\nModel gpt2 (3.6G, used 2 hours ago)")

        # Only revision of `gpt2`
        self.assertIsInstance(choices[7], Choice)
        self.assertEqual(choices[7].value, "abcdef123456789")
        self.assertEqual(
            choices[7].name, "abcdef12: main, refs/pr/1 # modified 2 years ago"
        )
        self.assertFalse(choices[7].enabled)

    def test_get_instructions_str_on_no_deletion_item(self) -> None:
        """Test `_get_instructions` when `_CANCEL_DELETION_STR` is passed."""
        self.assertEqual(
            _get_instructions_str(
                hf_cache_info=Mock(),
                selected_hashes=["hash_1", _CANCEL_DELETION_STR, "hash_2"],
            ),
            "Nothing will be deleted.",
        )

    def test_get_instructions_str_with_selection(self) -> None:
        """Test `_get_instructions` with 2 revisions selected."""
        strategy_mock = Mock()
        strategy_mock.expected_freed_size_str = "5.1M"

        cache_mock = Mock()
        cache_mock.delete_revisions.return_value = strategy_mock

        self.assertEqual(
            _get_instructions_str(
                hf_cache_info=cache_mock,
                selected_hashes=["hash_1", "hash_2"],
            ),
            "2 revisions selected counting for 5.1M.",
        )
        cache_mock.delete_revisions.assert_called_once_with("hash_1", "hash_2")


@patch("huggingface_hub.commands.delete_cache.scan_cache_dir")
@patch("huggingface_hub.commands.delete_cache.inquirer.confirm")
@patch("huggingface_hub.commands.delete_cache._manual_review_tui")
@patch("huggingface_hub.commands.delete_cache._get_instructions_str")
class TestMockedDeleteCacheCommand(unittest.TestCase):
    """Test case with a patched `DeleteCacheCommand` to test `.run()` without testing
    the manual review.
    """

    args: Mock
    command: DeleteCacheCommand

    def setUp(self) -> None:
        self.args = Mock()
        self.command = DeleteCacheCommand(self.args)

    def _check_mock_injections(
        self,
        confirm_mock: Mock,
        scan_cache_dir_mock: Mock,
        manual_review_tui_mock: Mock,
        get_instructions_str_mock: Mock,
    ) -> None:
        """Hacky way to check that mocks injection is done correctly.

        When adding/removing a patch, the order might change (and is not alphabetical)
        which mess up the tests. This helper checks that the order is as expected.
        """
        self.assertEqual(confirm_mock._extract_mock_name(), "confirm")
        self.assertEqual(scan_cache_dir_mock._extract_mock_name(), "scan_cache_dir")
        self.assertEqual(
            manual_review_tui_mock._extract_mock_name(), "_manual_review_tui"
        )
        self.assertEqual(
            get_instructions_str_mock._extract_mock_name(), "_get_instructions_str"
        )

    def test_run_and_delete_with_tui(
        self,
        get_instructions_str_mock: Mock,
        manual_review_tui_mock: Mock,
        confirm_mock: Mock,
        scan_cache_dir_mock: Mock,
    ) -> None:
        """Test command run with a mocked manual review step."""
        self._check_mock_injections(
            confirm_mock,
            scan_cache_dir_mock,
            manual_review_tui_mock,
            get_instructions_str_mock,
        )

        # Mock return values
        manual_review_tui_mock.return_value = ["hash_1", "hash_2"]
        get_instructions_str_mock.return_value = "Will delete A and B."
        confirm_mock.return_value.execute.return_value = True
        strategy_mock = scan_cache_dir_mock.return_value.delete_revisions.return_value
        strategy_mock.expected_freed_size_str = "8M"

        # Run
        self.command.disable_tui = False
        with capture_output() as output:
            self.command.run()

        # Step 1: scan
        scan_cache_dir_mock.assert_called_once_with(self.args.dir)
        cache_mock = scan_cache_dir_mock.return_value

        # Step 2: manual review
        manual_review_tui_mock.assert_called_once_with(cache_mock, preselected=[])

        # Step 3: ask confirmation
        get_instructions_str_mock.assert_called_once_with(
            cache_mock, ["hash_1", "hash_2"]
        )
        confirm_mock.assert_called_once_with(
            "Will delete A and B. Confirm deletion ?", default=True
        )
        confirm_mock().execute.assert_called_once_with()

        # Step 4: delete
        cache_mock.delete_revisions.assert_called_once_with("hash_1", "hash_2")
        strategy_mock = cache_mock.delete_revisions.return_value
        strategy_mock.execute.assert_called_once_with()

        # Check output
        self.assertEqual(
            output.getvalue(),
            "Start deletion.\n"
            "Done. Deleted 0 repo(s) and 0 revision(s) for a total of 8M.\n",
        )

    def test_run_nothing_selected_with_tui(
        self,
        get_instructions_str_mock: Mock,
        manual_review_tui_mock: Mock,
        confirm_mock: Mock,
        scan_cache_dir_mock: Mock,
    ) -> None:
        """Test command run but nothing is selected in manual review."""
        self._check_mock_injections(
            confirm_mock,
            scan_cache_dir_mock,
            manual_review_tui_mock,
            get_instructions_str_mock,
        )

        # Mock return value
        manual_review_tui_mock.return_value = []

        # Run
        self.command.disable_tui = False
        with capture_output() as output:
            self.command.run()

        # Check output
        self.assertEqual(output.getvalue(), "Deletion is cancelled. Do nothing.\n")

    def test_run_stuff_selected_but_cancel_item_as_well_with_tui(
        self,
        get_instructions_str_mock: Mock,
        manual_review_tui_mock: Mock,
        confirm_mock: Mock,
        scan_cache_dir_mock: Mock,
    ) -> None:
        """Test command run when some are selected but "cancel item" as well."""
        self._check_mock_injections(
            confirm_mock,
            scan_cache_dir_mock,
            manual_review_tui_mock,
            get_instructions_str_mock,
        )

        # Mock return value
        manual_review_tui_mock.return_value = ["hash_1", "hash_2", _CANCEL_DELETION_STR]

        # Run
        self.command.disable_tui = False
        with capture_output() as output:
            self.command.run()

        # Check output
        self.assertEqual(output.getvalue(), "Deletion is cancelled. Do nothing.\n")
