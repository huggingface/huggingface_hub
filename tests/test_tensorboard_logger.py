from unittest.mock import MagicMock, patch

from huggingface_hub._tensorboard_logger import HFSummaryWriter


class FakeSummaryWriter:
    def __init__(self, logdir=None, **kwargs):
        self.logdir = logdir or "/tmp/runs"
        self.scalars = []
        self.exited = False

    def add_scalar(self, *args):
        self.scalars.append(args)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exited = True


def test_summary_writer_is_loaded_on_instantiation_and_forwards_its_api() -> None:
    scheduler = MagicMock(repo_id="user/repo", repo_type="model", revision="main")
    card = MagicMock(data={"tags": ["hf-summary-writer"]})

    with (
        patch("huggingface_hub._tensorboard_logger._load_summary_writer", return_value=FakeSummaryWriter) as load,
        patch("huggingface_hub._tensorboard_logger.CommitScheduler", return_value=scheduler),
        patch("huggingface_hub._tensorboard_logger.ModelCard.load", return_value=card),
    ):
        writer = HFSummaryWriter(repo_id="user/repo", logdir="/tmp/logs")

    load.assert_called_once_with()
    writer.add_scalar("loss", 0.5, 1)
    assert writer._summary_writer.scalars == [("loss", 0.5, 1)]

    writer.__enter__()
    writer.__exit__(None, None, None)
    assert writer._summary_writer.exited
    scheduler.trigger.assert_called_once_with()
    scheduler.trigger.return_value.result.assert_called_once_with()
