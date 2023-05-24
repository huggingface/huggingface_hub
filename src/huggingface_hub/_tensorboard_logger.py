from .utils import is_tensorboard_available


if not is_tensorboard_available():
    raise ImportError(
        "Tensorboard not available. To use `HFTensorBoardLogger` please install it by running `pip install"
        " tensorboardX`."
    )
from tensorboardX import SummaryWriter

from ._training_logger import HFLoggerMixin


class HFTensorBoardLogger(HFLoggerMixin, SummaryWriter):
    @property
    def log_dir(self) -> str:
        return self.logdir

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Push to hub in a non-blocking way when exiting the logger's context manager."""
        super().__exit__(exc_type, exc_val, exc_tb)
        future = self.push_to_hub(commit_message="Closing Tensorboard logger.")
        future.result()
