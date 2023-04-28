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
