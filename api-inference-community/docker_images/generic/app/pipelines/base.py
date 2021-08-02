import os
import sys
from abc import ABC, abstractmethod
from typing import Any

from huggingface_hub import snapshot_download


class Pipeline(ABC):
    @abstractmethod
    def __init__(self, model_id: str):
        filepath = snapshot_download(model_id)
        sys.path.append(filepath)
        if "requirements.txt" in os.listdir(filepath):
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    os.path.join(filepath, "requirements.txt"),
                ]
            )

        from pipeline import PreTrainedPipeline

        self.model = PreTrainedPipeline(filepath)

    @abstractmethod
    def __call__(self, inputs: Any) -> Any:
        return self.model(inputs)


class PipelineException(Exception):
    pass
