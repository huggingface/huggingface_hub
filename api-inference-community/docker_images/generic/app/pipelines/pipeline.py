import os
import subprocess
import sys
from typing import Any

from huggingface_hub import snapshot_download


class Pipeline:
    def __init__(self, model_id: str):
        filepath = snapshot_download(model_id)
        sys.path.append(filepath)
        if "requirements.txt" in os.listdir(filepath):
            cache_dir = os.environ["PIP_CACHE"]
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--cache-dir",
                    cache_dir,
                    "-r",
                    os.path.join(filepath, "requirements.txt"),
                ]
            )

        from pipeline import PreTrainedPipeline

        self.model = PreTrainedPipeline(filepath)
        if hasattr(self.model, "sampling_rate"):
            self.sampling_rate = self.model.sampling_rate
        else:
            # 16000 by default if not specified
            self.sampling_rate = 16000

    def __call__(self, inputs: Any) -> Any:
        return self.model(inputs)


class PipelineException(Exception):
    pass
