from abc import ABC, abstractmethod
from typing import Any

import fasttext
from huggingface_hub import hf_hub_download


class Pipeline(ABC):
    @abstractmethod
    def __init__(self, model_id: str):
        model_path = hf_hub_download(model_id, "model.bin", library_name="fasttext")
        self.model = fasttext.load_model(model_path)

    @abstractmethod
    def __call__(self, inputs: Any) -> Any:
        raise NotImplementedError("Pipelines should implement a __call__ method")


class PipelineException(Exception):
    pass
