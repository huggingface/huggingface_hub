from abc import ABC, abstractmethod
from typing import Any


class Pipeline(ABC):
    @abstractmethod
    def __init__(self, model_id: str):
        raise NotImplementedError("Pipelines should implement an __init__ method")

    @abstractmethod
    def __call__(self, inputs: Any) -> Any:
        raise NotImplementedError("Pipelines should implement a __call__ method")


class PipelineException(Exception):
    pass
