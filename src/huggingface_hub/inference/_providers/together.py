import base64
from abc import ABC
from typing import Any, Dict, Optional, Union

from huggingface_hub.inference._common import _as_dict
from huggingface_hub.inference._providers._common import TaskProviderHelper, filter_none


class TogetherTask(TaskProviderHelper, ABC):
    """Base class for Together API tasks."""

    def __init__(self, task: str):
        super().__init__(provider="together", base_url="https://api.together.xyz", task=task)

    def _prepare_route(self, mapped_model: str) -> str:
        if self.task == "text-to-image":
            return "/v1/images/generations"
        elif self.task == "conversational":
            return "/v1/chat/completions"
        elif self.task == "text-generation":
            return "/v1/completions"
        raise ValueError(f"Unsupported task '{self.task}' for Together API.")


class TogetherTextGenerationTask(TogetherTask):
    # Handle both "text-generation" and "conversational"
    def _prepare_payload(
        self, inputs: Any, parameters: Dict, mapped_model: str, extra_payload: Optional[Dict] = None
    ) -> Optional[Dict]:
        return {"messages": inputs, **filter_none(parameters), "model": mapped_model, **(extra_payload or {})}


class TogetherTextToImageTask(TogetherTask):
    def __init__(self):
        super().__init__("text-to-image")

    def _prepare_payload(
        self, inputs: Any, parameters: Dict, mapped_model: str, extra_payload: Optional[Dict] = None
    ) -> Optional[Dict]:
        parameters = filter_none(parameters)
        if "num_inference_steps" in parameters:
            parameters["steps"] = parameters.pop("num_inference_steps")
        if "guidance_scale" in parameters:
            parameters["guidance"] = parameters.pop("guidance_scale")

        return {"prompt": inputs, "response_format": "base64", **parameters, "model": mapped_model}

    def get_response(self, response: Union[bytes, Dict]) -> Any:
        response_dict = _as_dict(response)
        return base64.b64decode(response_dict["data"][0]["b64_json"])
