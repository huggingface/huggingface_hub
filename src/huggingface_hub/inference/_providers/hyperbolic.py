import base64
from abc import ABC
from typing import Any, Dict, Optional, Union

from huggingface_hub.inference._common import _as_dict
from huggingface_hub.inference._providers._common import TaskProviderHelper, filter_none


class HyperbolicTask(TaskProviderHelper, ABC):
    """Base class for Hyperbolic API tasks."""

    def __init__(self, task: str):
        super().__init__(provider="hyperbolic", base_url="https://api.hyperbolic.xyz", task=task)

    def _prepare_route(self, mapped_model: str) -> str:
        if self.task == "text-to-image":
            return "/v1/images/generations"
        elif self.task == "conversational" or self.task == "text-generation":
            return "/v1/chat/completions"
        raise ValueError(f"Unsupported task '{self.task}' for Hyperbolic API.")


class HyperbolicTextGenerationTask(HyperbolicTask):
    # Handle both "text-generation" and "conversational"
    def _prepare_payload_as_dict(self, inputs: Any, parameters: Dict, mapped_model: str) -> Optional[Dict]:
        return {"messages": inputs, **filter_none(parameters), "model": mapped_model}


class HyperbolicTextToImageTask(HyperbolicTask):
    def __init__(self):
        super().__init__("text-to-image")

    def _prepare_payload_as_dict(self, inputs: Any, parameters: Dict, mapped_model: str) -> Optional[Dict]:
        parameters = filter_none(parameters)
        if "num_inference_steps" in parameters:
            parameters["steps"] = parameters.pop("num_inference_steps")
        if "guidance_scale" in parameters:
            parameters["cfg_scale"] = parameters.pop("guidance_scale")
        return {"prompt": inputs, "model_name": mapped_model, **parameters}

    def get_response(self, response: Union[bytes, Dict]) -> Any:
        response_dict = _as_dict(response)
        return base64.b64decode(response_dict["images"][0]["image"])
