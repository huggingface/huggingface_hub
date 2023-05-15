import warnings
from typing import Any, Dict, List, Optional, Union

from requests import Response

from .constants import INFERENCE_ENDPOINT
from .utils import build_hf_headers, get_session, hf_raise_for_status


# Related resources:
#    https://huggingface.co/tasks
#    https://huggingface.co/docs/huggingface.js/inference/README
#    https://github.com/huggingface/text-generation-inference/tree/main/clients/python
#    https://github.com/huggingface/text-generation-inference/blob/main/clients/python/text_generation/client.py
#    https://huggingface.slack.com/archives/C03E4DQ9LAJ/p1680169099087869

# TODO:
# - handle options? wait_for_model, use_gpu,... See list: https://github.com/huggingface/huggingface.js/blob/main/packages/inference/src/types.ts#L1
# - handle parameters? we can based implementation on inference.js
# - validate inputs/options/parameters? with Pydantic for instance? or only optionally?
# - add all tasks
# - handle async requests


class InferenceClient:
    def __init__(
        self, model: Optional[str] = None, token: Optional[str] = None, timeout: Optional[int] = None
    ) -> None:
        # If set, `model` can be either a repo_id on the Hub or an endpoint URL.
        self.model: Optional[str] = model
        self.headers = build_hf_headers(token=token)
        self.timeout = timeout

    def __repr__(self):
        return f"<InferenceClient(model='{self.model if self.model else ''}', timeout={self.timeout})>"

    def post(
        self,
        json: Optional[Union[str, Dict, List]] = None,
        data: Optional[bytes] = None,
        model: Optional[str] = None,
        task: Optional[str] = None,
    ) -> Response:
        url = self._resolve_url(model, task)

        if data is not None and json is not None:
            warnings.warn("Ignoring `json` as `data` is passed as binary.")

        response = get_session().post(url, json=json, data=data, headers=self.headers, timeout=self.timeout)
        hf_raise_for_status(response)
        return response

    def summarization(
        self,
        text: str,
        parameters: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ):
        payload: Dict[str, Any] = {"inputs": text}
        if parameters is not None:
            payload["parameters"] = parameters
        response = self.post(json=payload, model=model, task="summarization")
        return response.json()[0]["summary_text"]

    def _resolve_url(self, model: Optional[str], task: Optional[str]) -> str:
        model = model or self.model

        # If model is already a URL, ignore `task` and return directly
        if model is not None and (model.startswith("http://") or model.startswith("https://")):
            return model

        # # If no model but task is set => fetch the recommended one for this task
        if model is None:
            if task is None:
                raise ValueError(
                    "You must specify at least a model (repo_id or URL) or a task, either when instantiating"
                    " `InferenceClient` or when making a request."
                )
            model = get_model_id_by_task(task)

        # If no task but model is set => fetch the default task for this pipeline
        if task is None:
            task = get_task_by_model_id(model)

        # Compute InferenceAPI url
        return self.get_inference_api_url(model, task)

    @staticmethod
    def get_inference_api_url(model_id: str, task: str) -> str:
        return f"{INFERENCE_ENDPOINT}/pipeline/{task}/{model_id}"


def get_model_id_by_task(task: str) -> str:
    if task == "summarization":
        return "facebook/bart-large-cnn"
    raise NotImplementedError()


def get_task_by_model_id(model_id: str) -> str:
    raise NotImplementedError()
