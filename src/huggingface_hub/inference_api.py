import json
import logging
from typing import Dict, List, Optional, Union

import requests

from .hf_api import HfApi, HfFolder


logger = logging.getLogger(__name__)


ENDPOINT = "https://api-inference.huggingface.co/pipeline"

ALL_TASKS = [
    # NLP
    "text-classification",
    "token-classification",
    "table-question-answering",
    "question-answering",
    "zero-shot-classification",
    "translation",
    "summarization",
    "conversational",
    "feature-extraction",
    "text-generation",
    "text2text-generation",
    "fill-mask",
    "sentence-similarity",
    # Audio
    "text-to-speech",
    "automatic-speech-recognition",
    "audio-source-separation",
    "voice-activity-detection",
    # Computer vision
    "image-classification",
    "object-detection",
    "image-segmentation",
]


class InferenceApi:
    """Client to configure requests and make calls to the HuggingFace Inference API.

    Example:

            >>> from huggingface_hub.inference_api import InferenceApi

            >>> # Mask-fill example
            >>> api = InferenceApi("bert-base-uncased")
            >>> api.set_inputs(inputs="The goal of life is [MASK].")
            >>> api.call()

            >>> # Question Answering example
            >>> api = InferenceApi("deepset/roberta-base-squad2")
            >>> api.set_inputs(question="What's my name?", context="My name is Clara and I live in Berkeley.")
            >>> api.call()

            >>> # Zero-shot example
            >>> api = InferenceApi("typeform/distilbert-base-uncased-mnli")
            >>> api.set_inputs(inputs="Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!")
            >>> api.set_params(candidate_labels=["refund", "legal", "faq"])
            >>> api.call()
    """

    def __init__(
        self, repoId: str, task: Optional[str] = None, gpu: Optional[bool] = False
    ):
        """Inits InferenceApi headers and API call information.

        Args:
            repoId (``str``): Id of model (e.g. `bert-base-uncased`).
            task (``str``, `optional`, defaults ``None``): Whether to force a task instead of using task specified in repository.
            gpu (``bool``, `optional`, defaults ``None``): Whether to use GPU instead of CPU for inference(requires Startup plan at least).
        """
        token = HfFolder.get_token()
        if token is None:
            raise EnvironmentError("A Hugging Face token was not found.")

        # Configure task
        modelInfo = HfApi().model_info(repo_id=repoId, token=token)
        if not modelInfo.pipeline_tag and not task:
            raise ValueError(
                "Task not specified in the repository. Please add it to the model card using pipeline_tag (https://huggingface.co/docs#how-is-a-models-type-of-inference-api-and-widget-determined)"
            )

        if task and task != modelInfo.pipeline_tag:
            if task not in ALL_TASKS:
                raise ValueError(f"Invalid task {task}. Make sure it's valid.")

            logger.warning(
                "You're using a different task than the one specified in the repository. Be sure to know what you're doing :)"
            )
            self.task = task
        else:
            self.task = modelInfo.pipeline_tag

        # Configure url, headers and options
        self.api_url = f"{ENDPOINT}/{self.task}/{repoId}"
        self.headers = {"authorization": "Bearer {}".format(token)}
        self.options = {"wait_for_model": True, "use_gpu": gpu}

        print(f"Initialized Inference API for {repoId} with task {self.task}")

    def __call__(
        self, inputs: Union[str, Dict, List[str], List[List[str]]], params: Optional[Dict] = None
    ):
        payload = {
            "inputs": inputs,
            "params": params,
            "options": self.options,
        }
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        return response.json()
