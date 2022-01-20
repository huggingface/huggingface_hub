from typing import Dict, List, Optional, Union

import requests

from .hf_api import HfApi
from .utils import logging


logger = logging.get_logger(__name__)


ENDPOINT = "https://api-inference.huggingface.co"

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
    "audio-to-audio",
    "audio-classification",
    "voice-activity-detection",
    # Computer vision
    "image-classification",
    "object-detection",
    "image-segmentation",
    "text-to-image",
    # Others
    "structured-data-classification",
]


class InferenceApi:
    """Client to configure requests and make calls to the HuggingFace Inference API.

    Example:

            >>> from huggingface_hub.inference_api import InferenceApi

            >>> # Mask-fill example
            >>> inference = InferenceApi("bert-base-uncased")
            >>> inference(inputs="The goal of life is [MASK].")
            >>> >> [{'sequence': 'the goal of life is life.', 'score': 0.10933292657136917, 'token': 2166, 'token_str': 'life'}]

            >>> # Question Answering example
            >>> inference = InferenceApi("deepset/roberta-base-squad2")
            >>> inputs = {"question":"What's my name?", "context":"My name is Clara and I live in Berkeley."}
            >>> inference(inputs)
            >>> >> {'score': 0.9326569437980652, 'start': 11, 'end': 16, 'answer': 'Clara'}

            >>> # Zero-shot example
            >>> inference = InferenceApi("typeform/distilbert-base-uncased-mnli")
            >>> inputs = "Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!"
            >>> params = {"candidate_labels":["refund", "legal", "faq"]}
            >>> inference(inputs, params)
            >>> >> {'sequence': 'Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!', 'labels': ['refund', 'faq', 'legal'], 'scores': [0.9378499388694763, 0.04914155602455139, 0.013008488342165947]}

            >>> # Overriding configured task
            >>> inference = InferenceApi("bert-base-uncased", task="feature-extraction")
    """

    def __init__(
        self,
        repo_id: str,
        task: Optional[str] = None,
        token: Optional[str] = None,
        gpu: Optional[bool] = False,
    ):
        """Inits headers and API call information.

        Args:
            repo_id (``str``): Id of repository (e.g. `user/bert-base-uncased`).
            task (``str``, `optional`, defaults ``None``): Whether to force a task instead of using task specified in the repository.
            token (:obj:`str`, `optional`):
                The API token to use as HTTP bearer authorization. This is not the authentication token.
                You can find the token in https://huggingface.co/settings/token. Alternatively, you can
                find both your organizations and personal API tokens using `HfApi().whoami(token)`.
            gpu (``bool``, `optional`, defaults ``False``): Whether to use GPU instead of CPU for inference(requires Startup plan at least).
        .. note::
            Setting :obj:`token` is required when you want to use a private model.
        """
        self.options = {"wait_for_model": True, "use_gpu": gpu}

        self.headers = {}
        if isinstance(token, str):
            self.headers["Authorization"] = f"Bearer {token}"

        # Configure task
        model_info = HfApi().model_info(repo_id=repo_id, token=token)
        if not model_info.pipeline_tag and not task:
            raise ValueError(
                "Task not specified in the repository. Please add it to the model card using pipeline_tag (https://huggingface.co/docs#how-is-a-models-type-of-inference-api-and-widget-determined)"
            )

        if task and task != model_info.pipeline_tag:
            if task not in ALL_TASKS:
                raise ValueError(f"Invalid task {task}. Make sure it's valid.")

            logger.warning(
                "You're using a different task than the one specified in the repository. Be sure to know what you're doing :)"
            )
            self.task = task
        else:
            self.task = model_info.pipeline_tag

        self.api_url = f"{ENDPOINT}/pipeline/{self.task}/{repo_id}"

    def __repr__(self):
        items = (f"{k}='{v}'" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({', '.join(items)})"

    def __call__(
        self,
        inputs: Optional[Union[str, Dict, List[str], List[List[str]]]] = None,
        params: Optional[Dict] = None,
        data: Optional[bytes] = None,
    ):
        payload = {
            "options": self.options,
        }

        if inputs:
            payload["inputs"] = inputs

        if params:
            payload["parameters"] = params

        # TODO: Decide if we should raise an error instead of
        # returning the json.
        response = requests.post(
            self.api_url, headers=self.headers, json=payload, data=data
        ).json()
        return response
