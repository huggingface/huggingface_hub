import warnings
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Union

from requests import Response
import io
import base64
from ._inference_types import ClassificationOutput, ConversationalOutput, ImageSegmentationOutput
from .constants import INFERENCE_ENDPOINT
from .utils import build_hf_headers, get_session, hf_raise_for_status, is_pillow_available


# Related resources:
#    https://huggingface.co/tasks
#    https://huggingface.co/docs/huggingface.js/inference/README
#    https://github.com/huggingface/huggingface.js/tree/main/packages/inference/src
#    https://github.com/huggingface/text-generation-inference/tree/main/clients/python
#    https://github.com/huggingface/text-generation-inference/blob/main/clients/python/text_generation/client.py
#    https://huggingface.slack.com/archives/C03E4DQ9LAJ/p1680169099087869

# TODO:
# - handle options? wait_for_model, use_gpu,... See list: https://github.com/huggingface/huggingface.js/blob/main/packages/inference/src/types.ts#L1
# - handle parameters? we can based implementation on inference.js
# - validate inputs/options/parameters? with Pydantic for instance? or only optionally?
# - add all tasks
# - handle async requests
# - if a user tries to call a task on a model that doesn't support it, I'll gracefully handle the error to print to the user the available task(s) for their model.
#       invalid task: client.summarization(EXAMPLE, model="codenamewei/speech-to-text")
# Make BinaryT work with URLs as well

RECOMMENDED_MODELS = {
    "audio-classification": "superb/hubert-large-superb-er",
    "automatic-speech-recognition": "facebook/wav2vec2-large-960h-lv60-self",
    "conversational": "microsoft/DialoGPT-large",
    "image-classification": "google/vit-base-patch16-224",
    "image-segmentation": "facebook/detr-resnet-50-panoptic",
    "summarization": "facebook/bart-large-cnn",
    "text-to-speech": "espnet/kan-bayashi_ljspeech_vits",
}

PathT = Union[str, Path]
BinaryT = Union[bytes, BinaryIO, PathT]


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

    def audio_classification(
        self,
        audio: BinaryT,
        model: Optional[str] = None,
    ) -> ClassificationOutput:
        # Recommended: superb/hubert-large-superb-er
        if isinstance(audio, (str, Path)):
            audio = Path(audio).read_bytes()
        response = self.post(data=audio, model=model, task="audio-classification")
        return response.json()

    def automatic_speech_recognition(
        self,
        audio: BinaryT,
        model: Optional[str] = None,
    ) -> str:
        # Recommended: facebook/wav2vec2-large-960h-lv60-self
        if isinstance(audio, (str, Path)):
            audio = Path(audio).read_bytes()
        response = self.post(data=audio, model=model, task="automatic-speech-recognition")
        return response.json()["text"]

    def conversational(
        self,
        text: str,
        generated_responses: Optional[List[str]] = None,
        past_user_inputs: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> ConversationalOutput:
        # Recommended: microsoft/DialoGPT-large
        payload: Dict[str, Any] = {"inputs": {"text": text}}
        if generated_responses is not None:
            payload["inputs"]["generated_responses"] = generated_responses
        if past_user_inputs is not None:
            payload["inputs"]["past_user_inputs"] = past_user_inputs
        if parameters is not None:
            payload["parameters"] = parameters
        response = self.post(json=payload, model=model, task="conversational")
        return response.json()

    def image_classification(
        self,
        image: BinaryT,
        model: Optional[str] = None,
    ) -> ClassificationOutput:
        # Recommended: google/vit-base-patch16-224
        if isinstance(image, (str, Path)):
            image = Path(image).read_bytes()
        response = self.post(data=image, model=model, task="image-classification")
        return response.json()

    def image_segmentation(
        self,
        image: BinaryT,
        model: Optional[str] = None,
    ) -> List[ImageSegmentationOutput]:
        # Recommended: facebook/detr-resnet-50-panoptic
        Image = _import_image("image-segmentation")

        # Segment
        if isinstance(image, (str, Path)):
            image = Path(image).read_bytes()
        response = self.post(data=image, model=model, task="image-segmentation")
        output = response.json()

        # Parse masks as PIL Image
        if not isinstance(output, list):
            raise ValueError(f"Server output must be a list. Got {type(output)}: {str(output)[:200]}...")
        for item in output:
            item["mask"] = Image.open(io.BytesIO(base64.b64decode(item["mask"])))
        return output

    def summarization(
        self,
        text: str,
        parameters: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> str:
        payload: Dict[str, Any] = {"inputs": text}
        if parameters is not None:
            payload["parameters"] = parameters
        response = self.post(json=payload, model=model, task="summarization")
        return response.json()[0]["summary_text"]

    def text_to_speech(self, text: str, model: Optional[str] = None) -> bytes:
        response = self.post(json={"inputs": text}, model=model, task="text-to-speech")
        return response.content

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
    if task in RECOMMENDED_MODELS:
        return RECOMMENDED_MODELS[task]
    raise NotImplementedError()


def get_task_by_model_id(model_id: str) -> str:
    raise NotImplementedError()


def _import_image(task: str):
    if not is_pillow_available():
        raise ImportError(
            f"Please install Pillow to use task '{task}' (`pip install Pillow`). If you don't want the image to be"
            f" post-processed, use `client.post(..., model=model, task='{task}')` to get the raw response from the"
            " server."
        )
    from PIL import Image

    return Image


if __name__ == "__main__":
    client = InferenceClient()

    # Text to speech to text
    audio = client.text_to_speech("Hello world")
    client.audio_classification(audio)
    client.automatic_speech_recognition(audio)

    # Image classification
    client.image_classification("cat.jpg")

    # Image segmentation
    for item in client.image_segmentation("cat.jpg"):
        item["mask"].save(f"cat_{item['label']}_{item['score']}.jpg")

    # NLP
    client.summarization("The Eiffel tower...")
    client.conversational("Hi, who are you?")
