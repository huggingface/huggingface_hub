# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import base64
import io
import json
import os
import string
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from huggingface_hub import (
    ChatCompletionOutput,
    ChatCompletionOutputComplete,
    ChatCompletionStreamOutput,
    DocumentQuestionAnsweringOutputElement,
    FillMaskOutputElement,
    ImageClassificationOutputElement,
    ImageToTextOutput,
    InferenceClient,
    ObjectDetectionBoundingBox,
    ObjectDetectionOutputElement,
    QuestionAnsweringOutputElement,
    TableQuestionAnsweringOutputElement,
    TextClassificationOutputElement,
    TokenClassificationOutputElement,
    TranslationOutput,
    VisualQuestionAnsweringOutputElement,
    ZeroShotClassificationOutputElement,
    hf_hub_download,
)
from huggingface_hub.errors import HfHubHTTPError, ValidationError
from huggingface_hub.inference._common import (
    MimeBytes,
    _as_url,
    _open_as_mime_bytes,
    _stream_chat_completion_response,
    _stream_text_generation_response,
)
from huggingface_hub.inference._providers import get_provider_helper
from huggingface_hub.inference._providers.hf_inference import _build_chat_completion_url

from .testing_utils import with_production_testing


# Avoid calling APIs in VCRed tests
_RECOMMENDED_MODELS_FOR_VCR = {
    "black-forest-labs": {
        "text-to-image": "black-forest-labs/FLUX.1-dev",
    },
    "cerebras": {
        "conversational": "meta-llama/Llama-3.3-70B-Instruct",
    },
    "together": {
        "conversational": "meta-llama/Meta-Llama-3-8B-Instruct",
        "text-generation": "meta-llama/Llama-2-70b-hf",
        "text-to-image": "stabilityai/stable-diffusion-xl-base-1.0",
    },
    "fal-ai": {
        "text-to-image": "black-forest-labs/FLUX.1-dev",
        "automatic-speech-recognition": "openai/whisper-large-v3",
    },
    "fireworks-ai": {
        "conversational": "meta-llama/Llama-3.3-70B-Instruct",
    },
    "hf-inference": {
        "audio-classification": "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech",
        "audio-to-audio": "speechbrain/sepformer-wham",
        "automatic-speech-recognition": "jonatasgrosman/wav2vec2-large-xlsr-53-english",
        "conversational": "meta-llama/Llama-3.1-8B-Instruct",
        "document-question-answering": "naver-clova-ix/donut-base-finetuned-docvqa",
        "feature-extraction": "facebook/bart-base",
        "image-classification": "google/vit-base-patch16-224",
        "image-to-text": "Salesforce/blip-image-captioning-base",
        "image-segmentation": "facebook/detr-resnet-50-panoptic",
        "object-detection": "facebook/detr-resnet-50",
        "sentence-similarity": "sentence-transformers/all-MiniLM-L6-v2",
        "summarization": "sshleifer/distilbart-cnn-12-6",
        "table-question-answering": "google/tapas-base-finetuned-wtq",
        "tabular-classification": "julien-c/wine-quality",
        "tabular-regression": "scikit-learn/Fish-Weight",
        "text-classification": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "text-to-image": "CompVis/stable-diffusion-v1-4",
        "text-to-speech": "espnet/kan-bayashi_ljspeech_vits",
        "token-classification": "dbmdz/bert-large-cased-finetuned-conll03-english",
        "translation": "t5-small",
        "visual-question-answering": "dandelin/vilt-b32-finetuned-vqa",
        "zero-shot-classification": "facebook/bart-large-mnli",
        "zero-shot-image-classification": "openai/clip-vit-base-patch32",
    },
    "hyperbolic": {
        "text-generation": "meta-llama/Llama-3.1-405B",
        "conversational": "meta-llama/Llama-3.2-3B-Instruct",
        "text-to-image": "stabilityai/stable-diffusion-2",
    },
    "nebius": {
        "conversational": "meta-llama/Llama-3.1-8B-Instruct",
        "text-generation": "Qwen/Qwen2.5-32B-Instruct",
        "text-to-image": "stabilityai/stable-diffusion-xl-base-1.0",
    },
    "novita": {
        "text-generation": "NousResearch/Nous-Hermes-Llama2-13b",
        "conversational": "meta-llama/Llama-3.1-8B-Instruct",
    },
    "ovhcloud": {
        "conversational": "meta-llama/Llama-3.1-8B-Instruct",
    },
    "replicate": {
        "text-to-image": "ByteDance/SDXL-Lightning",
    },
    "sambanova": {
        "conversational": "meta-llama/Llama-3.1-8B-Instruct",
    },
}

CHAT_COMPLETION_MODEL = "HuggingFaceH4/zephyr-7b-beta"
CHAT_COMPLETION_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is deep learning?"},
]
CHAT_COMPLETE_NON_TGI_MODEL = "microsoft/DialoGPT-small"

CHAT_COMPLETION_TOOL_INSTRUCTIONS = [
    {
        "role": "system",
        "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.",
    },
    {
        "role": "user",
        "content": "What's the weather like the next 3 days in San Francisco, CA?",
    },
]
CHAT_COMPLETION_TOOLS = [  # 1 tool to get current weather, 1 to get N-day weather forecast
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                },
                "required": ["location", "format"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_n_day_weather_forecast",
            "description": "Get an N-day weather forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                    "num_days": {
                        "type": "integer",
                        "description": "The number of days to forecast",
                    },
                },
                "required": ["location", "format", "num_days"],
            },
        },
    },
]

CHAT_COMPLETION_RESPONSE_FORMAT_MESSAGE = [
    {
        "role": "user",
        "content": "I saw a puppy a cat and a raccoon during my bike ride in the park. What did I saw and when?",
    },
]

CHAT_COMPLETION_RESPONSE_FORMAT = {
    "type": "json_object",
    "value": {
        "properties": {
            "location": {"type": "string"},
            "activity": {"type": "string"},
            "animals_seen": {"type": "integer", "minimum": 1, "maximum": 5},
            "animals": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["location", "activity", "animals_seen", "animals"],
    },
}


def list_clients(task: str) -> list[pytest.param]:
    """Get list of clients for a specific task, with proper skip handling."""
    clients = []
    for provider, tasks in _RECOMMENDED_MODELS_FOR_VCR.items():
        if task in tasks:
            api_key = os.getenv("HF_INFERENCE_TEST_TOKEN")
            clients.append(
                pytest.param(
                    (provider, tasks[task], api_key),
                    id=f"{provider},{task}",
                )
            )
    return clients


@pytest.fixture()
@with_production_testing
def client(request):
    """
    Fixture to create client with proper skip handling.
    Note: VCR mode is only accessible through a fixture.
    """
    provider, model, api_key = request.param
    vcr_record_mode = request.config.getoption("--vcr-record")
    # If we are recording and the api key is not set, skip the test
    # replaying modes are "all", "new_episodes" and "once"
    # non replaying modes are "none" and None
    if vcr_record_mode not in ["none", None] and not api_key:
        pytest.skip(f"API KEY not set for provider {provider}, skipping test")

    # If api_key is provided, use it
    if api_key:
        return InferenceClient(model=model, provider=provider, token=api_key)

    # Otherwise use dummy token for VCR playback
    return InferenceClient(model=model, provider=provider, token="hf_dummy_token")


# Define fixtures for the files
@pytest.fixture(scope="module")
@with_production_testing
def audio_file():
    return hf_hub_download(repo_id="Narsil/image_dummy", repo_type="dataset", filename="sample1.flac")


@pytest.fixture(scope="module")
@with_production_testing
def image_file():
    return hf_hub_download(repo_id="Narsil/image_dummy", repo_type="dataset", filename="lena.png")


@pytest.fixture(scope="module")
@with_production_testing
def document_file():
    return hf_hub_download(repo_id="impira/docquery", repo_type="space", filename="contract.jpeg")


class TestBase:
    @pytest.fixture(autouse=True)
    def setup(self, audio_file, image_file, document_file):
        self.audio_file = audio_file
        self.image_file = image_file
        self.document_file = document_file

    @pytest.fixture(autouse=True)
    def mock_recommended_models(self, monkeypatch):
        def mock_fetch():
            return _RECOMMENDED_MODELS_FOR_VCR["hf-inference"]

        monkeypatch.setattr("huggingface_hub.inference._providers.hf_inference._fetch_recommended_models", mock_fetch)


@with_production_testing
@pytest.mark.skip("Temporary skipping tests for InferenceClient")
class TestInferenceClient(TestBase):
    @pytest.mark.parametrize("client", list_clients("audio-classification"), indirect=True)
    def test_audio_classification(self, client: InferenceClient):
        output = client.audio_classification(self.audio_file)
        assert isinstance(output, list)
        assert len(output) > 0
        for item in output:
            assert isinstance(item.score, float)
            assert isinstance(item.label, str)

    @pytest.mark.parametrize("client", list_clients("audio-to-audio"), indirect=True)
    def test_audio_to_audio(self, client: InferenceClient):
        output = client.audio_to_audio(self.audio_file)
        assert isinstance(output, list)
        assert len(output) > 0
        for item in output:
            assert isinstance(item.label, str)
            assert isinstance(item.blob, bytes)
            assert item.content_type == "audio/flac"

    @pytest.mark.parametrize("client", list_clients("automatic-speech-recognition"), indirect=True)
    def test_automatic_speech_recognition(self, client: InferenceClient):
        output = client.automatic_speech_recognition(self.audio_file)
        # Remove punctuation from the output
        normalized_output = output.text.translate(str.maketrans("", "", string.punctuation))
        assert normalized_output.lower().strip() == "a man said to the universe sir i exist"

    @pytest.mark.parametrize("client", list_clients("conversational"), indirect=True)
    def test_chat_completion_no_stream(self, client: InferenceClient):
        output = client.chat_completion(messages=CHAT_COMPLETION_MESSAGES, stream=False)
        assert isinstance(output, ChatCompletionOutput)
        assert output.created < time.time()
        assert isinstance(output.choices, list)
        assert len(output.choices) == 1
        assert isinstance(output.choices[0], ChatCompletionOutputComplete)

    @pytest.mark.parametrize("client", list_clients("conversational"), indirect=True)
    def test_chat_completion_with_stream(self, client: InferenceClient):
        output = list(
            client.chat_completion(
                messages=CHAT_COMPLETION_MESSAGES,
                stream=True,
                max_tokens=20,
            )
        )

        assert isinstance(output, list)
        assert all(isinstance(item, ChatCompletionStreamOutput) for item in output)
        # All items except the last one have a single choice with role/content delta
        for item in output[:-1]:
            assert len(item.choices) == 1
            assert item.choices[0].finish_reason is None
            assert item.choices[0].index == 0

        # Last item has a finish reason
        assert output[-1].choices[0].finish_reason == "length"

    def test_chat_completion_with_non_tgi(self) -> None:
        client = InferenceClient(provider="hf-inference")
        output = client.chat_completion(
            messages=CHAT_COMPLETION_MESSAGES,
            model=CHAT_COMPLETE_NON_TGI_MODEL,
            stream=False,
            max_tokens=20,
        )
        assert isinstance(output, ChatCompletionOutput)
        assert output.model == "microsoft/DialoGPT-small"
        assert len(output.choices) == 1

    @pytest.mark.skip(reason="Schema not aligned between providers")
    @pytest.mark.parametrize("client", list_clients("conversational"), indirect=True)
    def test_chat_completion_with_tool(self, client: InferenceClient):
        response = client.chat_completion(
            messages=CHAT_COMPLETION_TOOL_INSTRUCTIONS,
            tools=CHAT_COMPLETION_TOOLS,
            tool_choice="auto",
            max_tokens=500,
        )
        output = response.choices[0]

        # Single message before EOS
        assert output.finish_reason in ["tool_calls", "eos_token", "stop"]
        assert output.index == 0
        assert output.message.role == "assistant"

        # No content but a tool call
        assert output.message.content is None
        assert len(output.message.tool_calls) == 1

        # Tool
        tool_call = output.message.tool_calls[0]
        assert tool_call.type == "function"
        # Since tool_choice="auto", we can't know which tool will be called
        assert tool_call.function.name in ["get_n_day_weather_forecast", "get_current_weather"]
        args = tool_call.function.arguments
        if isinstance(args, str):
            args = json.loads(args)
        assert args["format"] in ["fahrenheit", "celsius"]
        assert args["location"] == "San Francisco, CA"
        assert args["num_days"] == 3

        # Now, test with tool_choice="get_current_weather"
        response = client.chat_completion(
            messages=CHAT_COMPLETION_TOOL_INSTRUCTIONS,
            tools=CHAT_COMPLETION_TOOLS,
            tool_choice={
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                },
            },
            max_tokens=500,
        )
        output = response.choices[0]
        tool_call = output.message.tool_calls[0]
        assert tool_call.function.name == "get_current_weather"
        # No need for 'num_days' with this tool
        assert tool_call.function.arguments == {
            "format": "fahrenheit",
            "location": "San Francisco, CA",
        }

    @pytest.mark.skip(reason="Schema not aligned between providers")
    @pytest.mark.parametrize("client", list_clients("conversational"), indirect=True)
    def test_chat_completion_with_response_format(self, client: InferenceClient):
        response = client.chat_completion(
            messages=CHAT_COMPLETION_RESPONSE_FORMAT_MESSAGE,
            response_format=CHAT_COMPLETION_RESPONSE_FORMAT,
            max_tokens=500,
        )
        output = response.choices[0].message.content
        assert json.loads(output) == {
            "activity": "biking",
            "animals": ["puppy", "cat", "raccoon"],
            "animals_seen": 3,
            "location": "park",
        }

    def test_chat_completion_unprocessable_entity(self) -> None:
        """Regression test for #2225.

        See https://github.com/huggingface/huggingface_hub/issues/2225.
        """
        client = InferenceClient(provider="hf-inference")
        with pytest.raises(HfHubHTTPError):
            client.chat_completion(
                "please output 'Observation'",  # Not a list of messages
                stop=["Observation", "Final Answer"],
                max_tokens=200,
                model="meta-llama/Meta-Llama-3-70B-Instruct",
            )

    @pytest.mark.parametrize("client", list_clients("document-question-answering"), indirect=True)
    def test_document_question_answering(self, client: InferenceClient):
        output = client.document_question_answering(self.document_file, "What is the purchase amount?")
        assert output == [
            DocumentQuestionAnsweringOutputElement(
                answer="$1,0000,000,00",
                end=None,
                score=None,
                start=None,
            )
        ]

    @pytest.mark.parametrize("client", list_clients("feature-extraction"), indirect=True)
    def test_feature_extraction_with_transformers(self, client: InferenceClient):
        embedding = client.feature_extraction("Hi, who are you?")
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1, 8, 768)

    @pytest.mark.parametrize("client", list_clients("feature-extraction"), indirect=True)
    def test_feature_extraction_with_sentence_transformers(self, client: InferenceClient):
        embedding = client.feature_extraction("Hi, who are you?")
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1, 8, 768)

    @pytest.mark.parametrize("client", list_clients("fill-mask"), indirect=True)
    def test_fill_mask(self, client: InferenceClient):
        output = client.fill_mask("The goal of life is <mask>.")
        assert output == [
            FillMaskOutputElement(
                score=0.06897063553333282,
                sequence="The goal of life is happiness.",
                token=11098,
                token_str=" happiness",
                fill_mask_output_token_str=None,
            ),
            FillMaskOutputElement(
                score=0.06554922461509705,
                sequence="The goal of life is immortality.",
                token=45075,
                token_str=" immortality",
                fill_mask_output_token_str=None,
            ),
            FillMaskOutputElement(
                score=0.0323575921356678,
                sequence="The goal of life is yours.",
                token=14314,
                token_str=" yours",
                fill_mask_output_token_str=None,
            ),
            FillMaskOutputElement(
                score=0.02431388944387436,
                sequence="The goal of life is liberation.",
                token=22211,
                token_str=" liberation",
                fill_mask_output_token_str=None,
            ),
            FillMaskOutputElement(
                score=0.023767812177538872,
                sequence="The goal of life is simplicity.",
                token=25342,
                token_str=" simplicity",
                fill_mask_output_token_str=None,
            ),
        ]

    def test_hf_inference_get_recommended_model_has_recommendation(self) -> None:
        from huggingface_hub.inference._providers.hf_inference import HFInferenceTask

        HFInferenceTask("feature-extraction")._prepare_mapping_info(None).provider_id == "facebook/bart-base"
        HFInferenceTask("translation")._prepare_mapping_info(None).provider_id == "t5-small"

    def test_hf_inference_get_recommended_model_no_recommendation(self) -> None:
        from huggingface_hub.inference._providers.hf_inference import HFInferenceTask

        with pytest.raises(ValueError):
            HFInferenceTask("text-generation")._prepare_mapping_info(None)

    @pytest.mark.parametrize("client", list_clients("image-classification"), indirect=True)
    def test_image_classification(self, client: InferenceClient):
        output = client.image_classification(self.image_file)
        assert output == [
            ImageClassificationOutputElement(label="brassiere, bra, bandeau", score=0.11767438799142838),
            ImageClassificationOutputElement(label="sombrero", score=0.09572819620370865),
            ImageClassificationOutputElement(label="cowboy hat, ten-gallon hat", score=0.0900089293718338),
            ImageClassificationOutputElement(label="bonnet, poke bonnet", score=0.06615174561738968),
            ImageClassificationOutputElement(label="fur coat", score=0.061511047184467316),
        ]

    @pytest.mark.parametrize("client", list_clients("image-segmentation"), indirect=True)
    def test_image_segmentation(self, client: InferenceClient):
        output = client.image_segmentation(self.image_file)
        assert isinstance(output, list)
        assert len(output) > 0
        for item in output:
            assert isinstance(item.score, float)
            assert isinstance(item.label, str)
            assert isinstance(item.mask, Image.Image)
            assert item.mask.height == 512
            assert item.mask.width == 512

    # ERROR 500 from server
    # Only during tests, not when running locally. Has to be investigated.
    # def test_image_to_image(self) -> None:
    #     image = self.client.image_to_image(self.image_file, prompt="turn the woman into a man")

    @pytest.mark.parametrize("client", list_clients("image-to-text"), indirect=True)
    def test_image_to_text(self, client: InferenceClient):
        caption = client.image_to_text(self.image_file)
        assert isinstance(caption, ImageToTextOutput)
        assert caption.generated_text == "a woman in a hat and dress posing for a photo"

    @pytest.mark.parametrize("client", list_clients("object-detection"), indirect=True)
    def test_object_detection(self, client: InferenceClient):
        output = client.object_detection(self.image_file)
        assert output == [
            ObjectDetectionOutputElement(
                box=ObjectDetectionBoundingBox(
                    xmin=59,
                    ymin=39,
                    xmax=420,
                    ymax=510,
                ),
                label="person",
                score=0.9486680030822754,
            ),
            ObjectDetectionOutputElement(
                box=ObjectDetectionBoundingBox(
                    xmin=143,
                    ymin=4,
                    xmax=510,
                    ymax=387,
                ),
                label="umbrella",
                score=0.5733323693275452,
            ),
            ObjectDetectionOutputElement(
                box=ObjectDetectionBoundingBox(
                    xmin=60,
                    ymin=162,
                    xmax=413,
                    ymax=510,
                ),
                label="person",
                score=0.5082724094390869,
            ),
        ]

    @pytest.mark.parametrize("client", list_clients("question-answering"), indirect=True)
    def test_question_answering(self, client: InferenceClient):
        output = client.question_answering(question="What is the meaning of life?", context="42")
        assert output == QuestionAnsweringOutputElement(answer="42", end=2, score=1.4291124728060822e-08, start=0)

    @pytest.mark.parametrize("client", list_clients("sentence-similarity"), indirect=True)
    def test_sentence_similarity(self, client: InferenceClient):
        scores = client.sentence_similarity(
            "Machine learning is so easy.",
            other_sentences=[
                "Deep learning is so straightforward.",
                "This is so difficult, like rocket science.",
                "I can't believe how much I struggled with this.",
            ],
        )
        assert scores == [0.7785724997520447, 0.4587624967098236, 0.29062220454216003]

    @pytest.mark.parametrize("client", list_clients("summarization"), indirect=True)
    def test_summarization(self, client: InferenceClient):
        summary = client.summarization("The sky is blue, the tree is green.")
        assert isinstance(summary.summary_text, str)

    @pytest.mark.skip(reason="This model is not available on InferenceAPI")
    @pytest.mark.parametrize("client", list_clients("tabular-classification"), indirect=True)
    def test_tabular_classification(self, client: InferenceClient):
        table = {
            "fixed_acidity": ["7.4", "7.8", "10.3"],
            "volatile_acidity": ["0.7", "0.88", "0.32"],
            "citric_acid": ["0", "0", "0.45"],
            "residual_sugar": ["1.9", "2.6", "6.4"],
            "chlorides": ["0.076", "0.098", "0.073"],
            "free_sulfur_dioxide": ["11", "25", "5"],
            "total_sulfur_dioxide": ["34", "67", "13"],
            "density": ["0.9978", "0.9968", "0.9976"],
            "pH": ["3.51", "3.2", "3.23"],
            "sulphates": ["0.56", "0.68", "0.82"],
            "alcohol": ["9.4", "9.8", "12.6"],
        }
        output = client.tabular_classification(table=table)
        assert output == ["5", "5", "5"]

    @pytest.mark.skip(reason="This model is not available on InferenceAPI")
    @pytest.mark.parametrize("client", list_clients("tabular-regression"), indirect=True)
    def test_tabular_regression(self, client: InferenceClient):
        table = {
            "Height": ["11.52", "12.48", "12.3778"],
            "Length1": ["23.2", "24", "23.9"],
            "Length2": ["25.4", "26.3", "26.5"],
            "Length3": ["30", "31.2", "31.1"],
            "Species": ["Bream", "Bream", "Bream"],
            "Width": ["4.02", "4.3056", "4.6961"],
        }
        output = client.tabular_regression(table=table)
        assert output == [110, 120, 130]

    @pytest.mark.parametrize("client", list_clients("table-question-answering"), indirect=True)
    def test_table_question_answering(self, client: InferenceClient):
        table = {
            "Repository": ["Transformers", "Datasets", "Tokenizers"],
            "Stars": ["36542", "4512", "3934"],
        }
        query = "How many stars does the transformers repository have?"
        output = client.table_question_answering(query=query, table=table)
        assert output == TableQuestionAnsweringOutputElement(
            answer="AVERAGE > 36542", cells=["36542"], coordinates=[[0, 1]], aggregator="AVERAGE"
        )

    @pytest.mark.parametrize("client", list_clients("text-classification"), indirect=True)
    def test_text_classification(self, client: InferenceClient):
        output = client.text_classification("I like you")
        assert output == [
            TextClassificationOutputElement(label="POSITIVE", score=0.9998695850372314),
            TextClassificationOutputElement(label="NEGATIVE", score=0.00013043530634604394),
        ]

    def test_text_generation(self) -> None:
        """Tested separately in `test_inference_text_generation.py`."""

    @pytest.mark.parametrize("client", list_clients("text-to-image"), indirect=True)
    def test_text_to_image_default(self, client: InferenceClient):
        image = client.text_to_image("An astronaut riding a horse on the moon.")
        assert isinstance(image, Image.Image)

    @pytest.mark.skip(reason="Need to check why fal.ai doesn't take image_size into account")
    @pytest.mark.parametrize("client", list_clients("text-to-image"), indirect=True)
    def test_text_to_image_with_parameters(self, client: InferenceClient):
        image = client.text_to_image("An astronaut riding a horse on the moon.", height=256, width=256)
        assert isinstance(image, Image.Image)
        assert image.height == 256
        assert image.width == 256

    @pytest.mark.parametrize("client", list_clients("text-to-speech"), indirect=True)
    def test_text_to_speech(self, client: InferenceClient):
        audio = client.text_to_speech("Hello world")
        assert isinstance(audio, bytes)

    @pytest.mark.parametrize("client", list_clients("translation"), indirect=True)
    def test_translation(self, client: InferenceClient):
        output = client.translation("Hello world")
        assert output == TranslationOutput(translation_text="Hallo Welt")

    @pytest.mark.parametrize("client", list_clients("translation"), indirect=True)
    def test_translation_with_source_and_target_language(self, client: InferenceClient):
        output_with_langs = client.translation(
            "Hello world", model="facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX", tgt_lang="fr_XX"
        )
        assert isinstance(output_with_langs, TranslationOutput)

        with pytest.raises(ValueError):
            client.translation("Hello world", model="facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX")

        with pytest.raises(ValueError):
            client.translation("Hello world", model="facebook/mbart-large-50-many-to-many-mmt", tgt_lang="en_XX")

    @pytest.mark.parametrize("client", list_clients("token-classification"), indirect=True)
    def test_token_classification(self, client: InferenceClient):
        output = client.token_classification(text="My name is Sarah Jessica Parker but you can call me Jessica")
        assert output == [
            TokenClassificationOutputElement(
                score=0.9991335868835449, end=31, entity_group="PER", start=11, word="Sarah Jessica Parker"
            ),
            TokenClassificationOutputElement(
                score=0.9979913234710693, end=59, entity_group="PER", start=52, word="Jessica"
            ),
        ]

    @pytest.mark.parametrize("client", list_clients("visual-question-answering"), indirect=True)
    def test_visual_question_answering(self, client: InferenceClient):
        output = client.visual_question_answering(image=self.image_file, question="Who's in the picture?")
        assert output == [
            VisualQuestionAnsweringOutputElement(score=0.9386942982673645, answer="woman"),
            VisualQuestionAnsweringOutputElement(score=0.3431190550327301, answer="girl"),
            VisualQuestionAnsweringOutputElement(score=0.08407800644636154, answer="lady"),
            VisualQuestionAnsweringOutputElement(score=0.05075192078948021, answer="female"),
            VisualQuestionAnsweringOutputElement(score=0.017771074548363686, answer="man"),
        ]

    @pytest.mark.parametrize("client", list_clients("zero-shot-classification"), indirect=True)
    def test_zero_shot_classification_single_label(self, client: InferenceClient):
        output = client.zero_shot_classification(
            "A new model offers an explanation for how the Galilean satellites formed around the solar system's"
            "largest world. Konstantin Batygin did not set out to solve one of the solar system's most puzzling"
            " mysteries when he went for a run up a hill in Nice, France.",
            candidate_labels=["space & cosmos", "scientific discovery", "microbiology", "robots", "archeology"],
        )
        assert output == [
            ZeroShotClassificationOutputElement(label="scientific discovery", score=0.796166181564331),
            ZeroShotClassificationOutputElement(label="space & cosmos", score=0.18570725619792938),
            ZeroShotClassificationOutputElement(label="microbiology", score=0.007308819331228733),
            ZeroShotClassificationOutputElement(label="archeology", score=0.0062583745457232),
            ZeroShotClassificationOutputElement(label="robots", score=0.004559362772852182),
        ]

    @pytest.mark.parametrize("client", list_clients("zero-shot-classification"), indirect=True)
    def test_zero_shot_classification_multi_label(self, client: InferenceClient):
        output = client.zero_shot_classification(
            text="A new model offers an explanation for how the Galilean satellites formed around the solar system's"
            "largest world. Konstantin Batygin did not set out to solve one of the solar system's most puzzling"
            " mysteries when he went for a run up a hill in Nice, France.",
            candidate_labels=["space & cosmos", "scientific discovery", "microbiology", "robots", "archeology"],
            multi_label=True,
        )
        assert output == [
            ZeroShotClassificationOutputElement(label="scientific discovery", score=0.9829296469688416),
            ZeroShotClassificationOutputElement(label="space & cosmos", score=0.7551906108856201),
            ZeroShotClassificationOutputElement(label="microbiology", score=0.0005462627159431577),
            ZeroShotClassificationOutputElement(label="archeology", score=0.0004713202069979161),
            ZeroShotClassificationOutputElement(label="robots", score=0.000304485292872414),
        ]

    @pytest.mark.parametrize("client", list_clients("zero-shot-image-classification"), indirect=True)
    def test_zero_shot_image_classification(self, client: InferenceClient):
        output = client.zero_shot_image_classification(
            image=self.image_file, candidate_labels=["tree", "woman", "cat"]
        )
        assert isinstance(output, list)
        assert len(output) > 0
        for item in output:
            assert isinstance(item.label, str)
            assert isinstance(item.score, float)


class TestOpenAsMimeBytes:
    @pytest.fixture(autouse=True)
    def setup(self, audio_file, image_file, document_file):
        self.audio_file = audio_file
        self.image_file = image_file
        self.document_file = document_file

    def test_open_as_mime_bytes_with_none(self) -> None:
        assert _open_as_mime_bytes(None) is None

    def test_open_as_mime_bytes_from_str_path(self) -> None:
        mime_bytes = _open_as_mime_bytes(self.image_file)
        assert isinstance(mime_bytes, MimeBytes)
        assert mime_bytes.mime_type == "image/png"

    def test_open_as_mime_bytes_from_pathlib_path(self) -> None:
        mime_bytes = _open_as_mime_bytes(Path(self.image_file))
        assert isinstance(mime_bytes, MimeBytes)
        assert mime_bytes.mime_type == "image/png"

    def test_open_as_mime_bytes_from_url(self) -> None:
        mime_bytes = _open_as_mime_bytes("https://huggingface.co/datasets/Narsil/image_dummy/resolve/main/tree.png")
        assert isinstance(mime_bytes, MimeBytes)
        assert mime_bytes.mime_type == "image/png"

    def test_open_as_mime_bytes_opened_file(self) -> None:
        with Path(self.image_file).open("rb") as f:
            mime_bytes = _open_as_mime_bytes(f)
        assert isinstance(mime_bytes, MimeBytes)
        assert mime_bytes.mime_type == "image/png"

    def test_open_as_mime_bytes_from_bytes(self) -> None:
        content_bytes = Path(self.image_file).read_bytes()
        mime_bytes = _open_as_mime_bytes(content_bytes)
        assert mime_bytes == content_bytes
        assert mime_bytes.mime_type is None

    def test_open_as_mime_bytes_from_pil_image(self) -> None:
        pil_image = Image.open(self.image_file)
        mime_bytes = _open_as_mime_bytes(pil_image)
        assert isinstance(mime_bytes, MimeBytes)
        assert mime_bytes.mime_type == "image/png"


class TestMimeBytes:
    """Tests for the MimeBytes subclass."""

    def test_mime_bytes_with_mime_type(self):
        """Test creating MimeBytes with data and mime type."""
        data = b"hello world"
        mime_type = "text/plain"
        mb = MimeBytes(data, mime_type)

        assert isinstance(mb, bytes)
        assert isinstance(mb, MimeBytes)
        assert mb == data
        assert mb.mime_type == mime_type

    def test_mime_bytes_without_mime_type(self):
        """Test creating MimeBytes without mime type defaults to None."""
        data = b"test data"
        mb = MimeBytes(data)

        assert isinstance(mb, MimeBytes)
        assert mb == data
        assert mb.mime_type is None

    def test_mime_bytes_with_none_mime_type(self):
        """Test creating MimeBytes with explicit None mime type."""
        data = b"test data"
        mb = MimeBytes(data, None)

        assert isinstance(mb, MimeBytes)
        assert mb == data
        assert mb.mime_type is None

    def test_mime_bytes_from_mime_bytes(self):
        """Test that mime_type is inherited from source MimeBytes when none provided."""
        original_data = b"original content"
        original_mime = "text/html"
        original_mb = MimeBytes(original_data, original_mime)

        # Create new MimeBytes from existing one without specifying mime_type
        new_mb = MimeBytes(original_mb)

        assert isinstance(new_mb, MimeBytes)
        assert new_mb == original_data
        assert new_mb.mime_type == original_mime

    def test_mime_type_override_when_provided(self):
        """Test that mime_type is overridden when explicitly provided."""
        original_data = b"original content"
        original_mime = "text/html"
        original_mb = MimeBytes(original_data, original_mime)

        new_mime = "application/json"
        new_mb = MimeBytes(original_mb, new_mime)

        assert isinstance(new_mb, MimeBytes)
        assert new_mb == original_data
        assert new_mb.mime_type == new_mime


class TestHeadersAndCookies(TestBase):
    def test_headers_and_cookies(self) -> None:
        client = InferenceClient(headers={"X-My-Header": "foo"}, cookies={"my-cookie": "bar"})
        assert client.headers["X-My-Header"] == "foo"
        assert client.cookies["my-cookie"] == "bar"

    @patch("huggingface_hub.inference._client._bytes_to_image")
    @patch("huggingface_hub.inference._client.get_session")
    @patch("huggingface_hub.inference._providers.hf_inference._check_supported_task")
    def test_accept_header_image(
        self,
        check_supported_task_mock: MagicMock,
        get_session_mock: MagicMock,
        bytes_to_image_mock: MagicMock,
    ) -> None:
        """Test that Accept: image/png header is set for image tasks."""
        client = InferenceClient(provider="hf-inference")

        response = client.text_to_image("An astronaut riding a horse")
        assert response == bytes_to_image_mock.return_value

        headers = get_session_mock().stream.call_args_list[0].kwargs["headers"]
        assert headers["Accept"] == "image/png"


@with_production_testing
@pytest.mark.skip("Temporary skipping tests for TestOpenAICompatibility")
class TestOpenAICompatibility(TestBase):
    def test_base_url_and_api_key(self):
        client = InferenceClient(
            provider="hf-inference",
            base_url="https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct",
            api_key=os.getenv("HF_INFERENCE_TEST_TOKEN"),
        )
        output = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Count to 10"},
            ],
            stream=False,
            max_tokens=1024,
        )
        assert "1, 2, 3, 4, 5, 6, 7, 8, 9, 10" in output.choices[0].message.content

    def test_without_base_url(self):
        client = InferenceClient(
            provider="hf-inference",
            token=os.getenv("HF_INFERENCE_TEST_TOKEN"),
        )
        output = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Count to 10"},
            ],
            stream=False,
            max_tokens=1024,
        )
        assert "1, 2, 3, 4, 5, 6, 7, 8, 9, 10" in output.choices[0].message.content

    def test_with_stream_true(self):
        client = InferenceClient(
            provider="hf-inference",
            token=os.getenv("HF_INFERENCE_TEST_TOKEN"),
        )
        output = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Count to 10"},
            ],
            stream=True,
            max_tokens=1024,
        )

        chunked_text = [chunk.choices[0].delta.content for chunk in output]
        assert len(chunked_text) == 30
        output_text = "".join(chunked_text)
        assert "1, 2, 3, 4, 5, 6, 7, 8, 9, 10" in output_text

    def test_model_and_base_url_mutually_exclusive(self):
        with pytest.raises(ValueError):
            InferenceClient(model="meta-llama/Meta-Llama-3-8B-Instruct", base_url="http://127.0.0.1:8000")

    def test_token_initialization_from_token(self):
        # Test with explicit token
        client = InferenceClient(token="my-token")
        assert client.token == "my-token"

    def test_token_initialization_from_api_key(self):
        # Test with api_key
        client = InferenceClient(api_key="my-api-key")
        assert client.token == "my-api-key"

    def test_token_initialization_cannot_be_both(self):
        # Test with both token and api_key raises error
        with pytest.raises(ValueError, match="Received both `token` and `api_key` arguments"):
            InferenceClient(token="my-token", api_key="my-api-key")

    def test_token_initialization_default_to_none(self):
        # Test with token=None (default behavior)
        client = InferenceClient()
        assert client.token is None

    def test_token_initialization_with_token_true(self, mocker):
        # Test with token=True and token is set with get_token()
        mocker.patch("huggingface_hub.inference._client.get_token", return_value="my-token")
        client = InferenceClient(token=True)
        assert client.token == "my-token"

    def test_token_initialization_cannot_be_token_false(self):
        # Test with token=False raises error
        with pytest.raises(ValueError, match="Cannot use `token=False` to disable authentication"):
            InferenceClient(token=False)


@pytest.mark.parametrize(
    "stop_signal",
    [
        "data: [DONE]",
        "data: [DONE]\n",
        "data: [DONE] ",
    ],
)
def test_stream_text_generation_response(stop_signal: bytes):
    data = [
        'data: {"index":1,"token":{"id":4560,"text":" trying","logprob":-2.078125,"special":false},"generated_text":null,"details":null}',
        "",  # Empty line is skipped
        "\n",  # Newline is skipped
        'data: {"index":2,"token":{"id":311,"text":" to","logprob":-0.026245117,"special":false},"generated_text":" trying to","details":null}',
        stop_signal,  # Stop signal
        # Won't parse after
        'data: {"index":2,"token":{"id":311,"text":" to","logprob":-0.026245117,"special":false},"generated_text":" trying to","details":null}',
    ]
    output = list(_stream_text_generation_response(data, details=False))
    assert len(output) == 2
    assert output == [" trying", " to"]


@pytest.mark.parametrize(
    "stop_signal",
    [
        "data: [DONE]",
        "data: [DONE]\n",
        "data: [DONE] ",
    ],
)
def test_stream_chat_completion_response(stop_signal: bytes):
    data = [
        'data: {"object":"chat.completion.chunk","id":"","created":1721737661,"model":"","system_fingerprint":"2.1.2-dev0-sha-5fca30e","choices":[{"index":0,"delta":{"role":"assistant","content":"Both"},"logprobs":null,"finish_reason":null}]}',
        "",  # Empty line is skipped
        "\n",  # Newline is skipped
        'data: {"object":"chat.completion.chunk","id":"","created":1721737661,"model":"","system_fingerprint":"2.1.2-dev0-sha-5fca30e","choices":[{"index":0,"delta":{"role":"assistant","content":" Rust"},"logprobs":null,"finish_reason":null}]}',
        stop_signal,  # Stop signal
        # Won't parse after
        'data: {"index":2,"token":{"id":311,"text":" to","logprob":-0.026245117,"special":false},"generated_text":" trying to","details":null}',
    ]
    output = list(_stream_chat_completion_response(data))
    assert len(output) == 2
    assert output[0].choices[0].delta.content == "Both"
    assert output[1].choices[0].delta.content == " Rust"


def test_chat_completion_error_in_stream():
    """
    Regression test for https://github.com/huggingface/huggingface_hub/issues/2514.
    When an error is encountered in the stream, it should raise a TextGenerationError (e.g. a ValidationError).
    """
    data = [
        'data: {"object":"chat.completion.chunk","id":"","created":1721737661,"model":"","system_fingerprint":"2.1.2-dev0-sha-5fca30e","choices":[{"index":0,"delta":{"role":"assistant","content":"Both"},"logprobs":null,"finish_reason":null}]}',
        'data: {"error":"Input validation error: `inputs` tokens + `max_new_tokens` must be <= 4096. Given: 6 `inputs` tokens and 4091 `max_new_tokens`","error_type":"validation"}',
    ]
    with pytest.raises(ValidationError):
        for token in _stream_chat_completion_response(data):
            pass


INFERENCE_API_URL = "https://api-inference.huggingface.co/models"
INFERENCE_ENDPOINT_URL = "https://rur2d6yoccusjxgn.us-east-1.aws.endpoints.huggingface.cloud"  # example
LOCAL_TGI_URL = "http://0.0.0.0:8080"


@pytest.mark.parametrize(
    ("model_url", "expected_url"),
    [
        # Inference API
        (
            f"{INFERENCE_API_URL}/username/repo_name",
            f"{INFERENCE_API_URL}/username/repo_name/v1/chat/completions",
        ),
        # Inference Endpoint
        (
            INFERENCE_ENDPOINT_URL,
            f"{INFERENCE_ENDPOINT_URL}/v1/chat/completions",
        ),
        # Inference Endpoint - full url
        (
            f"{INFERENCE_ENDPOINT_URL}/v1/chat/completions",
            f"{INFERENCE_ENDPOINT_URL}/v1/chat/completions",
        ),
        # Inference Endpoint - url with '/v1' (OpenAI compatibility)
        (
            f"{INFERENCE_ENDPOINT_URL}/v1",
            f"{INFERENCE_ENDPOINT_URL}/v1/chat/completions",
        ),
        # Inference Endpoint - url with '/v1/' (OpenAI compatibility)
        (
            f"{INFERENCE_ENDPOINT_URL}/v1/",
            f"{INFERENCE_ENDPOINT_URL}/v1/chat/completions",
        ),
        # Local TGI with trailing '/v1'
        (
            f"{LOCAL_TGI_URL}/v1",
            f"{LOCAL_TGI_URL}/v1/chat/completions",
        ),
        # With query parameters
        (
            f"{INFERENCE_ENDPOINT_URL}/v1/chat/completions?api-version=1",
            f"{INFERENCE_ENDPOINT_URL}/v1/chat/completions?api-version=1",
        ),
        (
            f"{INFERENCE_ENDPOINT_URL}/chat/completions?api-version=1",
            f"{INFERENCE_ENDPOINT_URL}/chat/completions?api-version=1",
        ),
        (
            f"{INFERENCE_ENDPOINT_URL}?api-version=1",
            f"{INFERENCE_ENDPOINT_URL}/v1/chat/completions?api-version=1",
        ),
        (
            f"{INFERENCE_ENDPOINT_URL}/v1?api-version=1",
            f"{INFERENCE_ENDPOINT_URL}/v1/chat/completions?api-version=1",
        ),
        (
            f"{INFERENCE_ENDPOINT_URL}/?api-version=1",
            f"{INFERENCE_ENDPOINT_URL}/v1/chat/completions?api-version=1",
        ),
    ],
)
def test_resolve_chat_completion_url(model_url: str, expected_url: str):
    url = _build_chat_completion_url(model_url)
    assert url == expected_url


def test_pass_url_as_base_url():
    client = InferenceClient(
        provider="hf-inference",
        base_url="http://localhost:8082/v1/",
    )
    provider = get_provider_helper("hf-inference", "text-generation", "test-model")
    request = provider.prepare_request(
        inputs="The huggingface_hub library is ", parameters={}, headers={}, model=client.model, api_key=None
    )
    assert request.url == "http://localhost:8082/v1/"


def test_cannot_pass_token_false():
    """Regression test for #2853.

    It is no longer possible to pass `token=False` to the InferenceClient constructor.
    This was a legacy behavior, broken since 0.28.x release as passing token=False does not prevent the token from being
    used. Better to drop this feature altogether and raise an error if `token=False` is passed.

    See https://github.com/huggingface/huggingface_hub/pull/2853.
    """
    with pytest.raises(ValueError):
        InferenceClient(token=False)


class TestBillToOrganization:
    def test_bill_to_added_to_new_headers(self):
        client = InferenceClient(bill_to="huggingface_hub")
        assert client.headers["X-HF-Bill-To"] == "huggingface_hub"

    def test_bill_to_added_to_existing_headers(self):
        headers = {"foo": "bar"}
        client = InferenceClient(bill_to="huggingface_hub", headers=headers)
        assert client.headers["X-HF-Bill-To"] == "huggingface_hub"
        assert client.headers["foo"] == "bar"
        assert headers == {"foo": "bar"}  # do not mutate the original headers

    def test_warning_if_bill_to_already_set(self):
        headers = {"X-HF-Bill-To": "huggingface"}
        with pytest.warns(UserWarning, match="Overriding existing 'huggingface' value in headers with 'openai'."):
            client = InferenceClient(bill_to="openai", headers=headers)
        assert client.headers["X-HF-Bill-To"] == "openai"
        assert headers == {"X-HF-Bill-To": "huggingface"}  # do not mutate the original headers

    def test_warning_if_bill_to_with_direct_calls(self):
        with pytest.warns(
            UserWarning,
            match="You've provided an external provider's API key, so requests will be billed directly by the provider.",
        ):
            InferenceClient(bill_to="openai", token="replicate_key", provider="replicate")


@pytest.mark.parametrize(
    "client_init_arg, init_kwarg_name, expected_request_url, expected_payload_model",
    [
        # passing a custom endpoint in the model argument
        pytest.param(
            "https://my-custom-endpoint.com/custom_path",
            "model",
            "https://my-custom-endpoint.com/custom_path/v1/chat/completions",
            "dummy",
            id="client_model_is_url",
        ),
        # passing a custom endpoint in the base_url argument
        pytest.param(
            "https://another-endpoint.com/v1/",
            "base_url",
            "https://another-endpoint.com/v1/chat/completions",
            "dummy",
            id="client_base_url_is_url",
        ),
        # passing a model ID
        pytest.param(
            "username/repo_name",
            "model",
            "https://router.huggingface.co/hf-inference/models/username/repo_name/v1/chat/completions",
            "username/repo_name",
            id="client_model_is_id",
        ),
        # passing a custom endpoint in the model argument
        pytest.param(
            "https://specific-chat-endpoint.com/v1/chat/completions",
            "model",
            "https://specific-chat-endpoint.com/v1/chat/completions",
            "dummy",
            id="client_model_is_full_chat_url",
        ),
        # passing a localhost URL in the model argument
        pytest.param(
            "http://localhost:8080",
            "model",
            "http://localhost:8080/v1/chat/completions",
            "dummy",
            id="client_model_is_localhost_url",
        ),
        # passing a localhost URL in the base_url argument
        pytest.param(
            "http://127.0.0.1:8000/custom/path/v1",
            "base_url",
            "http://127.0.0.1:8000/custom/path/v1/chat/completions",
            "dummy",
            id="client_base_url_is_localhost_ip_with_path",
        ),
    ],
)
def test_chat_completion_url_resolution(
    mocker, client_init_arg, init_kwarg_name, expected_request_url, expected_payload_model
):
    init_kwargs = {init_kwarg_name: client_init_arg, "provider": "hf-inference"}
    client = InferenceClient(**init_kwargs)

    mock_response_content = b'{"choices": [{"message": {"content": "Mock response"}}]}'
    mocker.patch(
        "huggingface_hub.inference._providers.hf_inference._check_supported_task",
        return_value=None,
    )

    with patch.object(InferenceClient, "_inner_post", return_value=mock_response_content) as mock_inner_post:
        client.chat_completion(messages=[{"role": "user", "content": "Hello?"}], stream=False)

        mock_inner_post.assert_called_once()

        request_params = mock_inner_post.call_args[0][0]
        assert request_params.url == expected_request_url
        assert request_params.json is not None
        assert request_params.json.get("model") == expected_payload_model


@pytest.mark.parametrize(
    "content_input, default_mime_type, expected, is_exact_match",
    [
        ("https://my-url.com/cat.gif", "image/jpeg", "https://my-url.com/cat.gif", True),
        ("assets/image.png", "image/jpeg", "data:image/png;base64,", False),
        (Path("assets/image.png"), "image/jpeg", "data:image/png;base64,", False),
        ("assets/image.foo", "image/jpeg", "data:image/jpeg;base64,", False),
        (b"some image bytes", "image/jpeg", "data:image/jpeg;base64,c29tZSBpbWFnZSBieXRlcw==", True),
        (io.BytesIO(b"some image bytes"), "image/jpeg", "data:image/jpeg;base64,c29tZSBpbWFnZSBieXRlcw==", True),
    ],
)
def test_as_url(content_input, default_mime_type, expected, is_exact_match, tmp_path: Path):
    if isinstance(content_input, (str, Path)) and not str(content_input).startswith("http"):
        file_path = tmp_path / content_input
        file_path.parent.mkdir(exist_ok=True, parents=True)
        file_path.touch()
        content_input = file_path

    result = _as_url(content_input, default_mime_type)
    if is_exact_match:
        assert result == expected
    else:
        assert result.startswith(expected)


def test_as_url_with_pil_image(image_file: str):
    """Test `_as_url` helper with a PIL Image."""
    pil_image = Image.open(image_file)

    pil_image.format = "PNG"
    png_url = _as_url(pil_image, default_mime_type="image/jpeg")
    assert png_url.startswith("data:image/png;base64,")

    pil_image.format = None
    png_url = _as_url(pil_image, default_mime_type="image/jpeg")
    assert png_url.startswith("data:image/png;base64,")

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    b64_encoded = base64.b64encode(buffer.getvalue()).decode()
    assert png_url == f"data:image/png;base64,{b64_encoded}"
