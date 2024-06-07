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
import io
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from huggingface_hub import (
    AutomaticSpeechRecognitionOutput,
    ChatCompletionOutput,
    ChatCompletionOutputComplete,
    ChatCompletionOutputMessage,
    ChatCompletionStreamOutput,
    DocumentQuestionAnsweringOutputElement,
    FillMaskOutputElement,
    ImageClassificationOutputElement,
    ImageToTextOutput,
    InferenceClient,
    ObjectDetectionOutputElement,
    QuestionAnsweringOutputElement,
    SummarizationOutput,
    TableQuestionAnsweringOutputElement,
    TextClassificationOutputElement,
    TokenClassificationOutputElement,
    TranslationOutput,
    VisualQuestionAnsweringOutputElement,
    ZeroShotClassificationOutputElement,
    hf_hub_download,
)
from huggingface_hub.constants import ALL_INFERENCE_API_FRAMEWORKS, MAIN_INFERENCE_API_FRAMEWORKS
from huggingface_hub.inference._client import _open_as_binary
from huggingface_hub.utils import HfHubHTTPError, build_hf_headers

from .testing_utils import expect_deprecation, with_production_testing


# Avoid call to hf.co/api/models in VCRed tests
_RECOMMENDED_MODELS_FOR_VCR = {
    "audio-classification": "speechbrain/google_speech_command_xvector",
    "audio-to-audio": "speechbrain/sepformer-wham",
    "automatic-speech-recognition": "facebook/wav2vec2-base-960h",
    "conversational": "facebook/blenderbot-400M-distill",
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
    "text-classification": "distilbert-base-uncased-finetuned-sst-2-english",
    "text-to-image": "CompVis/stable-diffusion-v1-4",
    "text-to-speech": "espnet/kan-bayashi_ljspeech_vits",
    "token-classification": "dbmdz/bert-large-cased-finetuned-conll03-english",
    "translation": "t5-small",
    "visual-question-answering": "dandelin/vilt-b32-finetuned-vqa",
    "zero-shot-classification": "facebook/bart-large-mnli",
    "zero-shot-image-classification": "openai/clip-vit-base-patch32",
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


class InferenceClientTest(unittest.TestCase):
    @classmethod
    @with_production_testing
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.image_file = hf_hub_download(repo_id="Narsil/image_dummy", repo_type="dataset", filename="lena.png")
        cls.document_file = hf_hub_download(repo_id="impira/docquery", repo_type="space", filename="contract.jpeg")
        cls.audio_file = hf_hub_download(repo_id="Narsil/image_dummy", repo_type="dataset", filename="sample1.flac")


@pytest.mark.vcr
@with_production_testing
@patch("huggingface_hub.inference._client._fetch_recommended_models", lambda: _RECOMMENDED_MODELS_FOR_VCR)
class InferenceClientVCRTest(InferenceClientTest):
    """
    Test for the main tasks implemented in InferenceClient. Since Inference API can be flaky, we use VCR.py and
    pytest-vcr to record and replay the inference calls (see https://pytest-vcr.readthedocs.io/en/latest/).

    Tips when adding new tasks:
    - Most of the time, we only test that the return values are correct. We don't always test the actual output of the model.
    - In the CI, VRC replay is always on. If you want to test locally against the server, you can use the `--vcr-mode`
      and `--disable-vcr` command line options. See https://pytest-vcr.readthedocs.io/en/latest/configuration/.
    - If you get rate-limited locally, you can use your own token when initializing InferenceClient.
      /!\\ WARNING: if you do so, you must delete the token from the cassette before committing!
    - If the model is not loaded on the server, you will save a lot of HTTP 503 responses in the cassette. We don't
      want those to be committed. Either delete them manually or rerun the test once the model is loaded on the server.
    """

    def setUp(self) -> None:
        super().setUp()
        self.client = InferenceClient()

    def test_audio_classification(self) -> None:
        output = self.client.audio_classification(self.audio_file)
        self.assertIsInstance(output, list)
        self.assertGreater(len(output), 0)
        for item in output:
            self.assertIsInstance(item.score, float)
            self.assertIsInstance(item.label, str)

    def test_audio_to_audio(self) -> None:
        output = self.client.audio_to_audio(self.audio_file)
        assert isinstance(output, list)
        assert len(output) > 0
        for item in output:
            assert isinstance(item.label, str)
            assert isinstance(item.blob, bytes)
            assert item.content_type == "audio/flac"

    def test_automatic_speech_recognition(self) -> None:
        output = self.client.automatic_speech_recognition(self.audio_file)
        assert output == AutomaticSpeechRecognitionOutput(
            text="A MAN SAID TO THE UNIVERSE SIR I EXIST",
            chunks=None,
        )

    def test_chat_completion_no_stream(self) -> None:
        output = self.client.chat_completion(
            messages=CHAT_COMPLETION_MESSAGES,
            model=CHAT_COMPLETION_MODEL,
            stream=False,
        )
        assert isinstance(output, ChatCompletionOutput)
        assert output.created < time.time()
        assert output.choices == [
            ChatCompletionOutputComplete(
                finish_reason="length",
                index=0,
                message=ChatCompletionOutputMessage(
                    content="Deep learning is a subfield of machine learning that focuses on training artificial neural networks with multiple layers of",
                    role="assistant",
                ),
            )
        ]

    def test_chat_completion_with_stream(self) -> None:
        output = list(
            self.client.chat_completion(
                messages=CHAT_COMPLETION_MESSAGES,
                model=CHAT_COMPLETION_MODEL,
                stream=True,
                max_tokens=20,
            )
        )

        assert isinstance(output, list)
        assert all(isinstance(item, ChatCompletionStreamOutput) for item in output)
        created = output[0].created
        assert all(item.created == created for item in output)  # all tokens share the same timestamp

        # All items except the last one have a single choice with role/content delta
        for item in output[:-1]:
            assert len(item.choices) == 1
            assert item.choices[0].finish_reason is None
            assert item.choices[0].index == 0
            assert item.choices[0].delta.role == "assistant"
            assert item.choices[0].delta.content is not None

        # Last item has a finish reason but no role/content delta
        assert output[-1].choices[0].finish_reason == "length"
        assert output[-1].choices[0].delta.role is None
        assert output[-1].choices[0].delta.content is None

        # Reconstruct generated text
        generated_text = "".join(
            item.choices[0].delta.content for item in output if item.choices[0].delta.content is not None
        )
        expected_text = "Deep learning is a subfield of machine learning that is based on artificial neural networks with multiple layers to"
        assert generated_text == expected_text

    def test_chat_completion_with_non_tgi(self) -> None:
        output = self.client.chat_completion(
            messages=CHAT_COMPLETION_MESSAGES,
            model=CHAT_COMPLETE_NON_TGI_MODEL,
            stream=False,
            max_tokens=20,
        )
        assert output == ChatCompletionOutput(
            id="dummy",
            model="dummy",
            object="dummy",
            system_fingerprint="dummy",
            usage=None,
            choices=[
                ChatCompletionOutputComplete(
                    finish_reason="unk",  # <- specific to models served with transformers (not possible to get details)
                    index=0,
                    message=ChatCompletionOutputMessage(
                        content="Deep learning is a thing.",
                        role="assistant",
                    ),
                )
            ],
            created=output.created,
        )

    def test_chat_completion_with_tool(self) -> None:
        response = self.client.chat_completion(
            model="meta-llama/Meta-Llama-3-70B-Instruct",
            messages=CHAT_COMPLETION_TOOL_INSTRUCTIONS,
            tools=CHAT_COMPLETION_TOOLS,
            tool_choice="auto",
            max_tokens=500,
        )
        output = response.choices[0]

        # Single message before EOS
        assert output.finish_reason == "eos_token"
        assert output.index == 0
        assert output.message.role == "assistant"

        # No content but a tool call
        assert output.message.content is None
        assert len(output.message.tool_calls) == 1

        # Tool
        tool_call = output.message.tool_calls[0]
        assert tool_call.type == "function"
        assert tool_call.function.name == "get_n_day_weather_forecast"
        assert tool_call.function.arguments == {
            "format": "fahrenheit",
            "location": "San Francisco, CA",
            "num_days": 3,
        }

        # Now, test with tool_choice="get_current_weather"
        response = self.client.chat_completion(
            model="meta-llama/Meta-Llama-3-70B-Instruct",
            messages=CHAT_COMPLETION_TOOL_INSTRUCTIONS,
            tools=CHAT_COMPLETION_TOOLS,
            tool_choice="get_current_weather",
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

    def test_chat_completion_unprocessable_entity(self) -> None:
        """Regression test for #2225.

        See https://github.com/huggingface/huggingface_hub/issues/2225.
        """
        with self.assertRaises(HfHubHTTPError):
            self.client.chat_completion(
                "please output 'Observation'",  # Not a list of messages
                stop=["Observation", "Final Answer"],
                max_tokens=200,
                model="meta-llama/Meta-Llama-3-70B-Instruct",
            )

    @expect_deprecation("InferenceClient.conversational")
    def test_conversational(self) -> None:
        output = self.client.conversational("Hi, who are you?")
        self.assertEqual(
            output,
            {
                "generated_text": "I am the one who knocks.",
                "conversation": {
                    "generated_responses": ["I am the one who knocks."],
                    "past_user_inputs": ["Hi, who are you?"],
                },
                "warnings": ["Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation."],
            },
        )

        output2 = self.client.conversational(
            "Wow, that's scary!",
            generated_responses=output["conversation"]["generated_responses"],
            past_user_inputs=output["conversation"]["past_user_inputs"],
        )
        self.assertEqual(
            output2,
            {
                "generated_text": "I am the one who knocks.",
                "conversation": {
                    "generated_responses": ["I am the one who knocks.", "I am the one who knocks."],
                    "past_user_inputs": ["Hi, who are you?", "Wow, that's scary!"],
                },
                "warnings": ["Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation."],
            },
        )

    def test_document_question_answering(self) -> None:
        output = self.client.document_question_answering(self.document_file, "What is the purchase amount?")
        self.assertEqual(
            output,
            [
                DocumentQuestionAnsweringOutputElement(
                    answer="$1,000,000,000", end=None, score=None, start=None, words=None
                )
            ],
        )

    def test_feature_extraction_with_transformers(self) -> None:
        embedding = self.client.feature_extraction("Hi, who are you?")
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape, (1, 8, 768))

    def test_feature_extraction_with_sentence_transformers(self) -> None:
        embedding = self.client.feature_extraction("Hi, who are you?", model="sentence-transformers/all-MiniLM-L6-v2")
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape, (384,))

    def test_fill_mask(self) -> None:
        model = "distilroberta-base"
        output = self.client.fill_mask("The goal of life is <mask>.", model=model)
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

    def test_get_recommended_model_has_recommendation(self) -> None:
        assert self.client.get_recommended_model("feature-extraction") == "facebook/bart-base"
        assert self.client.get_recommended_model("translation") == "t5-small"

    def test_get_recommended_model_no_recommendation(self) -> None:
        with pytest.raises(ValueError):
            self.client.get_recommended_model("text-generation")

    def test_image_classification(self) -> None:
        output = self.client.image_classification(self.image_file)
        assert output == [
            ImageClassificationOutputElement(label="brassiere, bra, bandeau", score=0.1176738440990448),
            ImageClassificationOutputElement(label="sombrero", score=0.0957278460264206),
            ImageClassificationOutputElement(label="cowboy hat, ten-gallon hat", score=0.09000881016254425),
            ImageClassificationOutputElement(label="bonnet, poke bonnet", score=0.06615243852138519),
            ImageClassificationOutputElement(label="fur coat", score=0.06151164695620537),
        ]

    def test_image_segmentation(self) -> None:
        output = self.client.image_segmentation(self.image_file)
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
    #     self.assertIsInstance(image, Image.Image)
    #     self.assertEqual(image.height, 512)
    #     self.assertEqual(image.width, 512)

    def test_image_to_text(self) -> None:
        caption = self.client.image_to_text(self.image_file)
        assert isinstance(caption, ImageToTextOutput)
        assert caption.generated_text == "a woman in a hat and dress posing for a photo"

    def test_object_detection(self) -> None:
        output = self.client.object_detection(self.image_file)
        assert output == [
            ObjectDetectionOutputElement(
                box={"xmin": 59, "ymin": 39, "xmax": 420, "ymax": 510}, label="person", score=0.9486683011054993
            )
        ]

    def test_question_answering(self) -> None:
        model = "deepset/roberta-base-squad2"
        output = self.client.question_answering(question="What is the meaning of life?", context="42", model=model)
        assert output == QuestionAnsweringOutputElement(answer="42", end=2, score=1.4291124728060822e-08, start=0)

    def test_sentence_similarity(self) -> None:
        scores = self.client.sentence_similarity(
            "Machine learning is so easy.",
            other_sentences=[
                "Deep learning is so straightforward.",
                "This is so difficult, like rocket science.",
                "I can't believe how much I struggled with this.",
            ],
        )
        self.assertEqual(scores, [0.7785726189613342, 0.45876261591911316, 0.2906220555305481])

    def test_summarization(self) -> None:
        summary = self.client.summarization(
            "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest"
            " structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its"
            " construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made"
            " structure in the world, a title it held for 41 years until the Chrysler Building in New York City was"
            " finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a"
            " broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2"
            " metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure"
            " in France after the Millau Viaduct."
        )
        assert summary == SummarizationOutput.parse_obj(
            {
                "summary_text": "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building. Its base is"
                " square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower"
                " surpassed the Washington Monument to become the tallest man-made structure in the world.",
            }
        )

    @pytest.mark.skip(reason="This model is not available on InferenceAPI")
    def test_tabular_classification(self) -> None:
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
        output = self.client.tabular_classification(table=table)
        self.assertEqual(output, ["5", "5", "5"])

    @pytest.mark.skip(reason="This model is not available on InferenceAPI")
    def test_tabular_regression(self) -> None:
        table = {
            "Height": ["11.52", "12.48", "12.3778"],
            "Length1": ["23.2", "24", "23.9"],
            "Length2": ["25.4", "26.3", "26.5"],
            "Length3": ["30", "31.2", "31.1"],
            "Species": ["Bream", "Bream", "Bream"],
            "Width": ["4.02", "4.3056", "4.6961"],
        }
        output = self.client.tabular_regression(table=table)
        self.assertEqual(output, [110, 120, 130])

    def test_table_question_answering(self) -> None:
        table = {
            "Repository": ["Transformers", "Datasets", "Tokenizers"],
            "Stars": ["36542", "4512", "3934"],
        }
        query = "How many stars does the transformers repository have?"
        output = self.client.table_question_answering(query=query, table=table)
        assert output == TableQuestionAnsweringOutputElement(
            answer="AVERAGE > 36542", cells=["36542"], coordinates=[[0, 1]], aggregator="AVERAGE"
        )

    def test_text_classification(self) -> None:
        output = self.client.text_classification("I like you")
        assert output == [
            TextClassificationOutputElement(label="POSITIVE", score=0.9998695850372314),
            TextClassificationOutputElement(label="NEGATIVE", score=0.0001304351753788069),
        ]

    def test_text_generation(self) -> None:
        """Tested separately in `test_inference_text_generation.py`."""

    def test_text_to_image_default(self) -> None:
        image = self.client.text_to_image("An astronaut riding a horse on the moon.")
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.height, 512)
        self.assertEqual(image.width, 512)

    def test_text_to_image_with_parameters(self) -> None:
        image = self.client.text_to_image("An astronaut riding a horse on the moon.", height=256, width=256)
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.height, 256)
        self.assertEqual(image.width, 256)

    def test_text_to_speech(self) -> None:
        audio = self.client.text_to_speech("Hello world")
        self.assertIsInstance(audio, bytes)

    def test_translation(self) -> None:
        output = self.client.translation("Hello world")
        assert output == TranslationOutput(translation_text="Hallo Welt")

    def test_translation_with_source_and_target_language(self) -> None:
        output_with_langs = self.client.translation(
            "Hello world", model="facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX", tgt_lang="fr_XX"
        )
        assert isinstance(output_with_langs, TranslationOutput)

        with self.assertRaises(ValueError):
            self.client.translation("Hello world", model="facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX")

        with self.assertRaises(ValueError):
            self.client.translation("Hello world", model="facebook/mbart-large-50-many-to-many-mmt", tgt_lang="en_XX")

    def test_token_classification(self) -> None:
        output = self.client.token_classification("My name is Sarah Jessica Parker but you can call me Jessica")
        assert output == [
            TokenClassificationOutputElement(
                label=None, score=0.9991335868835449, end=31, entity_group="PER", start=11, word="Sarah Jessica Parker"
            ),
            TokenClassificationOutputElement(
                label=None, score=0.9979913234710693, end=59, entity_group="PER", start=52, word="Jessica"
            ),
        ]

    def test_visual_question_answering(self) -> None:
        output = self.client.visual_question_answering(self.image_file, "Who's in the picture?")
        assert output == [
            VisualQuestionAnsweringOutputElement(label=None, score=0.9386941194534302, answer="woman"),
            VisualQuestionAnsweringOutputElement(label=None, score=0.34311845898628235, answer="girl"),
            VisualQuestionAnsweringOutputElement(label=None, score=0.08407749235630035, answer="lady"),
            VisualQuestionAnsweringOutputElement(label=None, score=0.0507517009973526, answer="female"),
            VisualQuestionAnsweringOutputElement(label=None, score=0.01777094043791294, answer="man"),
        ]

    def test_zero_shot_classification_single_label(self) -> None:
        output = self.client.zero_shot_classification(
            "A new model offers an explanation for how the Galilean satellites formed around the solar system's"
            "largest world. Konstantin Batygin did not set out to solve one of the solar system's most puzzling"
            " mysteries when he went for a run up a hill in Nice, France.",
            labels=["space & cosmos", "scientific discovery", "microbiology", "robots", "archeology"],
        )
        self.assertEqual(
            output,
            [
                {"label": "scientific discovery", "score": 0.7961668968200684},
                {"label": "space & cosmos", "score": 0.18570658564567566},
                {"label": "microbiology", "score": 0.00730885099619627},
                {"label": "archeology", "score": 0.006258360575884581},
                {"label": "robots", "score": 0.004559356719255447},
            ],
        )

    def test_zero_shot_classification_multi_label(self) -> None:
        output = self.client.zero_shot_classification(
            "A new model offers an explanation for how the Galilean satellites formed around the solar system's"
            "largest world. Konstantin Batygin did not set out to solve one of the solar system's most puzzling"
            " mysteries when he went for a run up a hill in Nice, France.",
            labels=["space & cosmos", "scientific discovery", "microbiology", "robots", "archeology"],
            multi_label=True,
        )
        assert output == [
            ZeroShotClassificationOutputElement(label="scientific discovery", score=0.9829297661781311),
            ZeroShotClassificationOutputElement(label="space & cosmos", score=0.755190908908844),
            ZeroShotClassificationOutputElement(label="microbiology", score=0.0005462635890580714),
            ZeroShotClassificationOutputElement(label="archeology", score=0.00047131875180639327),
            ZeroShotClassificationOutputElement(label="robots", score=0.00030448526376858354),
        ]

    def test_zero_shot_image_classification(self) -> None:
        output = self.client.zero_shot_image_classification(self.image_file, ["tree", "woman", "cat"])
        self.assertIsInstance(output, list)
        self.assertGreater(len(output), 0)
        for item in output:
            self.assertIsInstance(item.label, str)
            self.assertIsInstance(item.score, float)

    @expect_deprecation("InferenceClient.conversational")
    def test_unprocessable_entity_error(self) -> None:
        with self.assertRaisesRegex(HfHubHTTPError, "Make sure 'conversational' task is supported by the model."):
            self.client.conversational("Hi, who are you?", model="HuggingFaceH4/zephyr-7b-alpha")


class TestOpenAsBinary(InferenceClientTest):
    def test_open_as_binary_with_none(self) -> None:
        with _open_as_binary(None) as content:
            self.assertIsNone(content)

    def test_open_as_binary_from_str_path(self) -> None:
        with _open_as_binary(self.image_file) as content:
            self.assertIsInstance(content, io.BufferedReader)

    def test_open_as_binary_from_pathlib_path(self) -> None:
        with _open_as_binary(Path(self.image_file)) as content:
            self.assertIsInstance(content, io.BufferedReader)

    def test_open_as_binary_from_url(self) -> None:
        with _open_as_binary("https://huggingface.co/datasets/Narsil/image_dummy/resolve/main/tree.png") as content:
            self.assertIsInstance(content, bytes)

    def test_open_as_binary_opened_file(self) -> None:
        with Path(self.image_file).open("rb") as f:
            with _open_as_binary(f) as content:
                self.assertEqual(content, f)
                self.assertIsInstance(content, io.BufferedReader)

    def test_open_as_binary_from_bytes(self) -> None:
        content_bytes = Path(self.image_file).read_bytes()
        with _open_as_binary(content_bytes) as content:
            self.assertEqual(content, content_bytes)


class TestResolveURL(InferenceClientTest):
    FAKE_ENDPOINT = "https://my-endpoint.hf.co"

    def test_model_as_url(self):
        self.assertEqual(InferenceClient()._resolve_url(model=self.FAKE_ENDPOINT), self.FAKE_ENDPOINT)

    def test_model_as_id_no_task(self):
        self.assertEqual(
            InferenceClient()._resolve_url(model="username/repo_name"),
            "https://api-inference.huggingface.co/models/username/repo_name",
        )

    def test_model_as_id_and_task_ignored(self):
        self.assertEqual(
            InferenceClient()._resolve_url(model="username/repo_name", task="text-to-image"),
            # /models endpoint
            "https://api-inference.huggingface.co/models/username/repo_name",
        )

    def test_model_as_id_and_task_not_ignored(self):
        # Special case for feature-extraction and sentence-similarity
        self.assertEqual(
            InferenceClient()._resolve_url(model="username/repo_name", task="feature-extraction"),
            # /pipeline/{task} endpoint
            "https://api-inference.huggingface.co/pipeline/feature-extraction/username/repo_name",
        )

    def test_method_level_has_priority(self) -> None:
        # Priority to method-level
        self.assertEqual(
            InferenceClient(model=self.FAKE_ENDPOINT + "_instance")._resolve_url(model=self.FAKE_ENDPOINT + "_method"),
            self.FAKE_ENDPOINT + "_method",
        )

    def test_recommended_model_from_supported_task(self) -> None:
        # Get recommended model
        self.assertEqual(
            InferenceClient()._resolve_url(task="text-to-image"),
            "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4",
        )

    def test_unsupported_task(self) -> None:
        with self.assertRaises(ValueError):
            InferenceClient()._resolve_url(task="unknown-task")


class TestHeadersAndCookies(unittest.TestCase):
    def test_headers_and_cookies(self) -> None:
        client = InferenceClient(headers={"X-My-Header": "foo"}, cookies={"my-cookie": "bar"})
        self.assertEqual(client.headers["X-My-Header"], "foo")
        self.assertEqual(client.cookies["my-cookie"], "bar")

    def test_headers_overwrite(self) -> None:
        # Default user agent
        self.assertTrue(InferenceClient().headers["user-agent"].startswith("unknown/None;"))

        # Overwritten user-agent
        self.assertEqual(InferenceClient(headers={"user-agent": "bar"}).headers["user-agent"], "bar")

        # Case-insensitive overwrite
        self.assertEqual(InferenceClient(headers={"USER-agent": "bar"}).headers["user-agent"], "bar")

    @patch("huggingface_hub.inference._client.get_session")
    def test_mocked_post(self, get_session_mock: MagicMock) -> None:
        """Test that headers and cookies are correctly passed to the request."""
        client = InferenceClient(headers={"X-My-Header": "foo"}, cookies={"my-cookie": "bar"})
        response = client.post(data=b"content", model="username/repo_name")
        self.assertEqual(response, get_session_mock().post.return_value.content)

        expected_user_agent = build_hf_headers()["user-agent"]
        get_session_mock().post.assert_called_once_with(
            "https://api-inference.huggingface.co/models/username/repo_name",
            json=None,
            data=b"content",
            headers={"user-agent": expected_user_agent, "X-My-Header": "foo"},
            cookies={"my-cookie": "bar"},
            timeout=None,
            stream=False,
        )

    @patch("huggingface_hub.inference._client._bytes_to_image")
    @patch("huggingface_hub.inference._client.get_session")
    def test_accept_header_image(self, get_session_mock: MagicMock, bytes_to_image_mock: MagicMock) -> None:
        """Test that Accept: image/png header is set for image tasks."""
        client = InferenceClient()

        response = client.text_to_image("An astronaut riding a horse")
        self.assertEqual(response, bytes_to_image_mock.return_value)

        headers = get_session_mock().post.call_args_list[0].kwargs["headers"]
        self.assertEqual(headers["Accept"], "image/png")


class TestModelStatus(unittest.TestCase):
    def test_too_big_model(self) -> None:
        client = InferenceClient()
        model_status = client.get_model_status("facebook/nllb-moe-54b")
        self.assertFalse(model_status.loaded)
        self.assertEqual(model_status.state, "TooBig")
        self.assertEqual(model_status.compute_type, "cpu")
        self.assertEqual(model_status.framework, "transformers")

    def test_loaded_model(self) -> None:
        client = InferenceClient()
        model_status = client.get_model_status("bigscience/bloom")
        assert model_status.loaded
        assert model_status.state == "Loaded"
        assert isinstance(model_status.compute_type, dict)  # e.g. {'gpu': {'gpu': 'a100', 'count': 8}}
        assert model_status.framework == "text-generation-inference"

    def test_unknown_model(self) -> None:
        client = InferenceClient()
        with self.assertRaises(HfHubHTTPError):
            client.get_model_status("unknown/model")

    def test_model_as_url(self) -> None:
        client = InferenceClient()
        with self.assertRaises(NotImplementedError):
            client.get_model_status("https://unkown/model")


class TestListDeployedModels(unittest.TestCase):
    @patch("huggingface_hub.inference._client.get_session")
    def test_list_deployed_models_main_frameworks_mock(self, get_session_mock: MagicMock) -> None:
        InferenceClient().list_deployed_models()
        self.assertEqual(
            len(get_session_mock.return_value.get.call_args_list),
            len(MAIN_INFERENCE_API_FRAMEWORKS),
        )

    @patch("huggingface_hub.inference._client.get_session")
    def test_list_deployed_models_all_frameworks_mock(self, get_session_mock: MagicMock) -> None:
        InferenceClient().list_deployed_models("all")
        self.assertEqual(
            len(get_session_mock.return_value.get.call_args_list),
            len(ALL_INFERENCE_API_FRAMEWORKS),
        )

    def test_list_deployed_models_single_frameworks(self) -> None:
        models_by_task = InferenceClient().list_deployed_models("text-generation-inference")
        self.assertIsInstance(models_by_task, dict)
        for task, models in models_by_task.items():
            self.assertIsInstance(task, str)
            self.assertIsInstance(models, list)
            for model in models:
                self.assertIsInstance(model, str)

        self.assertIn("text-generation", models_by_task)
        self.assertIn("bigscience/bloom", models_by_task["text-generation"])
