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
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from huggingface_hub import InferenceClient, hf_hub_download
from huggingface_hub.inference._client import _open_as_binary
from huggingface_hub.utils import HfHubHTTPError, build_hf_headers

from .testing_utils import with_production_testing


# Avoid call to hf.co/api/models in VCRed tests
_RECOMMENDED_MODELS_FOR_VCR = {
    "audio-classification": "speechbrain/google_speech_command_xvector",
    "audio-to-audio": "speechbrain/sepformer-wham",
    "automatic-speech-recognition": "facebook/wav2vec2-base-960h",
    "conversational": "facebook/blenderbot-400M-distill",
    "feature-extraction": "facebook/bart-base",
    "image-classification": "google/vit-base-patch16-224",
    "image-segmentation": "facebook/detr-resnet-50-panoptic",
    "object-detection": "facebook/detr-resnet-50",
    "sentence-similarity": "sentence-transformers/all-MiniLM-L6-v2",
    "summarization": "sshleifer/distilbart-cnn-12-6",
    "text-classification": "distilbert-base-uncased-finetuned-sst-2-english",
    "text-to-image": "CompVis/stable-diffusion-v1-4",
    "text-to-speech": "espnet/kan-bayashi_ljspeech_vits",
    "zero-shot-image-classification": "openai/clip-vit-base-patch32",
}


class InferenceClientTest(unittest.TestCase):
    @classmethod
    @with_production_testing
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.image_file = hf_hub_download(repo_id="Narsil/image_dummy", repo_type="dataset", filename="lena.png")
        cls.audio_file = hf_hub_download(repo_id="Narsil/image_dummy", repo_type="dataset", filename="sample1.flac")


@pytest.mark.vcr
@with_production_testing
@patch("huggingface_hub.inference._common._fetch_recommended_models", lambda: _RECOMMENDED_MODELS_FOR_VCR)
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
            self.assertIsInstance(item["score"], float)
            self.assertIsInstance(item["label"], str)

    def test_automatic_speech_recognition(self) -> None:
        output = self.client.automatic_speech_recognition(self.audio_file)
        self.assertEqual(output, "A MAN SAID TO THE UNIVERSE SIR I EXIST")

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

    def test_feature_extraction(self) -> None:
        embedding = self.client.feature_extraction("Hi, who are you?")
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape, (8, 768))

    def test_image_classification(self) -> None:
        output = self.client.image_classification(self.image_file)
        self.assertIsInstance(output, list)
        self.assertGreater(len(output), 0)
        for item in output:
            self.assertIsInstance(item["score"], float)
            self.assertIsInstance(item["label"], str)

    def test_image_segmentation(self) -> None:
        output = self.client.image_segmentation(self.image_file)
        self.assertIsInstance(output, list)
        self.assertGreater(len(output), 0)
        for item in output:
            self.assertIsInstance(item["score"], float)
            self.assertIsInstance(item["label"], str)
            self.assertIsInstance(item["mask"], Image.Image)
            self.assertEqual(item["mask"].height, 512)
            self.assertEqual(item["mask"].width, 512)

    # ERROR 500 from server
    # Only during tests, not when running locally. Has to be investigated.
    # def test_image_to_image(self) -> None:
    #     image = self.client.image_to_image(self.image_file, prompt="turn the woman into a man")
    #     self.assertIsInstance(image, Image.Image)
    #     self.assertEqual(image.height, 512)
    #     self.assertEqual(image.width, 512)

    # ERROR 500 from server
    # Only during tests, not when running locally. Has to be investigated.
    # def test_image_to_text(self) -> None:
    #     caption = self.client.image_to_text(self.image_file)
    #     self.assertEqual(caption, "")

    def test_object_detection(self) -> None:
        output = self.client.object_detection(self.image_file)
        self.assertIsInstance(output, list)
        self.assertGreater(len(output), 0)
        for item in output:
            self.assertIsInstance(item["score"], float)
            self.assertIsInstance(item["label"], str)
            self.assertIsInstance(item["box"], dict)
            self.assertIn("xmin", item["box"])
            self.assertIn("ymin", item["box"])
            self.assertIn("xmax", item["box"])
            self.assertIn("ymax", item["box"])

    def test_question_answering(self) -> None:
        model = "deepset/roberta-base-squad2"
        output = self.client.question_answering(question="What is the meaning of life?", context="42", model=model)
        self.assertIsInstance(output, dict)
        self.assertGreater(len(output), 0)
        self.assertIsInstance(output["score"], float)
        self.assertIsInstance(output["start"], int)
        self.assertIsInstance(output["end"], int)
        self.assertIsInstance(output["answer"], str)

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
        self.assertEqual(
            summary,
            "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building. Its base is"
            " square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower"
            " surpassed the Washington Monument to become the tallest man-made structure in the world.",
        )

    def test_text_classification(self) -> None:
        output = self.client.text_classification("I like you")
        self.assertIsInstance(output, list)
        self.assertEqual(len(output), 2)
        for item in output:
            self.assertIsInstance(item["score"], float)
            self.assertIsInstance(item["label"], str)

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
        output = self.client.translation("Hello world", model="t5-small")
        self.assertEqual(output, "Hallo Welt")

    def test_token_classification(self) -> None:
        model = "dbmdz/bert-large-cased-finetuned-conll03-english"
        output = self.client.token_classification(
            "My name is Sarah Jessica Parker but you can call me Jessica", model=model
        )
        self.assertIsInstance(output, list)
        self.assertGreater(len(output), 0)
        for item in output:
            self.assertIsInstance(item["entity_group"], str)
            self.assertIsInstance(item["score"], float)
            self.assertIsInstance(item["word"], str)
            self.assertIsInstance(item["start"], int)
            self.assertIsInstance(item["end"], int)

    def test_zero_shot_image_classification(self) -> None:
        output = self.client.zero_shot_image_classification(self.image_file, ["tree", "woman", "cat"])
        self.assertIsInstance(output, list)
        self.assertGreater(len(output), 0)
        for item in output:
            self.assertIsInstance(item["label"], str)
            self.assertIsInstance(item["score"], float)


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
        model_status = client.get_model_status("bigcode/starcoder")
        self.assertTrue(model_status.loaded)
        self.assertEqual(model_status.state, "Loaded")
        self.assertEqual(model_status.compute_type, "gpu")
        self.assertEqual(model_status.framework, "text-generation-inference")

    def test_unknown_model(self) -> None:
        client = InferenceClient()
        with self.assertRaises(HfHubHTTPError):
            client.get_model_status("unknown/model")

    def test_model_as_url(self) -> None:
        client = InferenceClient()
        with self.assertRaises(NotImplementedError):
            client.get_model_status("https://unkown/model")
