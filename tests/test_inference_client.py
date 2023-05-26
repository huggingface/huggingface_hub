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
import unittest
import numpy as np
import pytest
from huggingface_hub import hf_hub_download, InferenceClient
from PIL import Image
from .testing_utils import with_production_testing


class InferenceClientTest(unittest.TestCase):
    @classmethod
    @with_production_testing
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.image_file = hf_hub_download(repo_id="Narsil/image_dummy", repo_type="dataset", filename="lena.png")
        cls.audio_file = hf_hub_download(repo_id="Narsil/image_dummy", repo_type="dataset", filename="sample1.flac")


@pytest.mark.vcr
@with_production_testing
class InferenceClientVCRTest(InferenceClientTest):
    """
    Test for the main tasks implemented in InferenceClient. Since Inference API can be flaky, we use VCR.py and
    pytest-vcr to record and replay the inference calls (see https://pytest-vcr.readthedocs.io/en/latest/).

    Tips when adding new tasks:
    - Most of the time, we only test that the return values are correct. We don't always test the actual output of the model.
    - In the CI, VRC replay is always on. If you want to test locally against the server, you can use the `--vcr-mode`
      and `--disable-vcr` command line options. See https://pytest-vcr.readthedocs.io/en/latest/configuration/.
    - If you get rate-limited locally, you can use your own token when initializing InferenceClient.
      /!\ WARNING: if you do so, you must delete the token from the cassette before committing!
    - If the model is not loaded on the server, you will saves a lot of HTTP 503 response in the cassette. We don't
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
            (
                "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building. Its base is"
                " square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower"
                " surpassed the Washington Monument to become the tallest man-made structure in the world."
            ),
        )

    def test_text_to_image(self) -> None:
        image = self.client.text_to_image("An astronaut riding a horse on the moon.")
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.height, 768)
        self.assertEqual(image.width, 768)

    def test_text_to_speech(self) -> None:
        audio = self.client.text_to_speech("Hello world")
        self.assertIsInstance(audio, bytes)


class InferenceClientOfflineTest(InferenceClientTest):
    pass
