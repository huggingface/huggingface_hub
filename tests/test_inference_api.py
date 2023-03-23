# Copyright 2020 The HuggingFace Team. All rights reserved.
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
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from huggingface_hub import hf_hub_download
from huggingface_hub.inference_api import InferenceApi

from .testing_utils import with_production_testing


@with_production_testing
class InferenceApiTest(unittest.TestCase):
    def read(self, filename: str) -> bytes:
        return Path(filename).read_bytes()

    @classmethod
    @with_production_testing
    def setUpClass(cls) -> None:
        cls.image_file = hf_hub_download(repo_id="Narsil/image_dummy", repo_type="dataset", filename="lena.png")
        return super().setUpClass()

    def test_simple_inference(self):
        api = InferenceApi("bert-base-uncased")
        inputs = "Hi, I think [MASK] is cool"
        results = api(inputs)
        self.assertIsInstance(results, list)

        result = results[0]
        self.assertIsInstance(result, dict)
        self.assertTrue("sequence" in result)
        self.assertTrue("score" in result)

    def test_inference_with_params(self):
        api = InferenceApi("typeform/distilbert-base-uncased-mnli")
        inputs = "I bought a device but it is not working and I would like to get reimbursed!"
        params = {"candidate_labels": ["refund", "legal", "faq"]}
        result = api(inputs, params)
        self.assertIsInstance(result, dict)
        self.assertTrue("sequence" in result)
        self.assertTrue("scores" in result)

    def test_inference_with_dict_inputs(self):
        api = InferenceApi("distilbert-base-cased-distilled-squad")
        inputs = {
            "question": "What's my name?",
            "context": "My name is Clara and I live in Berkeley.",
        }
        result = api(inputs)
        self.assertIsInstance(result, dict)
        self.assertTrue("score" in result)
        self.assertTrue("answer" in result)

    def test_inference_with_audio(self):
        api = InferenceApi("facebook/wav2vec2-base-960h")
        file = hf_hub_download(
            repo_id="hf-internal-testing/dummy-flac-single-example",
            repo_type="dataset",
            filename="example.flac",
        )
        data = self.read(file)
        result = api(data=data)
        self.assertIsInstance(result, dict)
        self.assertTrue("text" in result, f"We received {result} instead")

    def test_inference_with_image(self):
        api = InferenceApi("google/vit-base-patch16-224")
        data = self.read(self.image_file)
        result = api(data=data)
        self.assertIsInstance(result, list)
        for classification in result:
            self.assertIsInstance(classification, dict)
            self.assertTrue("score" in classification)
            self.assertTrue("label" in classification)

    def test_text_to_image(self):
        api = InferenceApi("stabilityai/stable-diffusion-2-1")
        with patch("huggingface_hub.inference_api.get_session") as mock:
            mock().post.return_value.headers = {"Content-Type": "image/jpeg"}
            mock().post.return_value.content = self.read(self.image_file)
            output = api("cat")
        self.assertIsInstance(output, Image.Image)

    def test_text_to_image_raw_response(self):
        api = InferenceApi("stabilityai/stable-diffusion-2-1")
        with patch("huggingface_hub.inference_api.get_session") as mock:
            mock().post.return_value.headers = {"Content-Type": "image/jpeg"}
            mock().post.return_value.content = self.read(self.image_file)
            output = api("cat", raw_response=True)
        # Raw response is returned
        self.assertEqual(output, mock().post.return_value)

    def test_inference_overriding_task(self):
        api = InferenceApi(
            "sentence-transformers/paraphrase-albert-small-v2",
            task="feature-extraction",
        )
        inputs = "This is an example again"
        result = api(inputs)
        self.assertIsInstance(result, list)

    def test_inference_overriding_invalid_task(self):
        with self.assertRaises(ValueError, msg="Invalid task invalid-task. Make sure it's valid."):
            InferenceApi("bert-base-uncased", task="invalid-task")

    def test_inference_missing_input(self):
        api = InferenceApi("deepset/roberta-base-squad2")
        result = api({"question": "What's my name?"})
        self.assertIsInstance(result, dict)
        self.assertTrue("error" in result)
