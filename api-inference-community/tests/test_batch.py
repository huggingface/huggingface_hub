import os
from unittest import TestCase
from unittest.mock import patch

import requests
from api_inference_community.batch import batch


class DummyPipeline:
    sampling_rate = 16000

    def __call__(self, *args, **kwargs):
        return {"text": "Something"}


class BatchTestCase(TestCase):
    @patch("api_inference_community.batch.normalize_payload")
    def test_batch_simple(self, normalize_payload):
        # We don't need to follow the real normalization.
        normalize_payload.return_value = None, {}
        pipeline = DummyPipeline()

        token = os.getenv("API_TOKEN")
        batch(
            dataset_name="Narsil/automatic_speech_recognition_dummy",
            dataset_config="asr",
            dataset_split="test",
            dataset_column="file",
            token=token,
            repo_id="Narsil/bulk-dummy",
            use_gpu=False,
            task="automatic-speech-recognition",
            pipeline=pipeline,
        )

        response = requests.get(
            "https://huggingface.co/datasets/Narsil/bulk-dummy/raw/main/data_asr_test_file.txt",
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.content,
            b'{"text": "Something"}\n{"text": "Something"}\n{"text": "Something"}\n',
        )
