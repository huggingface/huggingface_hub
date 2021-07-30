import os
from unittest import TestCase
import requests

from api_inference_community.batch import batch


class DummyPipeline:
    def __call__(self, *args, **kwargs):
        return {"text": "Something"}


class BatchTestCase(TestCase):
    def test_batch_simple(self):
        pipeline = DummyPipeline()

        token = os.getenv("API_TOKEN")
        batch(
            dataset_name="Narsil/asr_dummy",
            dataset_config="asr",
            dataset_split="test",
            dataset_column="file",
            token=token,
            repo_id="Narsil/bulk-dummy",
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
