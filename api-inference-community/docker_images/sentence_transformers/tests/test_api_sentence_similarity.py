import json
import os
from unittest import TestCase, skipIf

from app.main import ALLOWED_TASKS
from parameterized import parameterized_class
from starlette.testclient import TestClient
from tests.test_api import TESTABLE_MODELS


@skipIf(
    "feature-extraction" not in ALLOWED_TASKS,
    "feature-extraction not implemented",
)
@parameterized_class(
    [{"model_id": model_id} for model_id in TESTABLE_MODELS["sentence-similarity"]]
)
class SentenceSimilarityTestCase(TestCase):
    def setUp(self):
        os.environ["MODEL_ID"] = self.model_id
        os.environ["TASK"] = "sentence-similarity"
        from app.main import app

        self.app = app

    def tearDown(self):
        del os.environ["MODEL_ID"]
        del os.environ["TASK"]

    def test_simple(self):
        source_sentence = "I am a very happy man"
        sentences = [
            "What is this?",
            "I am a super happy man",
            "I am a sad man",
            "I am a happy dog",
        ]
        inputs = {"source_sentence": source_sentence, "sentences": sentences}

        with TestClient(self.app) as client:
            response = client.post("/", json={"inputs": inputs})

        self.assertEqual(
            response.status_code,
            200,
        )

        content = json.loads(response.content)
        self.assertEqual(type(content), list)
        self.assertEqual({type(item) for item in content}, {float})

        with TestClient(self.app) as client:
            response = client.post("/", json=inputs)

        self.assertEqual(
            response.status_code,
            200,
        )
        content = json.loads(response.content)
        self.assertEqual(type(content), list)
        self.assertEqual({type(item) for item in content}, {float})

    def test_missing_input_sentences(self):
        source_sentence = "I am a very happy man"
        inputs = {"source_sentence": source_sentence}

        with TestClient(self.app) as client:
            response = client.post("/", json={"inputs": inputs})

        self.assertEqual(
            response.status_code,
            400,
        )

    def test_malformed_input(self):
        with TestClient(self.app) as client:
            response = client.post("/", data=b"\xc3\x28")

        self.assertEqual(
            response.status_code,
            400,
        )
        self.assertEqual(
            response.content,
            b'{"error":"\'utf-8\' codec can\'t decode byte 0xc3 in position 0: invalid continuation byte"}',
        )
