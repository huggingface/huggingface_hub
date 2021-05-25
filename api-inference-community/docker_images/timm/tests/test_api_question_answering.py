import json
import os
from unittest import TestCase, skipIf

from app.main import ALLOWED_TASKS
from starlette.testclient import TestClient
from tests.test_api import TESTABLE_MODELS


@skipIf(
    "question-answering" not in ALLOWED_TASKS,
    "question-answering not implemented",
)
class QuestionAnsweringTestCase(TestCase):
    def setUp(self):
        model_id = TESTABLE_MODELS["question-answering"]
        self.old_model_id = os.getenv("MODEL_ID")
        self.old_task = os.getenv("TASK")
        os.environ["MODEL_ID"] = model_id
        os.environ["TASK"] = "question-answering"
        from app.main import app

        self.app = app

    def tearDown(self):
        os.environ["MODEL_ID"] = self.old_model_id
        os.environ["TASK"] = self.old_task

    def test_simple(self):
        inputs = {"question": "Where do I live ?", "context": "I live in New-York"}

        with TestClient(self.app) as client:
            response = client.post("/", json={"inputs": inputs})

        self.assertEqual(
            response.status_code,
            200,
        )
        content = json.loads(response.content)
        self.assertEqual(set(content.keys()), {"answer", "start", "end", "score"})

    def test_malformed_question(self):
        inputs = {"question": "Where do I live ?", "context": "I live in New-York"}

        with TestClient(self.app) as client:
            response = client.post("/", json=inputs)

        self.assertEqual(
            response.status_code,
            400,
        )
        content = json.loads(response.content)
        self.assertEqual(set(content.keys()), {"error"})

        with TestClient(self.app) as client:
            response = client.post("/", data=b"Where do I live ?")

        self.assertEqual(
            response.status_code,
            400,
        )
        content = json.loads(response.content)
        self.assertEqual(set(content.keys()), {"error"})
