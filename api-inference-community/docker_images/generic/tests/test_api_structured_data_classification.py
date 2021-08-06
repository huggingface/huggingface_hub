import json
import os
from unittest import TestCase, skipIf

from app.main import ALLOWED_TASKS
from starlette.testclient import TestClient
from tests.test_api import TESTABLE_MODELS
from parameterized import parameterized_class


@skipIf(
    "structured-data-classification" not in ALLOWED_TASKS,
    "structured-data-classification not implemented",
)
@parameterized_class(
    [
        {"model_id": model_id}
        for model_id in TESTABLE_MODELS["structured-data-classification"]
    ]
)
class StructuredDataClassificationTestCase(TestCase):
    def setUp(self):
        self.old_model_id = os.getenv("MODEL_ID")
        self.old_task = os.getenv("TASK")
        os.environ["MODEL_ID"] = self.model_id
        os.environ["TASK"] = "structured-data-classification"

        from app.main import app, get_pipeline

        get_pipeline.cache_clear()

        self.app = app

    @classmethod
    def setUpClass(cls):
        from app.main import get_pipeline

        get_pipeline.cache_clear()

    def tearDown(self):
        if self.old_model_id is not None:
            os.environ["MODEL_ID"] = self.old_model_id
        else:
            del os.environ["MODEL_ID"]
        if self.old_task is not None:
            os.environ["TASK"] = self.old_task
        else:
            del os.environ["TASK"]

    def test_simple(self):
        data = {
            "1": [7.4, 7.8],
            "2": [0.7, 0.88],
            "3": [7.4, 7.8],
            "4": [7.4, 7.8],
            "5": [7.4, 7.8],
            "6": [7.4, 7.8],
            "7": [7.4, 7.8],
            "8": [7.4, 7.8],
            "9": [7.4, 7.8],
            "10": [7.4, 7.8],
            "11": [7.4, 7.8],
        }

        inputs = {"data": data}
        with TestClient(self.app) as client:
            response = client.post("/", json={"inputs": inputs})
        self.assertEqual(
            response.status_code,
            200,
        )
        content = json.loads(response.content)
        self.assertEqual(type(content), list)
        self.assertEqual(len(content), 2)

    def test_malformed_input(self):
        with TestClient(self.app) as client:
            response = client.post("/", data=b"Where do I live ?")

        self.assertEqual(
            response.status_code,
            400,
        )
        content = json.loads(response.content)
        self.assertEqual(set(content.keys()), {"error"})

    def test_missing_columns(self):
        data = {"1": [7.4, 7.8], "2": [0.7, 0.88]}

        inputs = {"data": data}
        with TestClient(self.app) as client:
            response = client.post("/", json={"inputs": inputs})
        self.assertEqual(
            response.status_code,
            400,
        )
