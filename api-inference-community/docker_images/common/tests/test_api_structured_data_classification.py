import json
import os
from unittest import TestCase, skipIf

from app.main import ALLOWED_TASKS
from starlette.testclient import TestClient
from tests.test_api import TESTABLE_MODELS


@skipIf(
    "structured-data-classification" not in ALLOWED_TASKS,
    "structured-data-classification not implemented",
)
class StructuredDataClassificationTestCase(TestCase):
    def setUp(self):
        model_id = TESTABLE_MODELS["structured-data-classification"]
        self.old_model_id = os.getenv("MODEL_ID")
        self.old_task = os.getenv("TASK")
        os.environ["MODEL_ID"] = model_id
        os.environ["TASK"] = "structured-data-classification"

        from app.main import app

        self.app = app

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
        # IMPLEMENT_THIS
        # Add one or multiple rows that the test model expects.
        data = {}

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
        # IMPLEMENT_THIS
        # Add wrong number of columns
        data = {}

        inputs = {"data": data}
        with TestClient(self.app) as client:
            response = client.post("/", json={"inputs": inputs})
        self.assertEqual(
            response.status_code,
            400,
        )
