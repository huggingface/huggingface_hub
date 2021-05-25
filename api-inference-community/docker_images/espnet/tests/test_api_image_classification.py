import json
import os
from unittest import TestCase, skipIf

from app.main import ALLOWED_TASKS
from starlette.testclient import TestClient
from tests.test_api import TESTABLE_MODELS


@skipIf(
    "image-classification" not in ALLOWED_TASKS,
    "image-classification not implemented",
)
class ImageClassificationTestCase(TestCase):
    def setUp(self):
        model_id = TESTABLE_MODELS["image-classification"]
        self.old_model_id = os.getenv("MODEL_ID")
        self.old_task = os.getenv("TASK")
        os.environ["MODEL_ID"] = model_id
        os.environ["TASK"] = "image-classification"
        from app.main import app

        self.app = app

    def tearDown(self):
        os.environ["MODEL_ID"] = self.old_model_id
        os.environ["TASK"] = self.old_task

    def read(self, filename: str) -> bytes:
        dirname = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(dirname, "samples", filename)
        with open(filename, "rb") as f:
            bpayload = f.read()
        return bpayload

    def test_simple(self):
        bpayload = self.read("plane.jpg")

        with TestClient(self.app) as client:
            response = client.post("/", data=bpayload)

        self.assertEqual(
            response.status_code,
            200,
        )
        content = json.loads(response.content)
        self.assertEqual(type(content), list)
        self.assertEqual(set(type(el) for el in content), dict)
        self.assertEqual(
            set(k for el in content for k in el.keys()), {"label", "score"}
        )

    def test_different_resolution(self):
        bpayload = self.read("plane2.jpg")

        with TestClient(self.app) as client:
            response = client.post("/", data=bpayload)

        self.assertEqual(
            response.status_code,
            200,
        )
        content = json.loads(response.content)
        self.assertEqual(type(content), list)
        self.assertEqual(set(type(el) for el in content), dict)
        self.assertEqual(
            set(k for el in content for k in el.keys()), {"label", "score"}
        )
