import json
import os
from unittest import TestCase, skipIf

from app.main import ALLOWED_TASKS
from starlette.testclient import TestClient
from tests.test_api import TESTABLE_MODELS


@skipIf(
    "speech-segmentation" not in ALLOWED_TASKS,
    "speech-segmentation not implemented",
)
class SpeechSegmentationTestCase(TestCase):
    def setUp(self):
        model_id = TESTABLE_MODELS["speech-segmentation"]
        self.old_model_id = os.getenv("MODEL_ID")
        self.old_task = os.getenv("TASK")
        os.environ["MODEL_ID"] = model_id
        os.environ["TASK"] = "speech-segmentation"
        from app.main import app

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

    def read(self, filename: str) -> bytes:
        dirname = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(dirname, "samples", filename)
        with open(filename, "rb") as f:
            bpayload = f.read()
        return bpayload

    def test_original_audiofile(self):
        bpayload = self.read("sample1.flac")

        with TestClient(self.app) as client:
            response = client.post("/", data=bpayload)

        self.assertEqual(
            response.status_code,
            200,
        )
        content = json.loads(response.content)
        self.assertIsInstance(content, list)
        for c in content:
            self.assertEqual(set(c.keys()), {"class", "start", "end"})
            self.assertIsInstance(c["class"], str)
            self.assertIsInstance(c["start"], float)
            self.assertIsInstance(c["end"], float)

    def test_malformed_audio(self):
        bpayload = self.read("malformed.flac")

        with TestClient(self.app) as client:
            response = client.post("/", data=bpayload)

        self.assertEqual(
            response.status_code,
            400,
        )
        self.assertEqual(response.content, b'{"error":"Malformed soundfile"}')

    def test_dual_channel_audiofile(self):
        bpayload = self.read("sample1_dual.ogg")

        with TestClient(self.app) as client:
            response = client.post("/", data=bpayload)

        self.assertEqual(
            response.status_code,
            200,
        )
        content = json.loads(response.content)
        self.assertIsInstance(content, list)
        for c in content:
            self.assertEqual(set(c.keys()), {"class", "start", "end"})
            self.assertIsInstance(c["class"], str)
            self.assertIsInstance(c["start"], float)
            self.assertIsInstance(c["end"], float)

    def test_webm_audiofile(self):
        bpayload = self.read("sample1.webm")

        with TestClient(self.app) as client:
            response = client.post("/", data=bpayload)

        self.assertEqual(
            response.status_code,
            200,
        )
        content = json.loads(response.content)
        self.assertIsInstance(content, list)
        for c in content:
            self.assertEqual(set(c.keys()), {"class", "start", "end"})
            self.assertIsInstance(c["class"], str)
            self.assertIsInstance(c["start"], float)
            self.assertIsInstance(c["end"], float)
