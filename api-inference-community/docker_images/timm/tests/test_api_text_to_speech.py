import os
from unittest import TestCase, skipIf

from app.main import ALLOWED_TASKS
from app.validation import ffmpeg_read
from starlette.testclient import TestClient
from tests.test_api import TESTABLE_MODELS


@skipIf(
    "text-to-speech" not in ALLOWED_TASKS,
    "text-to-speech not implemented",
)
class AudioSourceSeparationTestCase(TestCase):
    def setUp(self):
        model_id = TESTABLE_MODELS["text-to-speech"]
        self.old_model_id = os.getenv("MODEL_ID")
        self.old_task = os.getenv("TASK")
        os.environ["MODEL_ID"] = model_id
        os.environ["TASK"] = "text-to-speech"
        from app.main import app

        self.app = app

    def tearDown(self):
        os.environ["MODEL_ID"] = self.old_model_id
        os.environ["TASK"] = self.old_task

    def test_simple(self):
        with TestClient(self.app) as client:
            response = client.post("/", json={"inputs": "This is some text"})

        self.assertEqual(
            response.status_code,
            200,
        )
        self.assertEqual(response.header["content-type"], "audio/wav")
        audio = ffmpeg_read(response.content)
        self.assertEqual(audio.shape, (10,))

    def test_malformed_input(self):
        with TestClient(self.app) as client:
            response = client.post("/", data=b"This is some test")

        self.assertEqual(
            response.status_code,
            400,
        )
        self.assertEqual(response.content, b'{"error":"Malformed soundfile"}')
