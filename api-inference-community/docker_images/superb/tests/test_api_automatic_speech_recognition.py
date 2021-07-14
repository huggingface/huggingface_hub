import json
import os
import shutil
from unittest import TestCase

from huggingface_hub import snapshot_download
from starlette.testclient import TestClient
from tests.test_api import TESTABLE_MODELS


class AutomaticSpeecRecognitionTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Clone the test repository and make its code available.
        This replicates a git clone + moving files to running directory.
        """
        model_id = TESTABLE_MODELS["automatic-speech-recognition"]
        filepath = snapshot_download(model_id, cache_dir=os.getcwd())
        file_names = os.listdir(filepath)
        for file_name in file_names:
            shutil.move(os.path.join(filepath, file_name), os.getcwd())
        shutil.rmtree(filepath)
        cls.file_names = file_names

    def setUp(self):
        """
        This logic is done in the docker prestart step, this is just
        to replicate the logic.
        """
        model_id = TESTABLE_MODELS["automatic-speech-recognition"]
        self.old_model_id = os.getenv("MODEL_ID")
        self.old_task = os.getenv("TASK")
        os.environ["MODEL_ID"] = model_id
        os.environ["TASK"] = "automatic-speech-recognition"

        from app.main import app

        self.app = app

    @classmethod
    def tearDownClass(cls):
        for file_name in cls.file_names:
            if os.path.isfile(file_name):
                os.remove(file_name)
            elif os.path.isdir(file_name):
                shutil.rmtree(file_name)

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
        self.assertEqual(set(content.keys()), {"text"})

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
        self.assertEqual(set(content.keys()), {"text"})

    def test_webm_audiofile(self):
        bpayload = self.read("sample1.webm")

        with TestClient(self.app) as client:
            response = client.post("/", data=bpayload)

        self.assertEqual(
            response.status_code,
            200,
        )
        content = json.loads(response.content)
        self.assertEqual(set(content.keys()), {"text"})
