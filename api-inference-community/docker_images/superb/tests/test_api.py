import os
import shutil
from typing import Dict
from unittest import TestCase, skipIf

from huggingface_hub import snapshot_download


# Must contain at least one example of each implemented pipeline
# Tests do not check the actual values of the model output, so small dummy
# models are recommended for faster tests.
TESTABLE_MODELS: Dict[str, str] = {
    "automatic-speech-recognition": "osanseviero/asr-with-transformers-wav2vec2",
}


ALL_TASKS = {
    "automatic-speech-recognition",
    "feature-extraction",
    "image-classification",
    "question-answering",
    "sentence-similarity",
    "text-generation",
    "text-to-speech",
    "token-classification",
}


class PipelineTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Clone the test repository and make its code available.
        This is required in order to import app without breaking.
        """
        model_id = TESTABLE_MODELS["automatic-speech-recognition"]
        filepath = snapshot_download(
            model_id, cache_dir="docker_images/superb/app/pipelines/"
        )
        os.rename(filepath, "docker_images/superb/app/pipelines/code")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree("docker_images/superb/app/pipelines/code")

    @skipIf(
        os.path.dirname(os.path.dirname(__file__)).endswith("common"),
        "common is a special case",
    )
    def test_has_at_least_one_task_enabled(self):
        from app.main import ALLOWED_TASKS

        self.assertGreater(
            len(ALLOWED_TASKS.keys()), 0, "You need to implement at least one task"
        )

    def test_unsupported_tasks(self):
        from app.main import ALLOWED_TASKS, get_pipeline

        unsupported_tasks = ALL_TASKS - ALLOWED_TASKS.keys()
        for unsupported_task in unsupported_tasks:
            with self.subTest(msg=unsupported_task, task=unsupported_task):
                os.environ["TASK"] = unsupported_task
                os.environ["MODEL_ID"] = "XX"
                with self.assertRaises(EnvironmentError):
                    get_pipeline()
