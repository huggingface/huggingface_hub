import os
import shutil
from typing import Dict, List
from unittest import TestCase, skipIf

from app.main import ALLOWED_TASKS, get_pipeline


# Must contain at least one example of each implemented pipeline
# Tests do not check the actual values of the model output, so small dummy
# models are recommended for faster tests.
TESTABLE_MODELS: Dict[str, List[str]] = {
    "automatic-speech-recognition": [
        # Repo using transformers library.
        "osanseviero/asr-with-transformers-wav2vec2",
        # Repo using s3prl.
        "osanseviero/hubert_s3prl",
        # Repo using s3prl and installing a custom dependency.
        "osanseviero/hubert_s3prl_req",
    ]
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
    @skipIf(
        os.path.dirname(os.path.dirname(__file__)).endswith("common"),
        "common is a special case",
    )
    def test_has_at_least_one_task_enabled(self):
        self.assertGreater(
            len(ALLOWED_TASKS.keys()), 0, "You need to implement at least one task"
        )

    def test_unsupported_tasks(self):
        unsupported_tasks = ALL_TASKS - ALLOWED_TASKS.keys()
        for unsupported_task in unsupported_tasks:
            with self.subTest(msg=unsupported_task, task=unsupported_task):
                os.environ["TASK"] = unsupported_task
                os.environ["MODEL_ID"] = "XX"
                with self.assertRaises(EnvironmentError):
                    get_pipeline()
