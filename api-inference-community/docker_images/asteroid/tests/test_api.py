import os
from typing import Dict
from unittest import TestCase, skipIf

from app.main import ALLOWED_TASKS, get_pipeline


# Must contain at least one example of each implemented pipeline
# Tests do not check the actual values of the model output, so small dummy
# models are recommended for faster tests.
TESTABLE_MODELS: Dict[str, str] = {
    #  IMPLEMENT_THIS
    # "automatic-speech-recognition": "mysample-ASR",
    # "text-generation": "mysample-gpt2",
    "audio-source-separation": "julien-c/DPRNNTasNet-ks16_WHAM_sepclean",
    "audio-to-audio": "julien-c/DPRNNTasNet-ks16_WHAM_sepclean"
}


ALL_TASKS = {
    "automatic-speech-recognition",
    "audio-source-separation",
    "feature-extraction",
    "image-classification",
    "question-answering",
    "text-generation",
    "text-to-speech",
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
                with self.assertRaises(EnvironmentError):
                    get_pipeline(unsupported_task, model_id="XX")
