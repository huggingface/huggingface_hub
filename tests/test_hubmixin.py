import unittest

from huggingface_hub.file_download import is_torch_available
from huggingface_hub.hub_mixin import ModelHubMixin


if is_torch_available():
    import torch.nn as nn


HUGGINGFACE_ID = "vasudevgupta"
DUMMY_REPO_NAME = "dummy"


def require_torch(test_case):
    """
    Decorator marking a test that requires PyTorch.

    These tests are skipped when PyTorch isn't installed.

    """
    if not is_torch_available():
        return unittest.skip("test requires PyTorch")(test_case)
    else:
        return test_case


@require_torch
class DummyModel(nn.Module, ModelHubMixin):
    def __init__(self, **kwargs):
        super().__init__()
        self.config = kwargs.pop("config", None)
        self.l1 = nn.Linear(2, 2)

    def forward(self, x):
        return self.l1(x)


@require_torch
class DummyModelTest(unittest.TestCase):
    def test_save_pretrained(self):
        model = DummyModel()
        model.save_pretrained(DUMMY_REPO_NAME)
        model.save_pretrained(
            DUMMY_REPO_NAME, config={"num": 12, "act": "gelu"}, push_to_hub=True
        )
        model.save_pretrained(
            DUMMY_REPO_NAME, config={"num": 24, "act": "relu"}, push_to_hub=True
        )
        model.save_pretrained(
            "dummy-wts", config=None, push_to_hub=True, model_id=DUMMY_REPO_NAME
        )

    def test_from_pretrained(self):
        model = DummyModel()
        model.save_pretrained(
            DUMMY_REPO_NAME, config={"num": 7, "act": "gelu_fast"}, push_to_hub=True
        )

        model = DummyModel.from_pretrained(f"{HUGGINGFACE_ID}/{DUMMY_REPO_NAME}@main")
        self.assertTrue(model.config == {"num": 7, "act": "gelu_fast"})

    def test_push_to_hub(self):
        model = DummyModel()
        model.save_pretrained("dummy-wts", push_to_hub=False)
        model.push_to_hub("dummy-wts", model_id=DUMMY_REPO_NAME)
