import unittest

import torch.nn as nn

from huggingface_hub import ModelHubMixin


HUGGINGFACE_ID = "vasudevgupta"
DUMMY_REPO_NAME = "dummy"


class DummyModel(nn.Module, ModelHubMixin):
    def __init__(self, **kwargs):
        super().__init__()
        self.config = kwargs.pop("config", None)
        self.l1 = nn.Linear(2, 2)

    def forward(self, x):
        return self.l1(x)


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


if __name__ == "__main__":
    DummyModelTest().test_push_to_hub()
    DummyModelTest().test_save_pretrained()
    DummyModelTest().test_from_pretrained()
    # DummyModel.from_pretrained("vasudevgupta/dummy")
