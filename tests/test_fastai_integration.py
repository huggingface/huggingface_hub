import os
from unittest import TestCase, skip

from huggingface_hub import HfApi
from huggingface_hub.fastai_utils import (
    _save_pretrained_fastai,
    from_pretrained_fastai,
    push_to_hub_fastai,
)
from huggingface_hub.utils import (
    SoftTemporaryDirectory,
    is_fastai_available,
    is_fastcore_available,
    is_torch_available,
)

from .testing_constants import ENDPOINT_STAGING, TOKEN, USER
from .testing_utils import repo_name


WORKING_REPO_SUBDIR = f"fixtures/working_repo_{__name__.split('.')[-1]}"
WORKING_REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), WORKING_REPO_SUBDIR)

if is_fastai_available():
    from fastai.data.block import DataBlock
    from fastai.test_utils import synth_learner

if is_torch_available():
    import torch


def require_fastai_fastcore(test_case):
    """
    Decorator marking a test that requires fastai and fastcore.
    These tests are skipped when fastai and fastcore are not installed.
    """
    if not is_fastai_available():
        return skip("Test requires fastai")(test_case)
    elif not is_fastcore_available():
        return skip("Test requires fastcore")(test_case)
    else:
        return test_case


def fake_dataloaders(a=2, b=3, bs=16, n=10):
    def get_data(n):
        x = torch.randn(bs * n, 1)
        return torch.cat((x, a * x + b + 0.1 * torch.randn(bs * n, 1)), 1)

    ds = get_data(n)
    dblock = DataBlock()
    return dblock.dataloaders(ds)


if is_fastai_available():
    dummy_model = synth_learner(data=fake_dataloaders())
    dummy_config = dict(test="test_0")
else:
    dummy_model = None
    dummy_config = None


@require_fastai_fastcore
class TestFastaiUtils(TestCase):
    def test_save_pretrained_without_config(self):
        with SoftTemporaryDirectory() as tmpdir:
            _save_pretrained_fastai(dummy_model, tmpdir)
            files = os.listdir(tmpdir)

        self.assertTrue("model.pkl" in files)
        self.assertTrue("pyproject.toml" in files)
        self.assertTrue("README.md" in files)
        self.assertEqual(len(files), 3)

    def test_save_pretrained_with_config(self):
        with SoftTemporaryDirectory() as tmpdir:
            _save_pretrained_fastai(dummy_model, tmpdir, config=dummy_config)
            files = os.listdir(tmpdir)

        self.assertTrue("config.json" in files)
        self.assertEqual(len(files), 4)

    def test_push_to_hub_and_from_pretrained_fastai(self):
        api = HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)
        repo_id = f"{USER}/{repo_name()}"

        push_to_hub_fastai(learner=dummy_model, repo_id=repo_id, token=TOKEN, config=dummy_config)
        model_info = api.model_info(repo_id)
        self.assertEqual(model_info.modelId, repo_id)
        loaded_model = from_pretrained_fastai(repo_id)
        self.assertEqual(dummy_model.show_training_loop(), loaded_model.show_training_loop())

        api.delete_repo(repo_id=repo_id)
