import os
import shutil
import time
import unittest

from huggingface_hub import HfApi
from huggingface_hub.file_download import is_torch_available
from huggingface_hub.hub_mixin import ModelHubMixin, SklearnPipelineHubMixin

from .testing_constants import ENDPOINT_STAGING, PASS, USER
from .testing_utils import set_write_permission_and_retry

REPO_NAME = "mixin-repo-{}".format(int(time.time() * 10e3))

WORKING_REPO_SUBDIR = "fixtures/working_repo_2"
WORKING_REPO_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), WORKING_REPO_SUBDIR
)

if is_torch_available():
    import torch.nn as nn


def require_torch(test_case):
    """
    Decorator marking a test that requires PyTorch.

    These tests are skipped when PyTorch isn't installed.

    """
    if not is_torch_available():
        return unittest.skip("test requires PyTorch")(test_case)
    else:
        return test_case


if is_torch_available():

    class DummyModel(nn.Module, ModelHubMixin):
        def __init__(self, **kwargs):
            super().__init__()
            self.config = kwargs.pop("config", None)
            self.l1 = nn.Linear(2, 2)

        def forward(self, x):
            return self.l1(x)


else:
    DummyModel = None


@require_torch
class HubMixinCommonTest(unittest.TestCase):
    _api = HfApi(endpoint=ENDPOINT_STAGING)


@require_torch
class HubMixinTest(HubMixinCommonTest):
    def tearDown(self) -> None:
        try:
            shutil.rmtree(WORKING_REPO_DIR, onerror=set_write_permission_and_retry)
        except FileNotFoundError:
            pass

    @classmethod
    def setUpClass(cls):
        """
        Share this valid token in all tests below.
        """
        cls._token = cls._api.login(username=USER, password=PASS)

    def test_save_pretrained(self):
        model = DummyModel()

        model.save_pretrained(f"{WORKING_REPO_DIR}/{REPO_NAME}")
        files = os.listdir(f"{WORKING_REPO_DIR}/{REPO_NAME}")
        self.assertTrue("pytorch_model.bin" in files)
        self.assertEqual(len(files), 1)

        model.save_pretrained(
            f"{WORKING_REPO_DIR}/{REPO_NAME}", config={"num": 12, "act": "gelu"}
        )
        files = os.listdir(f"{WORKING_REPO_DIR}/{REPO_NAME}")
        self.assertTrue("config.json" in files)
        self.assertTrue("pytorch_model.bin" in files)
        self.assertEqual(len(files), 2)

    def test_rel_path_from_pretrained(self):
        model = DummyModel()
        model.save_pretrained(
            f"tests/{WORKING_REPO_SUBDIR}/FROM_PRETRAINED",
            config={"num": 10, "act": "gelu_fast"},
        )

        model = DummyModel.from_pretrained(
            f"tests/{WORKING_REPO_SUBDIR}/FROM_PRETRAINED"
        )
        self.assertTrue(model.config == {"num": 10, "act": "gelu_fast"})

    def test_abs_path_from_pretrained(self):
        model = DummyModel()
        model.save_pretrained(
            f"{WORKING_REPO_DIR}/{REPO_NAME}-FROM_PRETRAINED",
            config={"num": 10, "act": "gelu_fast"},
        )

        model = DummyModel.from_pretrained(
            f"{WORKING_REPO_DIR}/{REPO_NAME}-FROM_PRETRAINED"
        )
        self.assertDictEqual(model.config, {"num": 10, "act": "gelu_fast"})

    def test_push_to_hub(self):
        model = DummyModel()
        model.save_pretrained(
            f"{WORKING_REPO_DIR}/{REPO_NAME}-PUSH_TO_HUB",
            config={"num": 7, "act": "gelu_fast"},
        )

        model.push_to_hub(
            f"{WORKING_REPO_DIR}/{REPO_NAME}-PUSH_TO_HUB",
            f"{REPO_NAME}-PUSH_TO_HUB",
            api_endpoint=ENDPOINT_STAGING,
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
        )

        model_info = self._api.model_info(
            f"{USER}/{REPO_NAME}-PUSH_TO_HUB",
        )
        self.assertEqual(model_info.modelId, f"{USER}/{REPO_NAME}-PUSH_TO_HUB")

        self._api.delete_repo(token=self._token, name=f"{REPO_NAME}-PUSH_TO_HUB")


def is_sklearn_available():
    try:
        import sklearn
        import cloudpickle
        return True
    except importlib_metadata.PackageNotFoundError:
        return False
    
if is_sklearn_available():
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    import cloudpickle

def require_sklearn(test_case):
    """
    Decorator marking a test that requires Sklearn and Cloudpickle.

    These tests are skipped when Sklearn and Cloudpickle aren't installed.

    """
    if not is_sklearn_available():
        return unittest.skip("test requires Sklearn and Cloudpickle")(test_case)
    else:
        return test_case

if is_sklearn_available():
    class SkLearnDummyModel(Pipeline, SklearnPipelineHubMixin):
        def __init__(self, steps):
            super().__init__(steps)
else:
    SkLearnDummyModel = None


@require_sklearn
class SklearnHubMixinCommonTest(unittest.TestCase):
    _api = HfApi(endpoint=ENDPOINT_STAGING)


@require_sklearn
class SklearnHubMixinTest(SklearnHubMixinCommonTest):
    def tearDown(self) -> None:
        try:
            shutil.rmtree(WORKING_REPO_DIR, onerror=set_write_permission_and_retry)
        except FileNotFoundError:
            pass

    @classmethod
    def setUpClass(cls):
        """
        Share this valid token in all tests below.
        """
        cls._token = cls._api.login(username=USER, password=PASS)
    
    def test_save_pretrained(self):
        model = SkLearnDummyModel([
            ('svc', SVC()),
        ])

        model.save_pretrained(f"{WORKING_REPO_DIR}/{REPO_NAME}")
        files = os.listdir(f"{WORKING_REPO_DIR}/{REPO_NAME}")
        self.assertTrue("sklearn_model.pickle" in files)
        self.assertEqual(len(files), 1)

    def test_save_pretrained_with_name(self):
        model = SkLearnDummyModel([
            ('svc', SVC()),
        ])

        model.save_pretrained(f"{WORKING_REPO_DIR}/{REPO_NAME}", model_filename="model.pickle")
        files = os.listdir(f"{WORKING_REPO_DIR}/{REPO_NAME}")
        self.assertTrue("model.pickle" in files)
        self.assertEqual(len(files), 1)

    def test_from_pretrained_rel_path(self):
        model = SkLearnDummyModel([
            ('svc', SVC()),
        ])
        model.save_pretrained(
            f"tests/{WORKING_REPO_SUBDIR}/FROM_PRETRAINED",
        )

        model = SklearnPipelineHubMixin.from_pretrained(
            f"tests/{WORKING_REPO_SUBDIR}/FROM_PRETRAINED"
        )
        self.assertTrue(len(model.steps) == 1)
        self.assertTrue(model.steps[0][0] == "svc")

    def test_from_pretrained_abs_path(self):
        model = SkLearnDummyModel([
            ('svc', SVC()),
        ])
        model.save_pretrained(
            f"{WORKING_REPO_DIR}/{REPO_NAME}-FROM_PRETRAINED"
        )

        model = SklearnPipelineHubMixin.from_pretrained(
            f"{WORKING_REPO_DIR}/{REPO_NAME}-FROM_PRETRAINED"
        )
        self.assertTrue(len(model.steps) == 1)
        self.assertTrue(model.steps[0][0] == "svc")
    
    def test_from_pretrained_with_custom_name(self):
        model = SkLearnDummyModel([
            ('svc', SVC()),
        ])
        model.save_pretrained(
            f"tests/{WORKING_REPO_SUBDIR}/FROM_PRETRAINED", "model.pickle"
        )

        model = SklearnPipelineHubMixin.from_pretrained(
            f"tests/{WORKING_REPO_SUBDIR}/FROM_PRETRAINED", "model.pickle"
        )
        self.assertTrue(len(model.steps) == 1)
        self.assertTrue(model.steps[0][0] == "svc")

    def test_push_to_hub(self):
        model = SkLearnDummyModel([
            ('svc', SVC()),
        ])
        model.save_pretrained(
            f"{WORKING_REPO_DIR}/{REPO_NAME}-PUSH_TO_HUB"
        )

        model.push_to_hub(
            f"{WORKING_REPO_DIR}/{REPO_NAME}-PUSH_TO_HUB",
            f"{REPO_NAME}-PUSH_TO_HUB",
            api_endpoint=ENDPOINT_STAGING,
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
        )

        model_info = self._api.model_info(
            f"{USER}/{REPO_NAME}-PUSH_TO_HUB",
        )
        self.assertEqual(model_info.modelId, f"{USER}/{REPO_NAME}-PUSH_TO_HUB")

        new_model = SklearnPipelineHubMixin.from_pretrained(
            f"{REPO_NAME}-PUSH_TO_HUB"
        )
        self.assertTrue(len(model.steps) == 1)
        self.assertTrue(model.steps[0][0] == "svc")

        self._api.delete_repo(token=self._token, name=f"{REPO_NAME}-PUSH_TO_HUB")
