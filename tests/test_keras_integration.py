import os
import shutil
import time
import unittest

import pytest

from huggingface_hub import HfApi
from huggingface_hub.file_download import is_tf_available
from huggingface_hub.keras_mixin import (
    KerasModelHubMixin,
    from_pretrained_keras,
    push_to_hub_keras,
    save_pretrained_keras,
)

from .testing_constants import ENDPOINT_STAGING, PASS, USER
from .testing_utils import set_write_permission_and_retry


REPO_NAME = "mixin-repo-{}".format(int(time.time() * 10e3))

WORKING_REPO_SUBDIR = "fixtures/working_repo_3"
WORKING_REPO_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), WORKING_REPO_SUBDIR
)

if is_tf_available():
    import tensorflow as tf


def require_tf(test_case):
    """
    Decorator marking a test that requires TensorFlow.

    These tests are skipped when TensorFlow isn't installed.

    """
    if not is_tf_available():
        return unittest.skip("test requires Tensorflow")(test_case)
    else:
        return test_case


if is_tf_available():

    # Define dummy mixin model...
    class DummyModel(tf.keras.Model, KerasModelHubMixin):
        def __init__(self, **kwargs):
            super().__init__()
            self.config = kwargs.pop("config", None)
            self.l1 = tf.keras.layers.Dense(2, activation="relu")
            dummy_batch_size = input_dim = 2
            self.dummy_inputs = tf.ones([dummy_batch_size, input_dim])

        def call(self, x):
            return self.l1(x)

    # Define dummy sequential model...
    dummy_model_sequential = tf.keras.models.Sequential()
    dummy_model_sequential.add(tf.keras.layers.Dense(2, activation="relu"))
    dummy_model_sequential.compile(optimizer="adam", loss="mse")

    # Define dummy functional model...
    inputs = tf.keras.layers.Input(shape=(2,))
    x = tf.keras.layers.Dense(2, activation="relu")(inputs)
    dummy_model_functional = tf.keras.models.Model(inputs=inputs, outputs=x)
    dummy_model_functional.compile(optimizer="adam", loss="mse")

else:
    DummyModel = None


@require_tf
class HubMixingCommonTest(unittest.TestCase):
    _api = HfApi(endpoint=ENDPOINT_STAGING)


@require_tf
class HubMixingTest(HubMixingCommonTest):
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
        model(model.dummy_inputs)
        model.save_pretrained(f"{WORKING_REPO_DIR}/{REPO_NAME}")
        files = os.listdir(f"{WORKING_REPO_DIR}/{REPO_NAME}")
        self.assertTrue("saved_model.pb" in files)
        self.assertTrue("keras_metadata.pb" in files)
        self.assertEqual(len(files), 4)

        model.save_pretrained(
            f"{WORKING_REPO_DIR}/{REPO_NAME}", config={"num": 12, "act": "gelu"}
        )
        files = os.listdir(f"{WORKING_REPO_DIR}/{REPO_NAME}")
        self.assertTrue("config.json" in files)
        self.assertTrue("saved_model.pb" in files)
        self.assertEqual(len(files), 5)

    def test_keras_from_pretrained_weights(self):
        model = DummyModel()
        model(model.dummy_inputs)

        model.save_pretrained(f"{WORKING_REPO_DIR}/{REPO_NAME}")
        new_model = DummyModel.from_pretrained(f"{WORKING_REPO_DIR}/{REPO_NAME}")

        # Check the reloaded model's weights match the original model's weights
        self.assertTrue(tf.reduce_all(tf.equal(new_model.weights[0], model.weights[0])))

        # Check a new model's weights are not the same as the reloaded model's weights
        another_model = DummyModel()
        another_model(tf.ones([2, 2]))
        self.assertFalse(
            tf.reduce_all(tf.equal(new_model.weights[0], another_model.weights[0]))
            .numpy()
            .item()
        )

    def test_rel_path_from_pretrained(self):
        model = DummyModel()
        model(model.dummy_inputs)
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
        model(model.dummy_inputs)
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
        model.push_to_hub(
            repo_path_or_name=f"{WORKING_REPO_DIR}/{REPO_NAME}-PUSH_TO_HUB",
            api_endpoint=ENDPOINT_STAGING,
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
            config={"num": 7, "act": "gelu_fast"},
        )

        model_info = self._api.model_info(
            f"{USER}/{REPO_NAME}-PUSH_TO_HUB",
        )
        self.assertEqual(model_info.modelId, f"{USER}/{REPO_NAME}-PUSH_TO_HUB")

        self._api.delete_repo(token=self._token, name=f"{REPO_NAME}-PUSH_TO_HUB")


@pytest.fixture()
def hf_token():
    _api = HfApi(endpoint=ENDPOINT_STAGING)
    token = _api.login(username=USER, password=PASS)
    yield token
    try:
        shutil.rmtree(WORKING_REPO_DIR, onerror=set_write_permission_and_retry)
    except FileNotFoundError:
        pass


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(dummy_model_sequential, id="sequential"),
        pytest.param(dummy_model_functional, id="functional"),
    ],
)
def test_save_pretrained(model, hf_token):
    # model = dummy_model_sequential if model_type == 'sequential' else dummy_model_functional
    model.build((None, 2))
    save_pretrained_keras(model, f"{WORKING_REPO_DIR}/{REPO_NAME}")
    files = os.listdir(f"{WORKING_REPO_DIR}/{REPO_NAME}")

    assert "saved_model.pb" in files
    assert "keras_metadata.pb" in files
    assert len(files) == 4


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(dummy_model_sequential, id="sequential"),
        pytest.param(dummy_model_functional, id="functional"),
    ],
)
def test_keras_from_pretrained_weights(model, hf_token):
    model.build((None, 2))

    save_pretrained_keras(model, f"{WORKING_REPO_DIR}/{REPO_NAME}")
    new_model = from_pretrained_keras(f"{WORKING_REPO_DIR}/{REPO_NAME}")

    # Check the reloaded model's weights match the original model's weights
    assert tf.reduce_all(tf.equal(new_model.weights[0], model.weights[0]))

    # Check a new model's weights are not the same as the reloaded model's weights
    another_model = DummyModel()
    another_model(tf.ones([2, 2]))
    assert not (
        tf.reduce_all(tf.equal(new_model.weights[0], another_model.weights[0]))
        .numpy()
        .item()
    )


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(dummy_model_sequential, id="sequential"),
        pytest.param(dummy_model_functional, id="functional"),
    ],
)
def test_rel_path_from_pretrained(model, hf_token):
    model.build((None, 2))
    save_pretrained_keras(
        model,
        f"tests/{WORKING_REPO_SUBDIR}/FROM_PRETRAINED",
    )

    new_model = from_pretrained_keras(f"tests/{WORKING_REPO_SUBDIR}/FROM_PRETRAINED")

    # Check the reloaded model's weights match the original model's weights
    assert tf.reduce_all(tf.equal(new_model.weights[0], model.weights[0]))


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(dummy_model_sequential, id="sequential"),
        pytest.param(dummy_model_functional, id="functional"),
    ],
)
def test_abs_path_from_pretrained(model, hf_token):
    model.build((None, 2))
    save_pretrained_keras(
        model,
        f"{WORKING_REPO_DIR}/{REPO_NAME}-FROM_PRETRAINED",
    )

    new_model = from_pretrained_keras(f"{WORKING_REPO_DIR}/{REPO_NAME}-FROM_PRETRAINED")
    assert tf.reduce_all(tf.equal(new_model.weights[0], model.weights[0]))


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(dummy_model_sequential, id="sequential"),
        pytest.param(dummy_model_functional, id="functional"),
    ],
)
def test_push_to_hub(model, hf_token):

    model.build((None, 2))
    push_to_hub_keras(
        model,
        repo_path_or_name=f"{WORKING_REPO_DIR}/{REPO_NAME}-PUSH_TO_HUB",
        api_endpoint=ENDPOINT_STAGING,
        use_auth_token=hf_token,
        git_user="ci",
        git_email="ci@dummy.com",
        config={"num": 7, "act": "gelu_fast"},
    )

    model_info = HfApi().model_info(
        f"{USER}/{REPO_NAME}-PUSH_TO_HUB",
    )
    assert model_info.modelId == f"{USER}/{REPO_NAME}-PUSH_TO_HUB"

    HfApi().delete_repo(token=hf_token, name=f"{REPO_NAME}-PUSH_TO_HUB")
