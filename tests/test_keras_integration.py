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

WORKING_REPO_SUBDIR = f"fixtures/working_repo_{__name__.split('.')[-1]}"
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


else:
    DummyModel = None
    dummy_model_sequential = None
    dummy_model_functional = None


@require_tf
class HubMixingTestKeras(unittest.TestCase):
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
        cls._api = HfApi(endpoint=ENDPOINT_STAGING)
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
        self.assertTrue(model.hf_config == {"num": 10, "act": "gelu_fast"})

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
        self.assertDictEqual(model.hf_config, {"num": 10, "act": "gelu_fast"})

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


@require_tf
class HubKerasSequentialTest(HubMixingTestKeras):
    def model_init(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(2, activation="relu"))
        model.compile(optimizer="adam", loss="mse")
        return model

    def test_save_pretrained(self):
        model = self.model_init()

        with pytest.raises(ValueError, match="Model should be built*"):
            save_pretrained_keras(model, f"{WORKING_REPO_DIR}/{REPO_NAME}")

        model.build((None, 2))

        save_pretrained_keras(model, f"{WORKING_REPO_DIR}/{REPO_NAME}")
        files = os.listdir(f"{WORKING_REPO_DIR}/{REPO_NAME}")

        assert "saved_model.pb" in files
        assert "keras_metadata.pb" in files
        assert len(files) == 4

    def test_from_pretrained_weights(self):
        model = self.model_init()
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

    def test_rel_path_from_pretrained(self):
        model = self.model_init()
        model.build((None, 2))
        save_pretrained_keras(
            model,
            f"tests/{WORKING_REPO_SUBDIR}/FROM_PRETRAINED",
            config={"num": 10, "act": "gelu_fast"},
        )

        new_model = from_pretrained_keras(
            f"tests/{WORKING_REPO_SUBDIR}/FROM_PRETRAINED"
        )

        # Check the reloaded model's weights match the original model's weights
        self.assertTrue(tf.reduce_all(tf.equal(new_model.weights[0], model.weights[0])))

        # Check saved configuration is what we expect
        self.assertTrue(new_model.hf_config == {"num": 10, "act": "gelu_fast"})

    def test_abs_path_from_pretrained(self):
        model = self.model_init()
        model.build((None, 2))
        save_pretrained_keras(
            model,
            f"{WORKING_REPO_DIR}/{REPO_NAME}-FROM_PRETRAINED",
            config={"num": 10, "act": "gelu_fast"},
        )

        new_model = from_pretrained_keras(
            f"{WORKING_REPO_DIR}/{REPO_NAME}-FROM_PRETRAINED"
        )
        assert tf.reduce_all(tf.equal(new_model.weights[0], model.weights[0]))
        self.assertTrue(new_model.hf_config == {"num": 10, "act": "gelu_fast"})

    def test_push_to_hub(self):
        model = self.model_init()
        model.build((None, 2))
        push_to_hub_keras(
            model,
            repo_path_or_name=f"{WORKING_REPO_DIR}/{REPO_NAME}-PUSH_TO_HUB",
            api_endpoint=ENDPOINT_STAGING,
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
            config={"num": 7, "act": "gelu_fast"},
        )

        model_info = HfApi(endpoint=ENDPOINT_STAGING).model_info(
            f"{USER}/{REPO_NAME}-PUSH_TO_HUB",
        )
        assert model_info.modelId == f"{USER}/{REPO_NAME}-PUSH_TO_HUB"

        self._api.delete_repo(token=self._token, name=f"{REPO_NAME}-PUSH_TO_HUB")


@require_tf
class HubKerasFunctionalTest(HubKerasSequentialTest):
    def model_init(self):
        inputs = tf.keras.layers.Input(shape=(2,))
        x = tf.keras.layers.Dense(2, activation="relu")(inputs)
        model = tf.keras.models.Model(inputs=inputs, outputs=x)
        model.compile(optimizer="adam", loss="mse")
        return model

    def test_save_pretrained(self):
        model = self.model_init()

        self.assertTrue(model.built)

        save_pretrained_keras(model, f"{WORKING_REPO_DIR}/{REPO_NAME}")
        files = os.listdir(f"{WORKING_REPO_DIR}/{REPO_NAME}")

        self.assertIn("saved_model.pb", files)
        self.assertIn("keras_metadata.pb", files)
        self.assertEqual(len(files), 4)
