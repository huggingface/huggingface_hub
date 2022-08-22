import json
import os
import re
import tempfile
import unittest

import pytest

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.file_download import (
    is_graphviz_available,
    is_pydot_available,
    is_tf_available,
)
from huggingface_hub.keras_mixin import (
    KerasModelHubMixin,
    from_pretrained_keras,
    push_to_hub_keras,
    save_pretrained_keras,
)
from huggingface_hub.repository import Repository
from huggingface_hub.utils import logging

from .conftest import CacheDirFixture, RepoIdFixture
from .testing_constants import ENDPOINT_STAGING, TOKEN
from .testing_utils import expect_deprecation, retry_endpoint, safe_chdir


logger = logging.get_logger(__name__)

PUSH_TO_HUB_KERAS_WARNING_REGEX = re.escape(
    "Deprecated argument(s) used in 'push_to_hub_keras':"
)

if is_tf_available():
    import tensorflow as tf


def require_tf(test_case):
    """
    Decorator marking a test that requires TensorFlow, graphviz and pydot.

    These tests are skipped when TensorFlow, graphviz and pydot are installed.

    """
    if not is_tf_available() or not is_pydot_available() or not is_graphviz_available():
        return unittest.skip("test requires Tensorflow, graphviz and pydot.")(test_case)
    else:
        return test_case


if is_tf_available():
    # Define dummy mixin model...
    class DummyModel(tf.keras.Model, KerasModelHubMixin):
        def __init__(self, **kwargs):
            super().__init__()
            self.l1 = tf.keras.layers.Dense(2, activation="relu")
            dummy_batch_size = input_dim = 2
            self.dummy_inputs = tf.ones([dummy_batch_size, input_dim])

        def call(self, x):
            return self.l1(x)

else:
    DummyModel = None


@require_tf
class HubMixingTestKeras(unittest.TestCase, CacheDirFixture, RepoIdFixture):
    @classmethod
    def setUpClass(cls):
        """
        Share this valid token in all tests below.
        """
        cls._api = HfApi(endpoint=ENDPOINT_STAGING)
        cls._token = TOKEN
        cls._api.set_access_token(TOKEN)

    def test_save_pretrained(self):
        model = DummyModel()
        model(model.dummy_inputs)
        model.save_pretrained(self.cache_dir_str)
        files = os.listdir(self.cache_dir_str)
        self.assertTrue("saved_model.pb" in files)
        self.assertTrue("keras_metadata.pb" in files)
        self.assertTrue("README.md" in files)
        self.assertTrue("model.png" in files)
        self.assertEqual(len(files), 6)

        model.save_pretrained(self.cache_dir_str, config={"num": 12, "act": "gelu"})
        files = os.listdir(self.cache_dir_str)
        self.assertTrue("config.json" in files)
        self.assertTrue("saved_model.pb" in files)
        self.assertEqual(len(files), 7)

    def test_keras_from_pretrained_weights(self):
        model = DummyModel()
        model(model.dummy_inputs)

        model.save_pretrained(self.cache_dir_str)
        new_model = DummyModel.from_pretrained(self.cache_dir_str)

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
        with safe_chdir(self.cache_dir.parent):
            rel_path = "./" + self.cache_dir.name  # building dumb relative path in /tmp

            model = DummyModel()
            model(model.dummy_inputs)
            model.save_pretrained(rel_path, config={"num": 10, "act": "gelu_fast"})

            model = DummyModel.from_pretrained(rel_path)
            self.assertTrue(model.config == {"num": 10, "act": "gelu_fast"})

    def test_abs_path_from_pretrained(self):
        model = DummyModel()
        model(model.dummy_inputs)
        model.save_pretrained(
            self.cache_dir_str, config={"num": 10, "act": "gelu_fast"}
        )

        model = DummyModel.from_pretrained(self.cache_dir_str)
        self.assertDictEqual(model.config, {"num": 10, "act": "gelu_fast"})

    @retry_endpoint
    def test_push_to_hub_keras_mixin_via_http_basic(self):
        model = DummyModel()
        model(model.dummy_inputs)

        model.push_to_hub(
            repo_id=self.repo_id,
            api_endpoint=ENDPOINT_STAGING,
            token=self._token,
            config={"num": 7, "act": "gelu_fast"},
        )

        # Test model id exists
        model_info = self._api.model_info(self.repo_id, token=self._token)
        self.assertEqual(model_info.modelId, self.repo_id)

        # Test config has been pushed to hub
        tmp_config_path = hf_hub_download(
            repo_id=self.repo_id, filename="config.json", use_auth_token=self._token
        )
        with open(tmp_config_path) as f:
            self.assertEqual(json.load(f), {"num": 7, "act": "gelu_fast"})

        # Delete tmp file and repo
        os.remove(tmp_config_path)
        self._api.delete_repo(repo_id=self.repo_id, token=self._token)

    @retry_endpoint
    @expect_deprecation("push_to_hub")
    def test_push_to_hub_keras_mixin_via_git_deprecated(self):
        model = DummyModel()
        model(model.dummy_inputs)

        # Trick to get a non-existing directory with expected repo_name
        repo_path_or_name = str(self.cache_dir / self.repo_name)

        model.push_to_hub(
            repo_path_or_name=repo_path_or_name,
            api_endpoint=ENDPOINT_STAGING,
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
            config={"num": 7, "act": "gelu_fast"},
        )

        model_info = self._api.model_info(self.repo_id)
        self.assertEqual(model_info.modelId, self.repo_id)
        self._api.delete_repo(repo_id=self.repo_id, token=self._token)


@require_tf
class HubKerasSequentialTest(HubMixingTestKeras):
    def model_init(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(2, activation="relu"))
        model.compile(optimizer="adam", loss="mse")
        return model

    def model_fit(self, model):
        x = tf.constant([[0.44, 0.90], [0.65, 0.39]])
        y = tf.constant([[1, 1], [0, 0]])
        model.fit(x, y)
        return model

    def test_save_pretrained(self):
        model = self.model_init()

        with pytest.raises(ValueError, match="Model should be built*"):
            save_pretrained_keras(model, self.cache_dir_str)

        model.build((None, 2))

        save_pretrained_keras(model, self.cache_dir_str)
        files = os.listdir(self.cache_dir_str)

        self.assertIn("saved_model.pb", files)
        self.assertIn("keras_metadata.pb", files)
        self.assertIn("model.png", files)
        self.assertIn("README.md", files)
        self.assertEqual(len(files), 6)
        loaded_model = from_pretrained_keras(self.cache_dir_str)
        self.assertIsNone(loaded_model.optimizer)

    def test_save_pretrained_model_card_fit(self):
        model = self.model_init()
        model = self.model_fit(model)

        save_pretrained_keras(model, self.cache_dir_str)
        files = os.listdir(self.cache_dir_str)

        self.assertIn("saved_model.pb", files)
        self.assertIn("keras_metadata.pb", files)
        self.assertIn("model.png", files)
        self.assertIn("README.md", files)
        self.assertIn("history.json", files)
        with open(f"{self.cache_dir_str}/history.json") as f:
            history = json.load(f)

        self.assertEqual(history, model.history.history)
        self.assertEqual(len(files), 7)

    def test_save_model_card_history_removal(self):
        model = self.model_init()
        model = self.model_fit(model)

        with open(f"{self.cache_dir_str}/history.json", "w+") as fp:
            fp.write("Keras FTW")

        with pytest.warns(UserWarning, match="`history.json` file already exists, *"):
            save_pretrained_keras(
                model,
                self.cache_dir_str,
            )
            # assert that it's not the same as old history file and it's overridden
            with open(f"{self.cache_dir_str}/history.json", "r") as f:
                history_content = f.read()
                self.assertNotEqual("Keras FTW", history_content)

        # Check the history is saved as a json in the repository.
        files = os.listdir(self.cache_dir_str)
        self.assertIn("history.json", files)

        # Check that there is no "Training Metrics" section in the model card.
        # This was done in an older version.
        with open(f"{self.cache_dir_str}/README.md", "r") as file:
            data = file.read()
        self.assertNotIn(data, "Training Metrics")

    def test_save_pretrained_optimizer_state(self):
        model = self.model_init()

        model.build((None, 2))
        save_pretrained_keras(model, self.cache_dir_str, include_optimizer=True)

        loaded_model = from_pretrained_keras(self.cache_dir_str)
        self.assertIsNotNone(loaded_model.optimizer)

    def test_save_pretrained_kwargs_load_fails_without_traces(self):
        model = self.model_init()

        model.build((None, 2))

        save_pretrained_keras(
            model, self.cache_dir_str, include_optimizer=False, save_traces=False
        )

        from_pretrained_keras(self.cache_dir_str)
        self.assertRaises(ValueError, msg="Exception encountered when calling layer*")

    def test_from_pretrained_weights(self):
        model = self.model_init()
        model.build((None, 2))

        save_pretrained_keras(model, self.cache_dir_str)
        new_model = from_pretrained_keras(self.cache_dir_str)

        # Check a new model's weights are not the same as the reloaded model's weights
        another_model = DummyModel()
        another_model(tf.ones([2, 2]))
        self.assertFalse(
            tf.reduce_all(tf.equal(new_model.weights[0], another_model.weights[0]))
            .numpy()
            .item()
        )

    def test_save_pretrained_task_name_deprecation(self):
        model = self.model_init()
        model.build((None, 2))

        with pytest.warns(
            FutureWarning,
            match="`task_name` input argument is deprecated. Pass `tags` instead.",
        ):
            save_pretrained_keras(
                model,
                self.cache_dir_str,
                tags=["test"],
                task_name="test",
                save_traces=True,
            )

    def test_rel_path_from_pretrained(self):
        with safe_chdir(self.cache_dir.parent):
            rel_path = "./" + self.cache_dir.name  # building dumb relative path in /tmp

            model = self.model_init()
            model.build((None, 2))
            save_pretrained_keras(
                model, rel_path, config={"num": 10, "act": "gelu_fast"}
            )

            new_model = from_pretrained_keras(rel_path)

            # Check the reloaded model's weights match the original model's weights
            self.assertTrue(
                tf.reduce_all(tf.equal(new_model.weights[0], model.weights[0]))
            )

            # Check saved configuration is what we expect
            self.assertTrue(new_model.config == {"num": 10, "act": "gelu_fast"})

    def test_abs_path_from_pretrained(self):
        model = self.model_init()
        model.build((None, 2))
        save_pretrained_keras(
            model,
            self.cache_dir_str,
            config={"num": 10, "act": "gelu_fast"},
            plot_model=True,
            tags=None,
        )

        new_model = from_pretrained_keras(self.cache_dir_str)
        self.assertTrue(tf.reduce_all(tf.equal(new_model.weights[0], model.weights[0])))
        self.assertTrue(new_model.config == {"num": 10, "act": "gelu_fast"})

    @retry_endpoint
    def test_push_to_hub_keras_sequential_via_http_basic(self):
        model = self.model_init()
        model = self.model_fit(model)

        push_to_hub_keras(
            model,
            repo_id=self.repo_id,
            token=self._token,
            api_endpoint=ENDPOINT_STAGING,
        )
        model_info = HfApi(endpoint=ENDPOINT_STAGING).model_info(self.repo_id)
        self.assertEqual(model_info.modelId, self.repo_id)
        self.assertTrue("README.md" in [f.rfilename for f in model_info.siblings])
        self.assertTrue("model.png" in [f.rfilename for f in model_info.siblings])
        self._api.delete_repo(repo_id=self.repo_id, token=self._token)

    @retry_endpoint
    def test_push_to_hub_keras_sequential_via_http_plot_false(self):
        model = self.model_init()
        model = self.model_fit(model)

        push_to_hub_keras(
            model,
            repo_id=self.repo_id,
            token=self._token,
            api_endpoint=ENDPOINT_STAGING,
            plot_model=False,
        )
        model_info = HfApi(endpoint=ENDPOINT_STAGING).model_info(self.repo_id)
        self.assertFalse("model.png" in [f.rfilename for f in model_info.siblings])
        self._api.delete_repo(repo_id=self.repo_id, token=self._token)

    @retry_endpoint
    @expect_deprecation("push_to_hub_keras")
    def test_push_to_hub_keras_sequential_via_git_deprecated(self):
        model = self.model_init()
        model.build((None, 2))

        # Trick to get a non-existing directory with expected repo_name
        repo_path_or_name = str(self.cache_dir / self.repo_name)

        model.push_to_hub(
            repo_path_or_name=repo_path_or_name,
            api_endpoint=ENDPOINT_STAGING,
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
            config={"num": 7, "act": "gelu_fast"},
            include_optimizer=False,
        )

        model_info = HfApi(endpoint=ENDPOINT_STAGING).model_info(self.repo_id)
        self.assertEqual(model_info.modelId, self.repo_id)
        self.assertTrue("README.md" in [f.rfilename for f in model_info.siblings])
        self.assertTrue("model.png" in [f.rfilename for f in model_info.siblings])
        self._api.delete_repo(repo_id=self.repo_name, token=self._token)

    @retry_endpoint
    def test_push_to_hub_keras_via_http_override_tensorboard(self):
        """Test log directory is overwritten when pushing a keras model a 2nd time."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            os.makedirs(f"{tmpdirname}/tb_log_dir")
            with open(f"{tmpdirname}/tb_log_dir/tensorboard.txt", "w") as fp:
                fp.write("Keras FTW")
            model = self.model_init()
            model.build((None, 2))
            push_to_hub_keras(
                model,
                repo_id=self.repo_id,
                log_dir=f"{tmpdirname}/tb_log_dir",
                api_endpoint=ENDPOINT_STAGING,
                token=self._token,
            )

            os.makedirs(f"{tmpdirname}/tb_log_dir2")
            with open(f"{tmpdirname}/tb_log_dir2/override.txt", "w") as fp:
                fp.write("Keras FTW")
            push_to_hub_keras(
                model,
                repo_id=self.repo_id,
                log_dir=f"{tmpdirname}/tb_log_dir2",
                api_endpoint=ENDPOINT_STAGING,
                token=self._token,
            )

            model_info = HfApi(endpoint=ENDPOINT_STAGING).model_info(self.repo_id)
            self.assertTrue(
                "logs/override.txt" in [f.rfilename for f in model_info.siblings]
            )
            self.assertFalse(
                "logs/tensorboard.txt" in [f.rfilename for f in model_info.siblings]
            )

            self._api.delete_repo(repo_id=self.repo_id, token=self._token)

    @retry_endpoint
    def test_push_to_hub_keras_via_http_with_model_kwargs(self):
        model = self.model_init()
        model = self.model_fit(model)
        push_to_hub_keras(
            model,
            repo_id=self.repo_id,
            api_endpoint=ENDPOINT_STAGING,
            token=self._token,
            include_optimizer=True,
            save_traces=False,
        )

        model_info = HfApi(endpoint=ENDPOINT_STAGING).model_info(self.repo_id)
        self.assertEqual(model_info.modelId, self.repo_id)

        with tempfile.TemporaryDirectory() as tmpdirname:
            Repository(
                local_dir=tmpdirname,
                clone_from=ENDPOINT_STAGING + "/" + self.repo_id,
                use_auth_token=self._token,
            )
            from_pretrained_keras(tmpdirname)
            self.assertRaises(
                ValueError, msg="Exception encountered when calling layer*"
            )

        self._api.delete_repo(repo_id=self.repo_name, token=self._token)


@require_tf
class HubKerasFunctionalTest(HubKerasSequentialTest):
    def model_init(self):
        inputs = tf.keras.layers.Input(shape=(2,))
        outputs = tf.keras.layers.Dense(2, activation="relu")(inputs)
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="mse")
        return model

    def test_save_pretrained(self):
        model = self.model_init()
        model.build((None, 2))
        self.assertTrue(model.built)

        save_pretrained_keras(model, self.cache_dir_str)
        files = os.listdir(self.cache_dir_str)

        self.assertIn("saved_model.pb", files)
        self.assertIn("keras_metadata.pb", files)
        self.assertEqual(len(files), 6)

    def test_save_pretrained_fit(self):
        model = self.model_init()
        model = self.model_fit(model)

        save_pretrained_keras(model, self.cache_dir_str)
        files = os.listdir(self.cache_dir_str)

        self.assertIn("saved_model.pb", files)
        self.assertIn("keras_metadata.pb", files)
        self.assertEqual(len(files), 7)
