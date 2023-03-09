import json
import os
import re
import unittest
from pathlib import Path

import pytest

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.keras_mixin import (
    KerasModelHubMixin,
    from_pretrained_keras,
    push_to_hub_keras,
    save_pretrained_keras,
)
from huggingface_hub.utils import (
    is_graphviz_available,
    is_pydot_available,
    is_tf_available,
    logging,
)

from .testing_constants import ENDPOINT_STAGING, TOKEN, USER
from .testing_utils import (
    repo_name,
    retry_endpoint,
)


logger = logging.get_logger(__name__)

WORKING_REPO_SUBDIR = f"fixtures/working_repo_{__name__.split('.')[-1]}"
WORKING_REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), WORKING_REPO_SUBDIR)

PUSH_TO_HUB_KERAS_WARNING_REGEX = re.escape("Deprecated argument(s) used in 'push_to_hub_keras':")

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
@pytest.mark.usefixtures("fx_cache_dir")
class CommonKerasTest(unittest.TestCase):
    cache_dir: Path

    @classmethod
    def setUpClass(cls):
        """
        Share this valid token in all tests below.
        """
        cls._api = HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)


class HubMixingTestKeras(CommonKerasTest):
    def test_save_pretrained(self):
        model = DummyModel()
        model(model.dummy_inputs)

        model.save_pretrained(self.cache_dir)
        files = os.listdir(self.cache_dir)
        self.assertTrue("saved_model.pb" in files)
        self.assertTrue("keras_metadata.pb" in files)
        self.assertTrue("README.md" in files)
        self.assertTrue("model.png" in files)
        self.assertEqual(len(files), 7)

        model.save_pretrained(self.cache_dir, config={"num": 12, "act": "gelu"})
        files = os.listdir(self.cache_dir)
        self.assertTrue("config.json" in files)
        self.assertTrue("saved_model.pb" in files)
        self.assertEqual(len(files), 8)

    def test_keras_from_pretrained_weights(self):
        model = DummyModel()
        model(model.dummy_inputs)

        model.save_pretrained(self.cache_dir)
        new_model = DummyModel.from_pretrained(self.cache_dir)

        # Check the reloaded model's weights match the original model's weights
        self.assertTrue(tf.reduce_all(tf.equal(new_model.weights[0], model.weights[0])))

        # Check a new model's weights are not the same as the reloaded model's weights
        another_model = DummyModel()
        another_model(tf.ones([2, 2]))
        self.assertFalse(tf.reduce_all(tf.equal(new_model.weights[0], another_model.weights[0])).numpy().item())

    def test_abs_path_from_pretrained(self):
        model = DummyModel()
        model(model.dummy_inputs)
        model.save_pretrained(self.cache_dir, config={"num": 10, "act": "gelu_fast"})
        model = DummyModel.from_pretrained(self.cache_dir)
        self.assertTrue(model.config == {"num": 10, "act": "gelu_fast"})

    @retry_endpoint
    def test_push_to_hub_keras_mixin_via_http_basic(self):
        repo_id = f"{USER}/{repo_name()}"

        model = DummyModel()
        model(model.dummy_inputs)

        model.push_to_hub(
            repo_id=repo_id, api_endpoint=ENDPOINT_STAGING, token=TOKEN, config={"num": 7, "act": "gelu_fast"}
        )

        # Test model id exists
        model_info = self._api.model_info(repo_id)
        self.assertEqual(model_info.modelId, repo_id)

        # Test config has been pushed to hub
        config_path = hf_hub_download(
            repo_id=repo_id, filename="config.json", use_auth_token=TOKEN, cache_dir=self.cache_dir
        )
        with open(config_path) as f:
            self.assertEqual(json.load(f), {"num": 7, "act": "gelu_fast"})

        # Delete tmp file and repo
        self._api.delete_repo(repo_id=repo_id)


@require_tf
class HubKerasSequentialTest(CommonKerasTest):
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
            save_pretrained_keras(model, save_directory=self.cache_dir)
        model.build((None, 2))

        save_pretrained_keras(model, save_directory=self.cache_dir)
        files = os.listdir(self.cache_dir)
        self.assertIn("saved_model.pb", files)
        self.assertIn("keras_metadata.pb", files)
        self.assertIn("model.png", files)
        self.assertIn("README.md", files)
        self.assertEqual(len(files), 7)

        loaded_model = from_pretrained_keras(self.cache_dir)
        self.assertIsNone(loaded_model.optimizer)

    def test_save_pretrained_model_card_fit(self):
        model = self.model_init()
        model = self.model_fit(model)

        save_pretrained_keras(model, save_directory=self.cache_dir)
        files = os.listdir(self.cache_dir)
        history = json.loads((self.cache_dir / "history.json").read_text())

        self.assertIn("saved_model.pb", files)
        self.assertIn("keras_metadata.pb", files)
        self.assertIn("model.png", files)
        self.assertIn("README.md", files)
        self.assertIn("history.json", files)
        self.assertEqual(history, model.history.history)
        self.assertEqual(len(files), 8)

    def test_save_model_card_history_removal(self):
        model = self.model_init()
        model = self.model_fit(model)

        history_path = self.cache_dir / "history.json"
        history_path.write_text("Keras FTW")

        with pytest.warns(UserWarning, match="`history.json` file already exists, *"):
            save_pretrained_keras(model, save_directory=self.cache_dir)
            # assert that it's not the same as old history file and it's overridden
            self.assertNotEqual("Keras FTW", history_path.read_text())

            # Check the history is saved as a json in the repository.
            files = os.listdir(self.cache_dir)
            self.assertIn("history.json", files)

            # Check that there is no "Training Metrics" section in the model card.
            # This was done in an older version.
            self.assertNotIn("Training Metrics", (self.cache_dir / "README.md").read_text())

    def test_save_pretrained_optimizer_state(self):
        model = self.model_init()
        model.build((None, 2))
        save_pretrained_keras(model, self.cache_dir, include_optimizer=True)
        loaded_model = from_pretrained_keras(self.cache_dir)
        self.assertIsNotNone(loaded_model.optimizer)

    def test_from_pretrained_weights(self):
        model = self.model_init()
        model.build((None, 2))

        save_pretrained_keras(model, self.cache_dir)
        new_model = from_pretrained_keras(self.cache_dir)

        # Check a new model's weights are not the same as the reloaded model's weights
        another_model = DummyModel()
        another_model(tf.ones([2, 2]))
        self.assertFalse(tf.reduce_all(tf.equal(new_model.weights[0], another_model.weights[0])).numpy().item())

    def test_save_pretrained_task_name_deprecation(self):
        model = self.model_init()
        model.build((None, 2))

        with pytest.warns(
            FutureWarning,
            match="`task_name` input argument is deprecated. Pass `tags` instead.",
        ):
            save_pretrained_keras(model, self.cache_dir, tags=["test"], task_name="test", save_traces=True)

    def test_abs_path_from_pretrained(self):
        model = self.model_init()
        model.build((None, 2))
        save_pretrained_keras(
            model, self.cache_dir, config={"num": 10, "act": "gelu_fast"}, plot_model=True, tags=None
        )
        new_model = from_pretrained_keras(self.cache_dir)
        self.assertTrue(tf.reduce_all(tf.equal(new_model.weights[0], model.weights[0])))
        self.assertTrue(new_model.config == {"num": 10, "act": "gelu_fast"})

    @retry_endpoint
    def test_push_to_hub_keras_sequential_via_http_basic(self):
        repo_id = f"{USER}/{repo_name()}"
        model = self.model_init()
        model = self.model_fit(model)

        push_to_hub_keras(model, repo_id=repo_id, token=TOKEN, api_endpoint=ENDPOINT_STAGING)
        model_info = self._api.model_info(repo_id)
        self.assertEqual(model_info.modelId, repo_id)
        self.assertTrue("README.md" in [f.rfilename for f in model_info.siblings])
        self.assertTrue("model.png" in [f.rfilename for f in model_info.siblings])
        self._api.delete_repo(repo_id=repo_id)

    @retry_endpoint
    def test_push_to_hub_keras_sequential_via_http_plot_false(self):
        repo_id = f"{USER}/{repo_name()}"
        model = self.model_init()
        model = self.model_fit(model)

        push_to_hub_keras(model, repo_id=repo_id, token=TOKEN, api_endpoint=ENDPOINT_STAGING, plot_model=False)
        model_info = self._api.model_info(repo_id)
        self.assertFalse("model.png" in [f.rfilename for f in model_info.siblings])
        self._api.delete_repo(repo_id=repo_id)

    @retry_endpoint
    def test_push_to_hub_keras_via_http_override_tensorboard(self):
        """Test log directory is overwritten when pushing a keras model a 2nd time."""
        repo_id = f"{USER}/{repo_name()}"

        log_dir = self.cache_dir / "tb_log_dir"
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "tensorboard.txt").write_text("Keras FTW")

        model = self.model_init()
        model.build((None, 2))
        push_to_hub_keras(model, repo_id=repo_id, log_dir=log_dir, api_endpoint=ENDPOINT_STAGING, token=TOKEN)

        log_dir2 = self.cache_dir / "tb_log_dir2"
        log_dir2.mkdir(parents=True, exist_ok=True)
        (log_dir2 / "override.txt").write_text("Keras FTW")
        push_to_hub_keras(model, repo_id=repo_id, log_dir=log_dir2, api_endpoint=ENDPOINT_STAGING, token=TOKEN)

        files = self._api.list_repo_files(repo_id)
        self.assertIn("logs/override.txt", files)
        self.assertNotIn("logs/tensorboard.txt", files)

        self._api.delete_repo(repo_id=repo_id)

    @retry_endpoint
    def test_push_to_hub_keras_via_http_with_model_kwargs(self):
        repo_id = f"{USER}/{repo_name()}"

        model = self.model_init()
        model = self.model_fit(model)
        push_to_hub_keras(
            model,
            repo_id=repo_id,
            api_endpoint=ENDPOINT_STAGING,
            token=TOKEN,
            include_optimizer=True,
            save_traces=False,
        )

        model_info = self._api.model_info(repo_id)
        self.assertEqual(model_info.modelId, repo_id)

        snapshot_path = snapshot_download(repo_id=repo_id, cache_dir=self.cache_dir)
        from_pretrained_keras(snapshot_path)

        self._api.delete_repo(repo_id)


@require_tf
class HubKerasFunctionalTest(CommonKerasTest):
    def model_init(self):
        inputs = tf.keras.layers.Input(shape=(2,))
        outputs = tf.keras.layers.Dense(2, activation="relu")(inputs)
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="mse")
        return model

    def model_fit(self, model):
        x = tf.constant([[0.44, 0.90], [0.65, 0.39]])
        y = tf.constant([[1, 1], [0, 0]])
        model.fit(x, y)
        return model

    def test_save_pretrained(self):
        model = self.model_init()
        model.build((None, 2))
        self.assertTrue(model.built)

        save_pretrained_keras(model, self.cache_dir)
        files = os.listdir(self.cache_dir)

        self.assertIn("saved_model.pb", files)
        self.assertIn("keras_metadata.pb", files)
        self.assertEqual(len(files), 7)

    def test_save_pretrained_fit(self):
        model = self.model_init()
        model = self.model_fit(model)

        save_pretrained_keras(model, self.cache_dir)
        files = os.listdir(self.cache_dir)

        self.assertIn("saved_model.pb", files)
        self.assertIn("keras_metadata.pb", files)
        self.assertEqual(len(files), 8)
