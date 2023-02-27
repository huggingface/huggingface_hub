import json
import os
import re
import unittest

import pytest

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.keras_mixin import (
    KerasModelHubMixin,
    from_pretrained_keras,
    push_to_hub_keras,
    save_pretrained_keras,
)
from huggingface_hub.repository import Repository
from huggingface_hub.utils import (
    SoftTemporaryDirectory,
    is_graphviz_available,
    is_pydot_available,
    is_tf_available,
    logging,
)

from .testing_constants import ENDPOINT_STAGING, TOKEN, USER
from .testing_utils import (
    expect_deprecation,
    repo_name,
    retry_endpoint,
    rmtree_with_retry,
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
class CommonKerasTest(unittest.TestCase):
    def tearDown(self) -> None:
        if os.path.exists(WORKING_REPO_DIR):
            rmtree_with_retry(WORKING_REPO_DIR)
        logger.info(f"Does {WORKING_REPO_DIR} exist: {os.path.exists(WORKING_REPO_DIR)}")

    @classmethod
    @expect_deprecation("set_access_token")
    def setUpClass(cls):
        """
        Share this valid token in all tests below.
        """
        cls._api = HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)
        cls._token = TOKEN
        cls._api.set_access_token(TOKEN)


class HubMixingTestKeras(CommonKerasTest):
    def test_save_pretrained(self):
        REPO_NAME = repo_name("save")
        model = DummyModel()
        model(model.dummy_inputs)
        model.save_pretrained(f"{WORKING_REPO_DIR}/{REPO_NAME}")
        files = os.listdir(f"{WORKING_REPO_DIR}/{REPO_NAME}")
        self.assertTrue("saved_model.pb" in files)
        self.assertTrue("keras_metadata.pb" in files)
        self.assertTrue("README.md" in files)
        self.assertTrue("model.png" in files)
        self.assertEqual(len(files), 7)

        model.save_pretrained(f"{WORKING_REPO_DIR}/{REPO_NAME}", config={"num": 12, "act": "gelu"})
        files = os.listdir(f"{WORKING_REPO_DIR}/{REPO_NAME}")
        self.assertTrue("config.json" in files)
        self.assertTrue("saved_model.pb" in files)
        self.assertEqual(len(files), 8)

    def test_keras_from_pretrained_weights(self):
        model = DummyModel()
        model(model.dummy_inputs)

        model.save_pretrained(f"{WORKING_REPO_DIR}/FROM_PRETRAINED")
        new_model = DummyModel.from_pretrained(f"{WORKING_REPO_DIR}/FROM_PRETRAINED")

        # Check the reloaded model's weights match the original model's weights
        self.assertTrue(tf.reduce_all(tf.equal(new_model.weights[0], model.weights[0])))

        # Check a new model's weights are not the same as the reloaded model's weights
        another_model = DummyModel()
        another_model(tf.ones([2, 2]))
        self.assertFalse(tf.reduce_all(tf.equal(new_model.weights[0], another_model.weights[0])).numpy().item())

    def test_rel_path_from_pretrained(self):
        model = DummyModel()
        model(model.dummy_inputs)
        model.save_pretrained(
            f"tests/{WORKING_REPO_SUBDIR}/FROM_PRETRAINED",
            config={"num": 10, "act": "gelu_fast"},
        )

        model = DummyModel.from_pretrained(f"tests/{WORKING_REPO_SUBDIR}/FROM_PRETRAINED")
        self.assertTrue(model.config == {"num": 10, "act": "gelu_fast"})

    def test_abs_path_from_pretrained(self):
        REPO_NAME = repo_name("FROM_PRETRAINED")
        model = DummyModel()
        model(model.dummy_inputs)
        model.save_pretrained(f"{WORKING_REPO_DIR}/{REPO_NAME}", config={"num": 10, "act": "gelu_fast"})

        model = DummyModel.from_pretrained(f"{WORKING_REPO_DIR}/{REPO_NAME}")
        self.assertDictEqual(model.config, {"num": 10, "act": "gelu_fast"})

    @retry_endpoint
    def test_push_to_hub_keras_mixin_via_http_basic(self):
        REPO_NAME = repo_name("PUSH_TO_HUB_KERAS_via_http")
        repo_id = f"{USER}/{REPO_NAME}"

        model = DummyModel()
        model(model.dummy_inputs)

        model.push_to_hub(
            repo_id=repo_id,
            api_endpoint=ENDPOINT_STAGING,
            token=self._token,
            config={"num": 7, "act": "gelu_fast"},
        )

        # Test model id exists
        model_info = self._api.model_info(repo_id)
        self.assertEqual(model_info.modelId, repo_id)

        # Test config has been pushed to hub
        tmp_config_path = hf_hub_download(repo_id=repo_id, filename="config.json", use_auth_token=self._token)
        with open(tmp_config_path) as f:
            self.assertEqual(json.load(f), {"num": 7, "act": "gelu_fast"})

        # Delete tmp file and repo
        os.remove(tmp_config_path)
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
        REPO_NAME = repo_name("save")
        model = self.model_init()

        with pytest.raises(ValueError, match="Model should be built*"):
            save_pretrained_keras(model, f"{WORKING_REPO_DIR}/{REPO_NAME}")

        model.build((None, 2))

        save_pretrained_keras(
            model,
            f"{WORKING_REPO_DIR}/{REPO_NAME}",
        )
        files = os.listdir(f"{WORKING_REPO_DIR}/{REPO_NAME}")

        self.assertIn("saved_model.pb", files)
        self.assertIn("keras_metadata.pb", files)
        self.assertIn("model.png", files)
        self.assertIn("README.md", files)
        self.assertEqual(len(files), 7)
        loaded_model = from_pretrained_keras(f"{WORKING_REPO_DIR}/{REPO_NAME}")
        self.assertIsNone(loaded_model.optimizer)

    def test_save_pretrained_model_card_fit(self):
        REPO_NAME = repo_name("save")
        model = self.model_init()
        model = self.model_fit(model)

        save_pretrained_keras(
            model,
            f"{WORKING_REPO_DIR}/{REPO_NAME}",
        )
        files = os.listdir(f"{WORKING_REPO_DIR}/{REPO_NAME}")

        self.assertIn("saved_model.pb", files)
        self.assertIn("keras_metadata.pb", files)
        self.assertIn("model.png", files)
        self.assertIn("README.md", files)
        self.assertIn("history.json", files)
        with open(f"{WORKING_REPO_DIR}/{REPO_NAME}/history.json") as f:
            history = json.load(f)

        self.assertEqual(history, model.history.history)
        self.assertEqual(len(files), 8)

    def test_save_model_card_history_removal(self):
        REPO_NAME = repo_name("save")
        model = self.model_init()
        model = self.model_fit(model)
        with SoftTemporaryDirectory() as tmpdir:
            os.makedirs(f"{tmpdir}/{WORKING_REPO_DIR}/{REPO_NAME}")
            with open(f"{tmpdir}/{WORKING_REPO_DIR}/{REPO_NAME}/history.json", "w+") as fp:
                fp.write("Keras FTW")

            with pytest.warns(UserWarning, match="`history.json` file already exists, *"):
                save_pretrained_keras(
                    model,
                    f"{tmpdir}/{WORKING_REPO_DIR}/{REPO_NAME}",
                )
                # assert that it's not the same as old history file and it's overridden
                with open(f"{tmpdir}/{WORKING_REPO_DIR}/{REPO_NAME}/history.json", "r") as f:
                    history_content = f.read()
                    self.assertNotEqual("Keras FTW", history_content)

            # Check the history is saved as a json in the repository.
            files = os.listdir(f"{tmpdir}/{WORKING_REPO_DIR}/{REPO_NAME}")
            self.assertIn("history.json", files)

            # Check that there is no "Training Metrics" section in the model card.
            # This was done in an older version.
            with open(f"{tmpdir}/{WORKING_REPO_DIR}/{REPO_NAME}/README.md", "r") as file:
                data = file.read()
            self.assertNotIn(data, "Training Metrics")

    def test_save_pretrained_optimizer_state(self):
        REPO_NAME = repo_name("save")
        model = self.model_init()

        model.build((None, 2))
        save_pretrained_keras(model, f"{WORKING_REPO_DIR}/{REPO_NAME}", include_optimizer=True)

        loaded_model = from_pretrained_keras(f"{WORKING_REPO_DIR}/{REPO_NAME}")
        self.assertIsNotNone(loaded_model.optimizer)

    def test_save_pretrained_kwargs_load_fails_without_traces(self):
        REPO_NAME = repo_name("save")
        model = self.model_init()

        model.build((None, 2))

        save_pretrained_keras(
            model,
            f"{WORKING_REPO_DIR}/{REPO_NAME}",
            include_optimizer=False,
            save_traces=False,
        )

        from_pretrained_keras(f"{WORKING_REPO_DIR}/{REPO_NAME}")
        self.assertRaises(ValueError, msg="Exception encountered when calling layer*")

    def test_from_pretrained_weights(self):
        REPO_NAME = repo_name("FROM_PRETRAINED")
        model = self.model_init()
        model.build((None, 2))

        save_pretrained_keras(model, f"{WORKING_REPO_DIR}/{REPO_NAME}")
        new_model = from_pretrained_keras(f"{WORKING_REPO_DIR}/{REPO_NAME}")

        # Check a new model's weights are not the same as the reloaded model's weights
        another_model = DummyModel()
        another_model(tf.ones([2, 2]))
        self.assertFalse(tf.reduce_all(tf.equal(new_model.weights[0], another_model.weights[0])).numpy().item())

    def test_save_pretrained_task_name_deprecation(self):
        REPO_NAME = repo_name("save")
        model = self.model_init()
        model.build((None, 2))

        with pytest.warns(
            FutureWarning,
            match="`task_name` input argument is deprecated. Pass `tags` instead.",
        ):
            save_pretrained_keras(
                model,
                f"{WORKING_REPO_DIR}/{REPO_NAME}",
                tags=["test"],
                task_name="test",
                save_traces=True,
            )

    def test_rel_path_from_pretrained(self):
        model = self.model_init()
        model.build((None, 2))
        save_pretrained_keras(
            model,
            f"tests/{WORKING_REPO_SUBDIR}/FROM_PRETRAINED",
            config={"num": 10, "act": "gelu_fast"},
        )

        new_model = from_pretrained_keras(f"tests/{WORKING_REPO_SUBDIR}/FROM_PRETRAINED")

        # Check the reloaded model's weights match the original model's weights
        self.assertTrue(tf.reduce_all(tf.equal(new_model.weights[0], model.weights[0])))

        # Check saved configuration is what we expect
        self.assertTrue(new_model.config == {"num": 10, "act": "gelu_fast"})

    def test_abs_path_from_pretrained(self):
        REPO_NAME = repo_name("FROM_PRETRAINED")
        model = self.model_init()
        model.build((None, 2))
        save_pretrained_keras(
            model,
            f"{WORKING_REPO_DIR}/{REPO_NAME}",
            config={"num": 10, "act": "gelu_fast"},
            plot_model=True,
            tags=None,
        )

        new_model = from_pretrained_keras(f"{WORKING_REPO_DIR}/{REPO_NAME}")
        self.assertTrue(tf.reduce_all(tf.equal(new_model.weights[0], model.weights[0])))
        self.assertTrue(new_model.config == {"num": 10, "act": "gelu_fast"})

    @retry_endpoint
    def test_push_to_hub_keras_sequential_via_http_basic(self):
        REPO_NAME = repo_name("PUSH_TO_HUB")
        repo_id = f"{USER}/{REPO_NAME}"
        model = self.model_init()
        model = self.model_fit(model)

        push_to_hub_keras(model, repo_id=repo_id, token=self._token, api_endpoint=ENDPOINT_STAGING)
        model_info = HfApi(endpoint=ENDPOINT_STAGING).model_info(repo_id)
        self.assertEqual(model_info.modelId, repo_id)
        self.assertTrue("README.md" in [f.rfilename for f in model_info.siblings])
        self.assertTrue("model.png" in [f.rfilename for f in model_info.siblings])
        self._api.delete_repo(repo_id=repo_id)

    @retry_endpoint
    def test_push_to_hub_keras_sequential_via_http_plot_false(self):
        REPO_NAME = repo_name("PUSH_TO_HUB")
        repo_id = f"{USER}/{REPO_NAME}"
        model = self.model_init()
        model = self.model_fit(model)

        push_to_hub_keras(
            model,
            repo_id=repo_id,
            token=self._token,
            api_endpoint=ENDPOINT_STAGING,
            plot_model=False,
        )
        model_info = HfApi(endpoint=ENDPOINT_STAGING).model_info(repo_id)
        self.assertFalse("model.png" in [f.rfilename for f in model_info.siblings])
        self._api.delete_repo(repo_id=repo_id)

    @retry_endpoint
    def test_push_to_hub_keras_via_http_override_tensorboard(self):
        """Test log directory is overwritten when pushing a keras model a 2nd time."""
        REPO_NAME = repo_name("PUSH_TO_HUB_KERAS_via_http_override_tensorboard")
        repo_id = f"{USER}/{REPO_NAME}"
        with SoftTemporaryDirectory() as tmpdir:
            os.makedirs(f"{tmpdir}/tb_log_dir")
            with open(f"{tmpdir}/tb_log_dir/tensorboard.txt", "w") as fp:
                fp.write("Keras FTW")
            model = self.model_init()
            model.build((None, 2))
            push_to_hub_keras(
                model,
                repo_id=repo_id,
                log_dir=f"{tmpdir}/tb_log_dir",
                api_endpoint=ENDPOINT_STAGING,
                token=self._token,
            )

            os.makedirs(f"{tmpdir}/tb_log_dir2")
            with open(f"{tmpdir}/tb_log_dir2/override.txt", "w") as fp:
                fp.write("Keras FTW")
            push_to_hub_keras(
                model,
                repo_id=repo_id,
                log_dir=f"{tmpdir}/tb_log_dir2",
                api_endpoint=ENDPOINT_STAGING,
                token=self._token,
            )

            model_info = self._api.model_info(repo_id)
            self.assertTrue("logs/override.txt" in [f.rfilename for f in model_info.siblings])
            self.assertFalse("logs/tensorboard.txt" in [f.rfilename for f in model_info.siblings])

            self._api.delete_repo(repo_id=repo_id)

    @retry_endpoint
    def test_push_to_hub_keras_via_http_with_model_kwargs(self):
        REPO_NAME = repo_name("PUSH_TO_HUB_KERAS_via_http_with_model_kwargs")
        repo_id = f"{USER}/{REPO_NAME}"

        model = self.model_init()
        model = self.model_fit(model)
        push_to_hub_keras(
            model,
            repo_id=repo_id,
            api_endpoint=ENDPOINT_STAGING,
            token=self._token,
            include_optimizer=True,
            save_traces=False,
        )

        model_info = HfApi(endpoint=ENDPOINT_STAGING).model_info(repo_id)
        self.assertEqual(model_info.modelId, repo_id)

        with SoftTemporaryDirectory() as tmpdir:
            Repository(local_dir=tmpdir, clone_from=ENDPOINT_STAGING + "/" + repo_id)
            from_pretrained_keras(tmpdir)

        self._api.delete_repo(repo_id=f"{REPO_NAME}")


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
        REPO_NAME = repo_name("functional")
        model = self.model_init()
        model.build((None, 2))
        self.assertTrue(model.built)

        save_pretrained_keras(model, f"{WORKING_REPO_DIR}/{REPO_NAME}")
        files = os.listdir(f"{WORKING_REPO_DIR}/{REPO_NAME}")

        self.assertIn("saved_model.pb", files)
        self.assertIn("keras_metadata.pb", files)
        self.assertEqual(len(files), 7)

    def test_save_pretrained_fit(self):
        REPO_NAME = repo_name("functional")
        model = self.model_init()
        model = self.model_fit(model)

        save_pretrained_keras(model, f"{WORKING_REPO_DIR}/{REPO_NAME}")
        files = os.listdir(f"{WORKING_REPO_DIR}/{REPO_NAME}")

        self.assertIn("saved_model.pb", files)
        self.assertIn("keras_metadata.pb", files)
        self.assertEqual(len(files), 8)
