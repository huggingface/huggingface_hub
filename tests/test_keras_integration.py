import math
import os
import shutil
import tempfile
import time
import unittest
import uuid
from subprocess import check_output

import numpy as np
import pytest

from huggingface_hub import HfApi
from huggingface_hub.file_download import (
    is_graphviz_available,
    is_pydot_available,
    is_tf_available,
)
from huggingface_hub.keras_mixin import (
    KerasModelHubMixin,
    PushToHubCallback,
    from_pretrained_keras,
    push_to_hub_keras,
    save_pretrained_keras,
)
from huggingface_hub.utils import logging

from .testing_constants import ENDPOINT_STAGING, PASS, USER
from .testing_utils import retry_endpoint, set_write_permission_and_retry


def repo_name(id=uuid.uuid4().hex[:6]):
    return "keras-repo-{0}-{1}".format(id, int(time.time() * 10e3))


logger = logging.get_logger(__name__)

WORKING_REPO_SUBDIR = f"fixtures/working_repo_{__name__.split('.')[-1]}"
WORKING_REPO_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), WORKING_REPO_SUBDIR
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
class HubMixingTestKeras(unittest.TestCase):
    def tearDown(self) -> None:
        if os.path.exists(WORKING_REPO_DIR):
            shutil.rmtree(WORKING_REPO_DIR, onerror=set_write_permission_and_retry)
        logger.info(
            f"Does {WORKING_REPO_DIR} exist: {os.path.exists(WORKING_REPO_DIR)}"
        )

    @classmethod
    def setUpClass(cls):
        """
        Share this valid token in all tests below.
        """
        cls._api = HfApi(endpoint=ENDPOINT_STAGING)
        cls._token = cls._api.login(username=USER, password=PASS)

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
        self.assertEqual(len(files), 6)

        model.save_pretrained(
            f"{WORKING_REPO_DIR}/{REPO_NAME}", config={"num": 12, "act": "gelu"}
        )
        files = os.listdir(f"{WORKING_REPO_DIR}/{REPO_NAME}")
        self.assertTrue("config.json" in files)
        self.assertTrue("saved_model.pb" in files)
        self.assertEqual(len(files), 7)

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
        REPO_NAME = repo_name("FROM_PRETRAINED")
        model = DummyModel()
        model(model.dummy_inputs)
        model.save_pretrained(
            f"{WORKING_REPO_DIR}/{REPO_NAME}", config={"num": 10, "act": "gelu_fast"}
        )

        model = DummyModel.from_pretrained(f"{WORKING_REPO_DIR}/{REPO_NAME}")
        self.assertDictEqual(model.config, {"num": 10, "act": "gelu_fast"})

    @retry_endpoint
    def test_push_to_hub(self):
        REPO_NAME = repo_name("PUSH_TO_HUB")
        model = DummyModel()
        model(model.dummy_inputs)
        model.push_to_hub(
            repo_path_or_name=f"{WORKING_REPO_DIR}/{REPO_NAME}",
            api_endpoint=ENDPOINT_STAGING,
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
            config={"num": 7, "act": "gelu_fast"},
        )

        model_info = self._api.model_info(
            f"{USER}/{REPO_NAME}",
        )
        self.assertEqual(model_info.modelId, f"{USER}/{REPO_NAME}")

        self._api.delete_repo(repo_id=f"{REPO_NAME}", token=self._token)


@require_tf
class HubKerasSequentialTest(HubMixingTestKeras):
    def model_init(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(None, 2)))
        model.add(tf.keras.layers.Dense(2, activation="relu"))
        model.compile(optimizer="adam", loss="mse")
        return model

    def dummy_training_data(self, batch_size=None):
        def batch_data(data, batch_size):
            batched_features = []
            for element in range(0, len(data), batch_size):
                batched_features.append(
                    data[element : min(element + batch_size, len(data))]
                )
            return batched_features

        features = np.ones((9, 2))
        labels = np.ones((9, 2))

        if batch_size is not None:
            features = tf.constant(batch_data(features, batch_size))
            labels = tf.constant(batch_data(labels, batch_size))
        else:
            features = tf.constant(features)
            labels = tf.constant(labels)

        return features, labels

    def model_fit(self, model, num_epochs, batch_size=None, callback=None):
        x, y = self.dummy_training_data(batch_size)
        if batch_size is not None:
            model.fit(
                tf.data.Dataset.from_tensor_slices((x, y)),
                callbacks=callback,
                epochs=num_epochs,
                batch_size=batch_size,
            )
        else:
            model.fit(
                x, y, callbacks=callback, epochs=num_epochs, batch_size=batch_size
            )
        return model

    def test_callback_end_of_training(self):
        REPO_NAME = repo_name("PUSH_TO_HUB")
        model = self.model_init()

        push_to_hub_callback = PushToHubCallback(
            save_strategy="end_of_training",
            repo_path_or_name=f"{WORKING_REPO_DIR}/{REPO_NAME}",
            api_endpoint=ENDPOINT_STAGING,
            token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
        )

        model = self.model_fit(
            model,
            callback=[push_to_hub_callback],
            num_epochs=3,
        )

        self.assertEqual(
            len(
                check_output(
                    f"git --git-dir {WORKING_REPO_DIR}/{REPO_NAME}/.git log --pretty=oneline".split()
                )
                .decode()
                .split("\n")
            ),
            3,
        )

    def test_callback_batch(self):
        REPO_NAME = repo_name("PUSH_TO_HUB")
        model = self.model_init()
        num_epochs = 2
        save_steps = 2
        batch_size = 3

        x, _ = self.dummy_training_data(batch_size=3)
        num_batches = x.shape[0]
        push_to_hub_callback = PushToHubCallback(
            save_strategy="steps",
            repo_path_or_name=f"{WORKING_REPO_DIR}/{REPO_NAME}",
            api_endpoint=ENDPOINT_STAGING,
            token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
            save_steps=save_steps,
        )

        model = self.model_fit(
            model,
            callback=[push_to_hub_callback],
            num_epochs=num_epochs,
            batch_size=batch_size,
        )

        self.assertEqual(
            len(
                check_output(
                    f"git --git-dir {WORKING_REPO_DIR}/{REPO_NAME}/.git log --pretty=oneline".split()
                )
                .decode()
                .split("\n")
            ),
            num_epochs * math.floor(num_batches / save_steps) + 3,
        )

    def test_callback_epoch(self):
        REPO_NAME = repo_name("PUSH_TO_HUB")
        model = self.model_init()

        push_to_hub_callback = PushToHubCallback(
            save_strategy="epoch",
            repo_path_or_name=f"{WORKING_REPO_DIR}/{REPO_NAME}",
            api_endpoint=ENDPOINT_STAGING,
            token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
        )
        num_epochs = 3

        model = self.model_fit(
            model, callback=[push_to_hub_callback], num_epochs=num_epochs
        )
        self.assertEqual(
            len(
                check_output(
                    f"git --git-dir {WORKING_REPO_DIR}/{REPO_NAME}/.git log --pretty=oneline".split()
                )
                .decode()
                .split("\n")
            ),
            num_epochs + 3,
        )
        # epochs, initial commit, end of training commit, one more line

    def test_save_pretrained(self):
        REPO_NAME = repo_name("save")
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(2, activation="relu"))
        model.compile(optimizer="adam", loss="mse")

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

        self.assertEqual(len(files), 6)
        loaded_model = from_pretrained_keras(f"{WORKING_REPO_DIR}/{REPO_NAME}")
        self.assertIsNone(loaded_model.optimizer)

    def test_save_pretrained_model_card_fit(self):
        REPO_NAME = repo_name("save")
        model = self.model_init()
        model = self.model_fit(model, num_epochs=2)

        save_pretrained_keras(
            model,
            f"{WORKING_REPO_DIR}/{REPO_NAME}",
        )
        files = os.listdir(f"{WORKING_REPO_DIR}/{REPO_NAME}")

        self.assertIn("saved_model.pb", files)
        self.assertIn("keras_metadata.pb", files)
        self.assertIn("model.png", files)
        self.assertIn("README.md", files)
        self.assertEqual(len(files), 6)

    def test_save_pretrained_optimizer_state(self):
        REPO_NAME = repo_name("save")
        model = self.model_init()

        model.build((None, 2))
        save_pretrained_keras(
            model, f"{WORKING_REPO_DIR}/{REPO_NAME}", include_optimizer=True
        )

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
        REPO_NAME = repo_name("from_pretrained_weights")
        model = self.model_init()
        model.build((None, 2))

        save_pretrained_keras(model, f"{WORKING_REPO_DIR}/{REPO_NAME}")
        new_model = from_pretrained_keras(f"{WORKING_REPO_DIR}/{REPO_NAME}")

        # Check a new model's weights are not the same as the reloaded model's weights
        another_model = DummyModel()
        another_model(tf.ones([2, 2]))
        self.assertFalse(
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
            task_name=None,
        )

        new_model = from_pretrained_keras(f"{WORKING_REPO_DIR}/{REPO_NAME}")
        self.assertTrue(tf.reduce_all(tf.equal(new_model.weights[0], model.weights[0])))
        self.assertTrue(new_model.config == {"num": 10, "act": "gelu_fast"})

    @retry_endpoint
    def test_push_to_hub(self):
        REPO_NAME = repo_name("PUSH_TO_HUB")
        model = self.model_init()
        self.model_fit(model, num_epochs=2)
        push_to_hub_keras(
            model,
            repo_path_or_name=f"{WORKING_REPO_DIR}/{REPO_NAME}",
            api_endpoint=ENDPOINT_STAGING,
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
            config={"num": 7, "act": "gelu_fast"},
            include_optimizer=False,
        )

        model_info = HfApi(endpoint=ENDPOINT_STAGING).model_info(
            f"{USER}/{REPO_NAME}",
        )
        self.assertEqual(model_info.modelId, f"{USER}/{REPO_NAME}")

        self._api.delete_repo(repo_id=f"{REPO_NAME}", token=self._token)

    @retry_endpoint
    def test_push_to_hub_model_card_build(self):
        REPO_NAME = repo_name("PUSH_TO_HUB")
        model = self.model_init()
        model.build((None, 2))
        push_to_hub_keras(
            model,
            repo_path_or_name=f"{WORKING_REPO_DIR}/{REPO_NAME}",
            api_endpoint=ENDPOINT_STAGING,
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
        )
        model_info = HfApi(endpoint=ENDPOINT_STAGING).model_info(
            f"{USER}/{REPO_NAME}",
        )
        self.assertTrue("README.md" in [f.rfilename for f in model_info.siblings])
        self.assertTrue("model.png" in [f.rfilename for f in model_info.siblings])
        self._api.delete_repo(repo_id=f"{REPO_NAME}", token=self._token)

    def test_push_to_hub_model_card(self):
        REPO_NAME = repo_name("PUSH_TO_HUB")
        model = self.model_init()
        model = self.model_fit(model, num_epochs=2)
        push_to_hub_keras(
            model,
            repo_path_or_name=f"{WORKING_REPO_DIR}/{REPO_NAME}",
            api_endpoint=ENDPOINT_STAGING,
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
            task_name="object-detection",
        )
        model_info = HfApi(endpoint=ENDPOINT_STAGING).model_info(
            f"{USER}/{REPO_NAME}",
        )
        self.assertTrue("README.md" in [f.rfilename for f in model_info.siblings])
        self.assertTrue("model.png" in [f.rfilename for f in model_info.siblings])
        self._api.delete_repo(name=f"{REPO_NAME}", token=self._token)

    @retry_endpoint
    def test_push_to_hub_model_card_plot_false(self):
        REPO_NAME = repo_name("PUSH_TO_HUB")
        model = self.model_init()
        model = self.model_fit(model, num_epochs=2)
        push_to_hub_keras(
            model,
            repo_path_or_name=f"{WORKING_REPO_DIR}/{REPO_NAME}",
            api_endpoint=ENDPOINT_STAGING,
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
            task_name="object-detection",
            plot_model=False,
        )
        model_info = HfApi(endpoint=ENDPOINT_STAGING).model_info(
            f"{USER}/{REPO_NAME}",
        )
        self.assertFalse("model.png" in [f.rfilename for f in model_info.siblings])
        self._api.delete_repo(name=f"{REPO_NAME}", token=self._token)

    @retry_endpoint
    def test_push_to_hub_tensorboard(self):
        REPO_NAME = "PUSH_TO_HUB_TB"
        with tempfile.TemporaryDirectory() as tmpdirname:
            os.makedirs(f"{tmpdirname}/log_dir")
            with open(f"{tmpdirname}/log_dir/tensorboard.txt", "w") as fp:
                fp.write("Keras FTW")
            model = self.model_init()
            model = self.model_fit(model, num_epochs=1)
            push_to_hub_keras(
                model,
                repo_path_or_name=f"{WORKING_REPO_DIR}/{REPO_NAME}",
                log_dir=f"{tmpdirname}/log_dir",
                api_endpoint=ENDPOINT_STAGING,
                use_auth_token=self._token,
                git_user="ci",
                git_email="ci@dummy.com",
            )
        model_info = HfApi(endpoint=ENDPOINT_STAGING).model_info(
            f"{USER}/{REPO_NAME}",
        )

        self.assertTrue(
            "logs/tensorboard.txt" in [f.rfilename for f in model_info.siblings]
        )
        self._api.delete_repo(name=f"{REPO_NAME}", token=self._token)

    @retry_endpoint
    def test_override_tensorboard(self):
        REPO_NAME = repo_name("TB_OVERRIDE")
        with tempfile.TemporaryDirectory() as tmpdirname:
            os.makedirs(f"{tmpdirname}/tb_log_dir")
            with open(f"{tmpdirname}/tb_log_dir/tensorboard.txt", "w") as fp:
                fp.write("Keras FTW")
            model = self.model_init()
            model.build((None, 2))
            push_to_hub_keras(
                model,
                repo_path_or_name=f"{WORKING_REPO_DIR}/{REPO_NAME}",
                log_dir=f"{tmpdirname}/tb_log_dir",
                api_endpoint=ENDPOINT_STAGING,
                use_auth_token=self._token,
                git_user="ci",
                git_email="ci@dummy.com",
            )
            os.makedirs(f"{tmpdirname}/tb_log_dir2")
            with open(f"{tmpdirname}/tb_log_dir2/override.txt", "w") as fp:
                fp.write("Keras FTW")
            push_to_hub_keras(
                model,
                repo_path_or_name=f"{WORKING_REPO_DIR}/{REPO_NAME}",
                log_dir=f"{tmpdirname}/tb_log_dir2",
                api_endpoint=ENDPOINT_STAGING,
                use_auth_token=self._token,
                git_user="ci",
                git_email="ci@dummy.com",
            )

        model_info = HfApi(endpoint=ENDPOINT_STAGING).model_info(
            f"{USER}/{REPO_NAME}",
        )
        self.assertTrue(
            "logs/override.txt" in [f.rfilename for f in model_info.siblings]
        )
        self.assertFalse(
            "logs/tensorboard.txt" in [f.rfilename for f in model_info.siblings]
        )

        self._api.delete_repo(repo_id=f"{REPO_NAME}", token=self._token)

    @retry_endpoint
    def test_push_to_hub_model_kwargs(self):
        REPO_NAME = repo_name("PUSH_TO_HUB")
        model = self.model_init()
        model = self.model_fit(model, num_epochs=2)
        push_to_hub_keras(
            model,
            repo_path_or_name=f"{WORKING_REPO_DIR}/{REPO_NAME}",
            api_endpoint=ENDPOINT_STAGING,
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
            config={"num": 7, "act": "gelu_fast"},
            include_optimizer=True,
            save_traces=False,
        )

        model_info = HfApi(endpoint=ENDPOINT_STAGING).model_info(
            f"{USER}/{REPO_NAME}",
        )
        self.assertEqual(model_info.modelId, f"{USER}/{REPO_NAME}")

        from_pretrained_keras(f"{WORKING_REPO_DIR}/{REPO_NAME}")
        self.assertRaises(ValueError, msg="Exception encountered when calling layer*")

        self._api.delete_repo(repo_id=f"{REPO_NAME}", token=self._token)


@require_tf
class HubKerasFunctionalTest(HubKerasSequentialTest):
    def model_init(self):
        inputs = tf.keras.layers.Input(shape=(None, 2))
        outputs = tf.keras.layers.Dense(2, activation="relu")(inputs)
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="mse")
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
        self.assertEqual(len(files), 6)

    def test_save_pretrained_fit(self):
        REPO_NAME = repo_name("functional")
        model = self.model_init()
        model = self.model_fit(model, num_epochs=2)

        save_pretrained_keras(model, f"{WORKING_REPO_DIR}/{REPO_NAME}")
        files = os.listdir(f"{WORKING_REPO_DIR}/{REPO_NAME}")

        self.assertIn("saved_model.pb", files)
        self.assertIn("keras_metadata.pb", files)
        self.assertEqual(len(files), 6)
