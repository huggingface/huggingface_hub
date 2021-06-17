import json
from typing import TYPE_CHECKING, Any, Dict, List

import numpy
import tensorflow as tf
from app.pipelines import Pipeline
from huggingface_hub import cached_download, hf_hub_url


if TYPE_CHECKING:
    from PIL import Image

MODEL_FILENAME = "tf_model.h5"
CONFIG_FILENAME = "config.json"


class ImageClassificationPipeline(Pipeline):
    def __init__(self, model_id: str):
        # This loading has strong assumptions and should be replaced with a better
        # wrapper for this. This currently assumed the full model is saved, not just
        # the weights.
        model_file = cached_download(hf_hub_url(model_id, filename=MODEL_FILENAME))
        self.model = tf.keras.models.load_model(model_file)

        # Handle binary classification with single output unit
        self.single_output_unit = self.model.output_shape[1] == 1

        # Config is required to know the mapping to label.
        config_file = cached_download(hf_hub_url(model_id, filename=CONFIG_FILENAME))
        with open(config_file) as config:
            config = json.load(config)
        self.id2label = config["id2label"]

    def __call__(self, inputs: "Image.Image") -> List[Dict[str, Any]]:
        """
        Args:
            inputs (:obj:`PIL.Image`):
                The raw image representation as PIL.
                No transformation made whatsoever from the input. Make all necessary transformations here.
        Return:
            A :obj:`list`:. The list contains items that are dicts should be liked {"label": "XXX", "score": 0.82}
                It is preferred if the returned list is in decreasing `score` order
        """
        # Resize image to expected size
        expected_input_size = self.model.input_shape
        if expected_input_size[-1] == 1:  # Single channel, we assume grayscale
            inputs = inputs.convert("L")
        target_size = (expected_input_size[1], expected_input_size[2])
        img = inputs.resize(target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = self.model.predict(img_array)

        if self.single_output_unit:
            score = predictions[0][0]
            labels = [
                {"label": self.id2label["1"], "score": str(score)},
                {"label": self.id2label["0"], "score": str(1 - score)},
            ]
            print("Return unique label", labels)
            return labels
        else:
            labels = [
                {"label": self.id2label[str(i)], "score": score.item()}
                for i, score in enumerate(predictions[0])
            ]
            return labels
