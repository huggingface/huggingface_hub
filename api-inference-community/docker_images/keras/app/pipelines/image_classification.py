import json
from typing import Any, Dict, List

import tensorflow as tf
from app.pipelines import Pipeline
from huggingface_hub import cached_download, from_pretrained_keras, hf_hub_url
from PIL import Image


# PIL Interpolation Methods - More info can be found in the link below
# https://pillow.readthedocs.io/en/stable/handbook/concepts.html#filters-comparison-table
_PIL_INTERPOLATION_METHODS = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "hamming": Image.HAMMING,
    "box": Image.BOX,
    "lanczos": Image.LANCZOS,
}

MODEL_FILENAME = "saved_model.pb"
CONFIG_FILENAME = "config.json"


class ImageClassificationPipeline(Pipeline):
    def __init__(self, model_id: str):

        # Reload Keras SavedModel
        self.model = from_pretrained_keras(model_id)

        # Handle binary classification with a single output unit.
        self.single_output_unit = self.model.output_shape[1] == 1
        self.num_labels = 2 if self.single_output_unit else self.model.output_shape[1]

        # Config is required to know the mapping to label.
        config_file = cached_download(hf_hub_url(model_id, filename=CONFIG_FILENAME))
        with open(config_file) as config:
            config = json.load(config)

        self.labels = config.get("labels", None)
        if not self.labels:
            self.labels = [f"LABEL_{i}" for i in range(self.num_labels)]

        self.id2label = {str(i): l for i, l in enumerate(self.labels)}
        self.interpolation = _PIL_INTERPOLATION_METHODS.get(
            config.get("interpolation", "nearest"), None
        )
        self.top_k = 5

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
        img = inputs.resize(target_size, self.interpolation)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = self.model.predict(img_array)

        if self.single_output_unit:
            score = predictions[0][0]
            labels = [
                {"label": str(self.id2label["1"]), "score": float(score)},
                {"label": str(self.id2label["0"]), "score": float(1 - score)},
            ]
        else:
            labels = [
                {"label": str(self.id2label[str(i)]), "score": float(score)}
                for i, score in enumerate(predictions[0])
            ]
        return sorted(labels, key=lambda tup: tup["score"], reverse=True)[: self.top_k]
