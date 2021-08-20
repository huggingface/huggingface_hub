import os
import subprocess
import sys
from typing import Dict, List

from app.pipelines import Pipeline


class TextClassificationPipeline(Pipeline):
    def __init__(
        self,
        model_id: str,
    ):
        # At the time, only public models from spaCy are allowed in the inference API.
        full_model_path = model_id.split("/")
        if len(full_model_path) != 2:
            raise ValueError(
                f"Invalid model_id: {model_id}. It should have a namespace (:namespace:/:model_name:)"
            )
        namespace, model_name = full_model_path
        package = f"https://huggingface.co/{namespace}/{model_name}/resolve/main/{model_name}-any-py3-none-any.whl"
        cache_dir = os.environ["PIP_CACHE"]
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--cache-dir", cache_dir, package]
        )

        import spacy

        self.model = spacy.load(model_name)

    def __call__(self, inputs: str) -> List[List[Dict[str, float]]]:
        """
        Args:
            inputs (:obj:`str`):
                a string containing some text
        Return:
            A :obj:`list`:. The object returned should be a list of one list like [[{"label": 0.9939950108528137}]] containing :
                - "label": A string representing what the label/class is. There can be multiple labels.
                - "score": A score between 0 and 1 describing how confident the model is for this label/class.
        """
        doc = self.model(inputs)

        categories = []
        for cat, score in doc.cats.items():
            categories.append({"label": cat, "score": score})

        return [categories]
