import os
import subprocess
import sys
from typing import Dict, List, Union

from app.pipelines import Pipeline


class SentenceSimilarityPipeline(Pipeline):
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

    def __call__(self, inputs: Dict[str, Union[str, List[str]]]) -> List[float]:
        """
        Args:
            inputs (:obj:`dict`):
                a dictionary containing two keys, 'source_sentence' mapping
                to the sentence that will be compared against all the others,
                and 'sentences', mapping to a list of strings to which the
                source will be compared.
        Return:
            A :obj:`list` of floats: Some similarity measure between `source_sentence` and each sentence from `sentences`.
        """
        source_sentence = inputs['source_sentence']
        source_doc = self.model(source_sentence)

        similarities = []
        for sentence in inputs['sentences']:
            search_doc = self.model(sentence)
            similarities.append(source_doc.similarity(search_doc))

        return similarities    