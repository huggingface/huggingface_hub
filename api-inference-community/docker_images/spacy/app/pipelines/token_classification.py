import os
import subprocess
import sys
from typing import Any, Dict, List

from app.pipelines import Pipeline


ALLOWLIST: List[str] = ["spacy"]


class TokenClassificationPipeline(Pipeline):
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
        if namespace not in ALLOWLIST:
            raise ValueError(
                f"Invalid namespace {namespace}. It should be in user/organization allowlist"
            )

        package = f"https://huggingface.co/{namespace}/{model_name}/resolve/main/{model_name}-any-py3-none-any.whl"
        cache_dir = os.environ["PIP_CACHE"]
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--cache-dir", cache_dir, package]
        )

        import spacy

        self.model = spacy.load(model_name)

    def __call__(self, inputs: str) -> List[Dict[str, Any]]:
        """
        Args:
            inputs (:obj:`str`):
                a string containing some text
        Return:
            A :obj:`list`:. The object returned should be like [{"entity_group": "XXX", "word": "some word", "start": 3, "end": 6, "score": 0.82}] containing :
                - "entity_group": A string representing what the entity is.
                - "word": A rubstring of the original string that was detected as an entity.
                - "start": the offset within `input` leading to `answer`. context[start:stop] == word
                - "end": the ending offset within `input` leading to `answer`. context[start:stop] === word
                - "score": A score between 0 and 1 describing how confident the model is for this entity.
        """
        doc = self.model(inputs)

        entities = []
        for ent in doc.ents:
            # Score is currently not well supported, see
            # https://github.com/explosion/spaCy/issues/5917.
            current_entity = {
                "entity_group": ent.label_,
                "word": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "score": 1.0,
            }
            entities.append(current_entity)

        return entities
