from typing import Any, Dict, List

import numpy as np
from app.pipelines import Pipeline
from transformers import (
    TokenClassificationPipeline as TransformersTokenClassificationPipeline,
)


class TokenClassificationPipeline(Pipeline):
    def __init__(
        self,
        adapter_id: str,
    ):
        self.pipeline = self._load_pipeline_instance(
            TransformersTokenClassificationPipeline, adapter_id
        )

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
        outputs = self.pipeline(inputs)
        # convert all numpy types to plain Python floats
        for output in outputs:
            # remove & rename keys
            output.pop("index")
            entity = output.pop("entity")
            for k, v in output.items():
                if isinstance(v, np.generic):
                    output[k] = v.item()
            output["entity_group"] = entity
        return outputs
