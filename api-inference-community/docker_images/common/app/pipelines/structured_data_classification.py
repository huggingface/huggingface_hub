from typing import Dict, List, Union

import joblib
import numpy as np
import pandas as pd
from app.pipelines import Pipeline
from huggingface_hub import cached_download, hf_hub_url


class StructuredDataClassificationPipeline(Pipeline):
    def __init__(self, model_id: str):
        # IMPLEMENT_THIS
        # Preload all the elements you are going to need at inference.
        # For instance your model, processors, tokenizer that might be needed.
        # This function is only called once, so do all the heavy processing I/O here
        raise NotImplementedError(
            "Please implement StructuredDataClassificationPipeline __init__ function"
        )

    def __call__(
        self, inputs: Dict[str, Union[List[str], List[List[Union[str, float]]]]]
    ) -> List[Union[str, float]]:
        """
        Args:
            inputs (:obj:`dict`):
                a dictionary containing one or two keys, 'data' mapping
                to a list of lists representing each row, and, *optionally*,
                column_names, containing the column name corresponding to
                each row.
        Return:
            A :obj:`list` of floats or strings: The classification output for each row.
        """
        # IMPLEMENT_THIS
        raise NotImplementedError(
            "Please implement StructuredDataClassificationPipeline __init__ function"
        )
