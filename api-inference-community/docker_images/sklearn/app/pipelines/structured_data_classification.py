from typing import Dict, List, Union

import joblib
import numpy as np
import pandas as pd
from app.pipelines import Pipeline
from huggingface_hub import cached_download, hf_hub_url


DEFAULT_FILENAME = "sklearn_model.joblib"


class StructuredDataClassificationPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model = joblib.load(
            cached_download(hf_hub_url(model_id, DEFAULT_FILENAME))
        )

    # Dict[str, Union[List[str], List[List[Union[str, float]]]]]
    def __call__(
        self, inputs: Dict[str, Union[List[str], List[List[Union[str, float]]]]]
    ) -> List[Union[str, float]]:
        """
        Args:
            inputs (:obj:`dict`):
                a dictionary containing one or two keys, 'data' mapping
                to a list of lists representing each row, and, optionally,
                column_names, containing the column name corresponding to
                each row.
        Return:
            A :obj:`list` of floats or strings: The classification output for each row.
        """
        return self.model.predict(inputs["data"]).tolist()
