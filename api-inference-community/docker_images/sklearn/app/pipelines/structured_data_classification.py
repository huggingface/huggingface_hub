from typing import Dict, List, Union

import joblib
from app.pipelines import Pipeline
from huggingface_hub import cached_download, hf_hub_url


ALLOWLIST: List[str] = ["scikit-learn-examples", "julien-c", "osanseviero"]
DEFAULT_FILENAME = "sklearn_model.joblib"


class StructuredDataClassificationPipeline(Pipeline):
    def __init__(self, model_id: str):
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

        self.model = joblib.load(
            cached_download(hf_hub_url(model_id, DEFAULT_FILENAME))
        )

    def __call__(
        self, inputs: Dict[str, Dict[str, List[Union[str, float]]]]
    ) -> List[Union[str, float]]:
        """
        Args:
            inputs (:obj:`dict`):
                a dictionary containing a key 'data' mapping to a dict in which
                the values represent each column.
        Return:
            A :obj:`list` of floats or strings: The classification output for each row.
        """
        column_values = list(inputs["data"].values())
        rows = list(zip(*column_values))
        return self.model.predict(rows).tolist()
