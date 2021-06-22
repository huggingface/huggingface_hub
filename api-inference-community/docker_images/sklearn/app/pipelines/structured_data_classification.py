from typing import Dict, List, Union

import joblib
from app.pipelines import Pipeline
from huggingface_hub import cached_download, hf_hub_url


ALLOWLIST: List[str] = ["scikit-learn-examples", "julien-c"]
DEFAULT_FILENAME = "sklearn_model.joblib"


class StructuredDataClassificationPipeline(Pipeline):
    def __init__(self, model_id: str):
        # Check if there are class definitions

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
    
        print("Loading")
        self.model = joblib.load(
            cached_download(hf_hub_url(model_id, DEFAULT_FILENAME)))
        print("This is the model")
        print(self.model)
        print("Classes:", self.model.columns_)
        print("Nice")

    def __call__(self, inputs: Dict[str, List[str]]) -> List[Union[str, float]]:
        """
        Args:
            inputs (:obj:`dict`):
                a dictionary containing a key 'data' mapping to a list representing
                a column.
        Return:
            A :obj:`list` of floats or strings: The classification output for each row.
        """
        print("Inputs")
        print(inputs)
        return self.model.predict(inputs["data"]).tolist()
