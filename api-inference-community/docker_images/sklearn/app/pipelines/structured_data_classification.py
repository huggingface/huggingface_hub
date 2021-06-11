from typing import Dict, List, Union

import cloudpickle
from app.pipelines import Pipeline
from huggingface_hub import cached_download, hf_hub_url


ALLOWLIST: List[str] = ["scikit-learn-examples"]
DEFAULT_FILENAME = "sklearn_model.pickle"


class StructuredDataClassificationPipeline(Pipeline):
    def __init__(self, model_id: str):
        # TODO: Obtain expected column names from repo.
        # TODO: Add to model info if it's sklearn_model.pickle" (default) or some other name.

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

        self.model = cloudpickle.load(
            open(cached_download(hf_hub_url(model_id, DEFAULT_FILENAME)), "rb")
        )

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
        # TODO: If there are expected column names, and the columns
        # are passed, change the input order so it matches the
        # expectation.
        return self.model.predict(inputs["data"]).tolist()
