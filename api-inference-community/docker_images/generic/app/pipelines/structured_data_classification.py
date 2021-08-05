from typing import Dict, List, Union

from app.pipelines import Pipeline


class StructuredDataClassificationPipeline(Pipeline):
    def __init__(self, model_id: str):
        # IMPLEMENT_THIS
        # Preload all the elements you are going to need at inference.
        # For instance your model, processors, tokenizer that might be needed.
        # This function is only called once, so do all the heavy processing I/O here
        super().__init__(model_id)

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
        return super().__call__(inputs)
