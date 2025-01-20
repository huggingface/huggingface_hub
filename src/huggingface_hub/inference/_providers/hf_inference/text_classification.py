from ._common import BaseInferenceTask


class TextClassification(BaseInferenceTask):
    TASK_NAME = "text-classification"


build_url = TextClassification.build_url
map_model = TextClassification.map_model
prepare_headers = TextClassification.prepare_headers
prepare_payload = TextClassification.prepare_payload
get_response = TextClassification.get_response
