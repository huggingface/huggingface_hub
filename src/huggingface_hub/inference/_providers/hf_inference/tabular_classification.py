from ._common import BaseInferenceTask


class TabularClassification(BaseInferenceTask):
    TASK_NAME = "tabular-classification"


build_url = TabularClassification.build_url
map_model = TabularClassification.map_model
prepare_headers = TabularClassification.prepare_headers
prepare_payload = TabularClassification.prepare_payload
get_response = TabularClassification.get_response
