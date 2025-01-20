from ._common import BaseInferenceTask


class TokenClassification(BaseInferenceTask):
    TASK_NAME = "token-classification"


build_url = TokenClassification.build_url
map_model = TokenClassification.map_model
prepare_headers = TokenClassification.prepare_headers
prepare_payload = TokenClassification.prepare_payload
get_response = TokenClassification.get_response
