from ._common import BaseInferenceTask


class ZeroShotClassification(BaseInferenceTask):
    TASK_NAME = "zero-shot-classification"


build_url = ZeroShotClassification.build_url
map_model = ZeroShotClassification.map_model
prepare_headers = ZeroShotClassification.prepare_headers
prepare_payload = ZeroShotClassification.prepare_payload
get_response = ZeroShotClassification.get_response
