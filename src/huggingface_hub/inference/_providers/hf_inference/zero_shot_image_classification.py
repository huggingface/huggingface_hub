from ._common import BaseInferenceTask


class ZeroShotImageClassification(BaseInferenceTask):
    TASK_NAME = "zero-shot-image-classification"


build_url = ZeroShotImageClassification.build_url
map_model = ZeroShotImageClassification.map_model
prepare_headers = ZeroShotImageClassification.prepare_headers
prepare_payload = ZeroShotImageClassification.prepare_payload
get_response = ZeroShotImageClassification.get_response
