from ._common import BaseInferenceTask


class ImageZeroShotClassification(BaseInferenceTask):
    TASK_NAME = "image-zero-shot-classification"


build_url = ImageZeroShotClassification.build_url
map_model = ImageZeroShotClassification.map_model
prepare_headers = ImageZeroShotClassification.prepare_headers
prepare_payload = ImageZeroShotClassification.prepare_payload
get_response = ImageZeroShotClassification.get_response
