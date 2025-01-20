from ._common import BaseInferenceTask


class ImageClassification(BaseInferenceTask):
    TASK_NAME = "image-classification"


build_url = ImageClassification.build_url
map_model = ImageClassification.map_model
prepare_headers = ImageClassification.prepare_headers
prepare_payload = ImageClassification.prepare_payload
get_response = ImageClassification.get_response
