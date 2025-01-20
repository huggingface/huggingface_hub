from ._common import BaseInferenceTask


class ObjectDetection(BaseInferenceTask):
    TASK_NAME = "object-detection"


build_url = ObjectDetection.build_url
map_model = ObjectDetection.map_model
prepare_headers = ObjectDetection.prepare_headers
prepare_payload = ObjectDetection.prepare_payload
get_response = ObjectDetection.get_response
