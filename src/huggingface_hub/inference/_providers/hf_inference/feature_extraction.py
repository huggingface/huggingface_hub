from ._common import BaseInferenceTask


class FeatureExtraction(BaseInferenceTask):
    TASK_NAME = "feature-extraction"


build_url = FeatureExtraction.build_url
map_model = FeatureExtraction.map_model
prepare_headers = FeatureExtraction.prepare_headers
prepare_payload = FeatureExtraction.prepare_payload
get_response = FeatureExtraction.get_response
