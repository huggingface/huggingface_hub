from ._common import BaseInferenceTask


class SentenceSimilarity(BaseInferenceTask):
    TASK_NAME = "sentence-similarity"


build_url = SentenceSimilarity.build_url
map_model = SentenceSimilarity.map_model
prepare_headers = SentenceSimilarity.prepare_headers
prepare_payload = SentenceSimilarity.prepare_payload
get_response = SentenceSimilarity.get_response
