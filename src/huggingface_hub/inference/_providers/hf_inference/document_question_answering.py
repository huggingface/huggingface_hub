from ._common import BaseInferenceTask


class DocumentQuestionAnswering(BaseInferenceTask):
    TASK_NAME = "document-question-answering"


build_url = DocumentQuestionAnswering.build_url
map_model = DocumentQuestionAnswering.map_model
prepare_headers = DocumentQuestionAnswering.prepare_headers
prepare_payload = DocumentQuestionAnswering.prepare_payload
get_response = DocumentQuestionAnswering.get_response
