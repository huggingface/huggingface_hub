from ._common import BaseInferenceTask


class TabularRegression(BaseInferenceTask):
    TASK_NAME = "tabular-regression"


build_url = TabularRegression.build_url
map_model = TabularRegression.map_model
prepare_headers = TabularRegression.prepare_headers
prepare_payload = TabularRegression.prepare_payload
get_response = TabularRegression.get_response
