from ._common import BaseConversationalTask, BaseTextGenerationTask


_PROVIDER = "featherless-ai"
_BASE_URL = "https://api.featherless.ai"


class FeatherlessTextGenerationTask(BaseTextGenerationTask):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL)
    
    def get_response(self, response: Union[bytes, Dict], request_params: Optional[RequestParameters] = None) -> Any:
        output = _as_dict(response)["choices"][0]
        return {
            "generated_text": output["text"],
            "details": {
                "finish_reason": output.get("finish_reason"),
                "seed": output.get("seed"),
            },
        }


class FeatherlessConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL)
