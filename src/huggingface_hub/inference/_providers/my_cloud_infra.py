from typing import Any, Dict
from .._common import BaseTextGenerationTask

class MyCloudInfraTextGenerationTask(BaseTextGenerationTask):
    def __init__(self, provider: str, base_url: str):
        # 1. 严格按照基类要求传参，砍掉错误的 task 参数
        super().__init__(
            provider="my-cloud-infra",
            base_url="https://calibration-gentle-constantly-liked.trycloudflare.com"
        )

    # 2. 完美重写响应解析函数，将 vLLM/OpenAI 的 choices[0].text 转换为 HF 标准的 generated_text
    def get_response(self, response_json: Any) -> Dict[str, Any]:
        if isinstance(response_json, dict) and "choices" in response_json:
            try:
                text = response_json["choices"][0]["text"]
                return {"generated_text": text}
            except (IndexError, KeyError):
                pass
        return {"generated_text": str(response_json)}
