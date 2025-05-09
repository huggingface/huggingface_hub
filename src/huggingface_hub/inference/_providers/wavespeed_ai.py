import base64
import time
from abc import ABC
from typing import Any, Dict, Optional, Union

from huggingface_hub.hf_api import InferenceProviderMapping
from huggingface_hub.inference._common import RequestParameters, _as_dict
from huggingface_hub.inference._providers._common import TaskProviderHelper, filter_none
from huggingface_hub.utils import get_session, hf_raise_for_status
from huggingface_hub.utils.logging import get_logger


logger = get_logger(__name__)

# 轮询间隔
_POLLING_INTERVAL = 0.1


class WavespeedAITask(TaskProviderHelper, ABC):
    def __init__(self, task: str):
        super().__init__(provider="wavespeed-ai", base_url="https://api.wavespeed.ai", task=task)

    def _prepare_headers(self, headers: Dict, api_key: str) -> Dict:
        headers = super()._prepare_headers(headers, api_key)
        if not api_key.startswith("hf_"):
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return f"/api/v2/{mapped_model}"
    
    def get_response(
        self,
        response: Union[bytes, Dict],
        request_params: Optional[RequestParameters] = None,
    ) -> Any:
        response_dict = _as_dict(response)
        data = response_dict.get("data", {})
        result_url = data.get("urls", {}).get("get")
        
        if not result_url:
            raise ValueError("No result URL found in the response")
        if request_params is None:
            raise ValueError(
                "A `RequestParameters` object should be provided to get responses with WaveSpeed AI."
            )
        
        logger.info("Processing request, polling for results...")
        
        # 轮询直到任务完成
        while True:
            time.sleep(_POLLING_INTERVAL)
            result_response = get_session().get(result_url, headers=request_params.headers)
            hf_raise_for_status(result_response)
            
            result = result_response.json()
            task_result = result.get("data", {})
            status = task_result.get("status")
            
            if status == "completed":
                # 获取输出URL里的内容
                if not task_result.get("outputs") or len(task_result["outputs"]) == 0:
                    raise ValueError("No output URL in completed response")
                
                output_url = task_result["outputs"][0]
                return get_session().get(output_url).content
            elif status == "failed":
                error_msg = task_result.get("error", "Task failed with no specific error message")
                raise ValueError(f"WaveSpeed AI task failed: {error_msg}")
            elif status in ["processing", "created"]:
                continue
            else:
                raise ValueError(f"Unknown status: {status}")


class WavespeedAITextToImageTask(WavespeedAITask):
    def __init__(self):
        super().__init__("text-to-image")
    
    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: Dict, provider_mapping_info: InferenceProviderMapping
    ) -> Optional[Dict]:
        return {"prompt": inputs, **filter_none(parameters)}


class WavespeedAITextToVideoTask(WavespeedAITask):
    def __init__(self):
        super().__init__("text-to-video")
    
    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: Dict, provider_mapping_info: InferenceProviderMapping
    ) -> Optional[Dict]:
        return {"prompt": inputs, **filter_none(parameters)}


class WavespeedAIImageToImageTask(WavespeedAITask):
    def __init__(self):
        super().__init__("image-to-image")
    
    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: Dict, provider_mapping_info: InferenceProviderMapping
    ) -> Optional[Dict]:
        # 如果输入是URL
        if isinstance(inputs, str) and inputs.startswith(("http://", "https://")):
            image = inputs
        # 如果输入是文件路径
        elif isinstance(inputs, str):
            with open(inputs, "rb") as f:
                file_content = f.read()
            image_b64 = base64.b64encode(file_content).decode("utf-8")
            image = f"data:image/jpeg;base64,{image_b64}"
        # 如果输入是二进制数据
        else:
            image_b64 = base64.b64encode(inputs).decode("utf-8")
            image = f"data:image/jpeg;base64,{image_b64}"
            
        return {"image": image, **filter_none(parameters)}


