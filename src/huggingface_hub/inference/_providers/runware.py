import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

from huggingface_hub.inference._common import (
    RequestParameters,
    TaskProviderHelper,
    _as_dict,
)
from huggingface_hub.utils import (
    build_hf_headers,
    get_session,
    get_token,
    logging,
)


logger = logging.get_logger(__name__)

RUNWARE_BASE_URL = "https://api.runware.ai/v1"

# Example of mapping HF names to Runware "AIR" IDs
RUNWARE_SUPPORTED_MODELS = {
    "text-to-image": {
        "black-forest-labs/FLUX.1-schnell": "runware:100@1",
        "black-forest-labs/FLUX.1-dev": "runware:101@1",
        # "black-forest-labs/flux.1-fill-dev": "runware:102@1",
        # Add more if needed...
    },
}


class RunwareTask(TaskProviderHelper, ABC):
    """
    Use prepare_response and get_response to run inference via Runware's REST API.
    """

    def __init__(self, task: str):
        self.task = task

    def prepare_request(
        self,
        *,
        inputs: Any,
        parameters: Dict[str, Any],
        headers: Dict,
        model: Optional[str],
        api_key: Optional[str],
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> RequestParameters:
        """
        We return a RequestParameters object that can then execute via requests.post or get_session().post
        """
        # If user didn’t provide an API key, try from HF get_token()
        if api_key is None:
            api_key = get_token()
        if api_key is None:
            raise ValueError("No API key found. Please provide one for Runware.")

        # Merge any caller-supplied headers with HF's standard headers
        final_headers = {
            **build_hf_headers(token=api_key),
            **headers,
        }
        # Overwrite authorization to Bearer
        final_headers["Authorization"] = f"Bearer {api_key}"

        payload = self._prepare_payload(inputs, parameters=parameters, model=model)

        # Log for debugging, if enabled.
        logger.info(f"Calling Runware for task '{self.task}', model='{model}'")

        # Build the usual huggingface_hub “RequestParameters”
        return RequestParameters(
            url=RUNWARE_BASE_URL,
            task=self.task,
            model=model if model else "N/A",
            json=payload,
            data=None,
            headers=final_headers,
        )

    @abstractmethod
    def _prepare_payload(self, inputs: Any, parameters: Dict[str, Any], model: Optional[str]) -> Any:
        """
        You must return an object that Runware expects as JSON.
        (Typically a list of dicts, because Runware wants an array.)
        """
        ...

    @abstractmethod
    def get_response(self, response: Union[bytes, Dict]) -> Any:
        """
        Parse the raw HTTP result and return the final data (e.g. text or bytes).
        """
        ...


class RunwareTextToImageTask(RunwareTask):
    """
    Calls Runware’s imageInference endpoint via REST API for text2image tasks.
    """

    def __init__(self):
        super().__init__("text-to-image")

    def _map_model(self, model: Optional[str]) -> str:
        if model is None:
            raise ValueError("Please provide a model for Runware.")
        if self.task not in RUNWARE_SUPPORTED_MODELS:
            raise ValueError(f"Task '{self.task}' not found in RUNWARE_SUPPORTED_MODELS.")
        mapped = RUNWARE_SUPPORTED_MODELS[self.task].get(model)
        if mapped is None:
            raise ValueError(f"Model '{model}' is not supported by Runware for {self.task}.")
        return mapped

    def _prepare_payload(self, inputs: Any, parameters: Dict[str, Any], model: Optional[str]) -> list:
        """
        Construct the list of dicts Runware’s API expects.
        E.g. [
          {
            "taskType": "imageInference",
            "taskUUID": "...",
            "positivePrompt": "...",
            "model": "...",
            "width": 512, ...
          }
        ]
        """
        mapped_model = self._map_model(model)
        task_uuid = str(uuid.uuid4())

        width = parameters.pop("width", None)
        height = parameters.pop("height", None)
        num_images = parameters.pop("num_images", 1)  # Example

        body = {
            "taskType": "imageInference",
            "taskUUID": task_uuid,
            "positivePrompt": str(inputs),
            "model": mapped_model,
            "numberResults": num_images,
        }

        if width:
            body["width"] = width
        if height:
            body["height"] = height

        # Runware's REST API will effectively ignore negative prompts instead of erroring.
        body["negativePrompt"] = parameters.get("negative_prompt", "")
        body["CFGScale"] = parameters.get("guidance_scale", 7.5)

        # Optionally pass additional stuff from parameters
        # e.g. steps, CFGScale, etc. if you want
        # body.update(parameters)

        return [body]

    def get_response(self, response: Union[bytes, Dict]) -> Any:
        """
        Parse the JSON from Runware. Return the raw bytes of the first image.
        """
        response_dict = _as_dict(response)
        if "error" in response_dict:
            raise ValueError(f"Runware Error: {response_dict['error']}")

        data_list = response_dict.get("data", [])
        if not data_list:
            raise ValueError("No 'data' found in Runware response.")

        # We'll just grab the first item
        first_item = data_list[0]
        image_url = first_item.get("imageURL")
        if not image_url:
            raise ValueError("No 'imageURL' field found in Runware response.")

        # Use the same shared session that HF uses
        sess = get_session()
        img_bytes = sess.get(image_url).content
        return img_bytes


# ------------------------------------------------------------------------
# Example usage (manually calling prepare_request and get_response)
# ------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Instantiate our custom task
    t2i_task = RunwareTextToImageTask()

    # 2. Gather inputs/parameters
    prompt = "A majestic cat wearing a wizard's hat"
    params = {
        "width": 512,
        "height": 512,
        # "negative_prompt": "ugly, deformed, poorly drawn", # if needed
    }

    # 3. Prepare the request
    import os

    request_params = t2i_task.prepare_request(
        inputs=prompt,
        parameters=params,
        headers={},  # Additional headers if needed
        model="black-forest-labs/FLUX.1-schnell",  # The HF name
        api_key=os.environ.get("RUNWARE_API_KEY"),  # Replace with real token
    )

    # 4. Actually perform the HTTP request
    response = get_session().post(
        request_params.url,
        headers=request_params.headers,
        json=request_params.json,
    )
    if response.status_code != 200:
        raise ValueError(f"HTTP Error: {response.status_code} => {response.text}")

    # 5. Parse final result with get_response
    image_data = t2i_task.get_response(response.json())

    # 6. Save or do whatever with the image bytes
    with open("runware_result.jpg", "wb") as f:
        f.write(image_data)

    print("Saved 'runware_result.jpg'")
