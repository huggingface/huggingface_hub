## How to add a new provider?

Before adding a new provider to the `huggingface_hub` library, make sure it has already been added to `huggingface.js` and is working on the Hub. Support in the Python library comes as a second step. In this guide, we are considering that the first part is complete. 

### 1. Implement the provider helper 

Create a new file under `src/huggingface_hub/inference/_providers/{provider_name}.py` and copy-paste the following snippet.

Implement the methods that require custom handling. Check out the base implementation to check default behavior. If you don't need to override a method, just remove it. At least one of `_prepare_payload_as_dict` or `_prepare_payload_as_bytes` must be overwritten.

If the provider supports multiple tasks that require different implementations, create dedicated subclasses for each task, following the pattern shown in `fal_ai.py`.

For `text-generation` and `conversational` tasks, one can just inherit from `BaseTextGenerationTask` and `BaseConversationalTask` respectively (defined in `_common.py`) and override the methods if needed. Examples can be found in `fireworks_ai.py` and `together.py`.

```py
from typing import Any, Optional, Union

from ._common import TaskProviderHelper, MimeBytes


class MyNewProviderTaskProviderHelper(TaskProviderHelper):
    def __init__(self):
        """Define high-level parameters."""
        super().__init__(provider=..., base_url=..., task=...)

    def get_response(
        self,
        response: Union[bytes, dict],
        request_params: Optional[RequestParameters] = None,
    ) -> Any:
        """
        Return the response in the expected format.

        Override this method in subclasses for customized response handling."""
        return super().get_response(response)

    def _prepare_headers(self, headers: dict, api_key: str) -> dict[str, Any]:
        """Return the headers to use for the request.

        Override this method in subclasses for customized headers.
        """
        return super()._prepare_headers(headers, api_key)

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        """Return the route to use for the request.

        Override this method in subclasses for customized routes.
        """
        return super()._prepare_route(mapped_model)

    def _prepare_payload_as_dict(self, inputs: Any, parameters: dict, mapped_model: str) -> Optional[dict]:
        """Return the payload to use for the request, as a dict.

        Override this method in subclasses for customized payloads.
        Only one of `_prepare_payload_as_dict` and `_prepare_payload_as_bytes` should return a value.
        """
        return super()._prepare_payload_as_dict(inputs, parameters, mapped_model)

    def _prepare_payload_as_bytes(
        self, inputs: Any, parameters: dict, mapped_model: str, extra_payload: Optional[dict]
    ) -> Optional[MimeBytes]:
        """Return the body to use for the request, as bytes.

        Override this method in subclasses for customized body data.
        Only one of `_prepare_payload_as_dict` and `_prepare_payload_as_bytes` should return a value.

        `MimeBytes` is a subclass of `bytes` that carries a `mime_type` attribute.
        """
        return super()._prepare_payload_as_bytes(inputs, parameters, mapped_model, extra_payload)
    
```

### 2. Register the provider helper in `__init__.py`

Go to `src/huggingface_hub/inference/_providers/__init__.py` and add your provider  to `PROVIDER_T` and `PROVIDERS`.
Please try to respect alphabetical order.

### 3. Update docstring in `InferenceClient.__init__` to document your provider

### 4. Add static tests in `tests/test_inference_providers.py`

You only have to add a test for overwritten methods.

### 5. Add VCR tests in `tests/test_inference_client.py`

#### a. Add test model mapping
Add an entry to `_RECOMMENDED_MODELS_FOR_VCR` at the top of the test module, It contains a mapping task <> test model. `model-id` must be the HF model id.
```python
_RECOMMENDED_MODELS_FOR_VCR = {
    "your-provider": {
        "task": "model-id",
        ...
    },
    ...
}
```
#### b. Set up authentication
To record VCR cassettes, you'll need authentication:

- If you are a member of the provider organization (e.g., Replicate organization: https://huggingface.co/replicate), you can set the `HF_INFERENCE_TEST_TOKEN` environment variable with your HF token:
   ```bash
   export HF_INFERENCE_TEST_TOKEN="your-hf-token"
   ```

- If you're not a member but the provider is officially released on the Hub, you can set the `HF_INFERENCE_TEST_TOKEN` environment variable as above. If you don't have enough inference credits, we can help you record the VCR cassettes.

#### c. Record and commit tests

1. Run the tests for your provider:
   ```bash
   pytest tests/test_inference_client.py -k <provider>
   ```
2. Commit the generated VCR cassettes with your PR
