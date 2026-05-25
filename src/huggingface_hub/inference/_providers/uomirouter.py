"""UomiRouter inference provider.

UomiRouter (https://uomi.ai) is a distributed inference network: each request is
dispatched across a pool of operator-run GPU nodes and the response carries an
``Inference-Id`` header for verifiable response attestation (off-chain today; on-chain anchoring on UOMI L1 is the next milestone). The gateway exposes
an OpenAI-compatible surface at ``/v1/chat/completions`` (streaming, tool calling
and structured output included), so the helper here is intentionally minimal —
``BaseConversationalTask`` already covers the request/response shape.

See the registered mapping of HF model ID => UomiRouter model ID here:

    https://huggingface.co/api/partners/uomirouter/models

- If you work at UomiRouter and want to update this mapping, please use the
  model-mapping API we provide on huggingface.co.
- If you're a community member and want to add a new supported HF model to
  UomiRouter, please open an issue on huggingface/huggingface_hub and tag the
  UomiRouter team.
"""

from ._common import BaseConversationalTask


class UomiRouterConversationalTask(BaseConversationalTask):
    """Conversational (chat completion) task helper for UomiRouter.

    The UomiRouter gateway is OpenAI-compatible, so no payload or response
    overrides are needed — the base class default of POSTing the OpenAI-format
    payload to ``/v1/chat/completions`` is exactly what the gateway expects.
    """

    def __init__(self):
        super().__init__(provider="uomirouter", base_url="https://gateway.uomi.ai")
