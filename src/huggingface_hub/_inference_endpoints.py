from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Optional

from .inference._client import InferenceClient
from .inference._generated._async_client import AsyncInferenceClient
from .utils import parse_datetime


class InferenceEndpointStatus(str, Enum):
    PENDING = "pending"
    INITIALIZING = "initializing"
    UPDATING = "updating"
    UPDATE_FAILED = "updateFailed"
    RUNNING = "running"
    PAUSED = "paused"
    FAILED = "failed"
    SCALED_TO_ZERO = "scaledToZero"


class InferenceEndpointType(str, Enum):
    PUBlIC = "public"
    PROTECTED = "protected"
    PRIVATE = "private"


@dataclass
class InferenceEndpoint:
    """
    Contains information about a deployed Inference Endpoint.

    Example:
    ```python
    >>> from huggingface_hub import HfApi
    >>> api = HfApi()
    >>> endpoint = api.get_inference_endpoint("my-text-to-image")
    >>> endpoint
    InferenceEndpoint(name='my-text-to-image', ...)

    # Get status
    >>> endpoint.status
    'running'
    >>> endpoint.url
    'https://my-text-to-image.region.vendor.endpoints.huggingface.cloud'

    # Run inference
    >>> endpoint.client.text_to_image(...)
    """

    # Field in __repr__
    name: str
    namespace: str
    model_repository: str
    status: InferenceEndpointStatus
    url: str

    # Other fields
    model_framework: str = field(repr=False)
    model_revision: str = field(repr=False)
    model_task: str = field(repr=False)
    created_at: datetime = field(repr=False)
    updated_at: datetime = field(repr=False)
    type: InferenceEndpointType = field(repr=False)

    # Raw dict from the API
    raw: Dict

    # Token to authenticate with the endpoint
    token: Optional[str] = field(repr=False, default=None, compare=False)

    @classmethod
    def from_raw(cls, raw: Dict, namespace: str, token: Optional[str] = None) -> "InferenceEndpoint":
        """Initialize object from raw dictionary."""
        return cls(
            # Repr fields
            name=raw["name"],
            namespace=namespace,
            model_repository=raw["model"]["repository"],
            status=raw["status"]["state"],
            url=raw["status"]["url"],
            # Other fields
            model_framework=raw["model"]["framework"],
            model_revision=raw["model"]["revision"],
            model_task=raw["model"]["task"],
            created_at=parse_datetime(raw["status"]["createdAt"]),
            updated_at=parse_datetime(raw["status"]["updatedAt"]),
            type=raw["type"],
            # Raw payload
            raw=raw,
            # Optional token
            token=token,
        )

    @property
    def client(self) -> InferenceClient:
        """Returns a client to make predictions on this endpoint."""
        return InferenceClient(model=self.url, token=self.token)

    @property
    def async_client(self) -> AsyncInferenceClient:
        """Returns a client to make predictions on this endpoint."""
        return AsyncInferenceClient(model=self.url, token=self.token)
