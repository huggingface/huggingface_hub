from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Optional

from .inference._client import InferenceClient
from .inference._generated._async_client import AsyncInferenceClient
from .utils import parse_datetime


class InferenceEndpointException(Exception):
    """Generic exception when dealing with Inference Endpoints."""


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

    Args:
        name (`str`):
            The unique name of the Inference Endpoint.
        namespace (`str`):
            The namespace where the endpoint is located.
        repository (`str`):
            The name of the model repository deployed on this endpoint.
        status ([`InferenceEndpointStatus`]):
            The current status of the Inference Endpoint.
        url (`str`, *optional*):
            The URL of the Inference Endpoint, if available. Only a deployed endpoint will have a URL.
        framework (`str`):
            The framework used for the model.
        revision (`str`):
            The specific model revision deployed on the Inference Endpoint.
        task (`str`):
            The task associated with the deployed model.
        created_at (`datetime.datetime`):
            The timestamp when the endpoint was created.
        updated_at (`datetime.datetime`):
            The timestamp of the last update of the endpoint.
        type ([`InferenceEndpointType`]):
            The type of the Inference Endpoint (public, protected, private)
        raw (`Dict`):
            The raw dictionary data returned from the API.
        token (`str`, *optional*):
            Authentication token for the endpoint, if set when requesting the API.

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
    ```
    """

    # Field in __repr__
    name: str
    namespace: str
    repository: str
    status: InferenceEndpointStatus
    url: Optional[str]

    # Other fields
    framework: str = field(repr=False)
    revision: str = field(repr=False)
    task: str = field(repr=False)
    created_at: datetime = field(repr=False)
    updated_at: datetime = field(repr=False)
    type: InferenceEndpointType = field(repr=False)

    # Raw dict from the API
    raw: Dict = field(repr=False)

    # Token to authenticate with the endpoint
    token: Optional[str] = field(repr=False, default=None, compare=False)

    @classmethod
    def from_raw(cls, raw: Dict, namespace: str, token: Optional[str] = None) -> "InferenceEndpoint":
        """Initialize object from raw dictionary."""
        return cls(
            # Repr fields
            name=raw["name"],
            namespace=namespace,
            repository=raw["model"]["repository"],
            status=raw["status"]["state"],
            url=raw["status"].get("url"),
            # Other fields
            framework=raw["model"]["framework"],
            revision=raw["model"]["revision"],
            task=raw["model"]["task"],
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
        """Returns a client to make predictions on this endpoint.

        Raises:
            [`InferenceEndpointException`]: If the endpoint is not yet deployed.
        """
        if self.url is None:
            raise InferenceEndpointException(
                "Cannot create a client for this endpoint as it is not yet deployed. "
                "Please wait for the endpoint to be deployed and try again."
            )
        return InferenceClient(model=self.url, token=self.token)

    @property
    def async_client(self) -> AsyncInferenceClient:
        """Returns a client to make predictions on this endpoint.

        Raises:
            [`InferenceEndpointException`]: If the endpoint is not yet deployed.
        """
        if self.url is None:
            raise InferenceEndpointException(
                "Cannot create a client for this endpoint as it is not yet deployed. "
                "Please wait for the endpoint to be deployed and try again."
            )
        return AsyncInferenceClient(model=self.url, token=self.token)
