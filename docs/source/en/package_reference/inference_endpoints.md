# Inference Endpoints

Inference Endpoints provides a secure production solution to easily deploy models on a dedicated and autoscaling infrastructure managed by Hugging Face. An Inference Endpoint is built from a model from the [Hub](https://huggingface.co/models). This page is a reference for `huggingface_hub`'s integration with Inference Endpoints. For more information about the Inference Endpoints product, check out its [official documentation](https://huggingface.co/docs/inference-endpoints/index).

> [!TIP]
> Check out the [related guide](../guides/inference_endpoints) to learn how to use `huggingface_hub` to manage your Inference Endpoints programmatically.

Inference Endpoints can be fully managed via API. The endpoints are documented with [Swagger](https://api.endpoints.huggingface.cloud/). The [`InferenceEndpoint`] class is a simple wrapper built on top on this API.

## Methods

A subset of the Inference Endpoint features are implemented in [`HfApi`]:

- [`get_inference_endpoint`] and [`list_inference_endpoints`] to get information about your Inference Endpoints
- [`create_inference_endpoint`], [`update_inference_endpoint`] and [`delete_inference_endpoint`] to deploy and manage Inference Endpoints
- [`pause_inference_endpoint`] and [`resume_inference_endpoint`] to pause and resume an Inference Endpoint
- [`scale_to_zero_inference_endpoint`] to manually scale an Endpoint to 0 replicas

## InferenceEndpoint

The main dataclass is [`InferenceEndpoint`]. It contains information about a deployed `InferenceEndpoint`, including its configuration and current state. Once deployed, you can run inference on the Endpoint using the  [`InferenceEndpoint.client`] and [`InferenceEndpoint.async_client`] properties that respectively return an [`InferenceClient`] and an [`AsyncInferenceClient`] object.

[[autodoc]] InferenceEndpoint
  - from_raw
  - client
  - async_client
  - all

## InferenceEndpointStatus

[[autodoc]] InferenceEndpointStatus

## InferenceEndpointType

[[autodoc]] InferenceEndpointType

## InferenceEndpointError

[[autodoc]] InferenceEndpointError
