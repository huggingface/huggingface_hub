# Inference Endpoints

Inference Endpoints offers a secure production solution to easily deploy any `transformers`, `sentence-transformers`, and `diffusers` models from the Hub on a dedicated and autoscaling infrastructure managed by Hugging Face.

An Inference Endpoint is built from a model from the [Hub](https://huggingface.co/models). When an Endpoint is created, the service creates image artifacts that are either built from the model you selected or a custom-provided container image. The image artifacts are completely decoupled from the Hub source repositories to ensure the highest security and reliability levels.

Inference Endpoints support all of the `transformers`, `sentence-transformers` and `diffusers` tasks as well as custom tasks not supported by `transformers` yet like speaker diarization.

In addition, 🤗 Inference Endpoints gives you the option to use a custom container image managed on an external service, for instance, [Docker Hub](https://hub.docker.com/), [AWS ECR](https://aws.amazon.com/fr/ecr/?nc1=h_ls), [Azure ACR](https://azure.microsoft.com/de-de/products/container-registry/), or [Google GCR](https://cloud.google.com/artifact-registry?hl=de).

<Tip>

This page is only a reference for `huggingface_hub`'s integration with Inference Endpoints. For more information about the Inference Endpoints product, check out its [official documentation](https://huggingface.co/docs/inference-endpoints/index).

</Tip>

Inference Endpoints can be fully managed via API. The endpoints are documented with [Swagger](https://api.endpoints.huggingface.cloud/).

## Methods

A subset of the Inference Endpoint features are implemented in [`HfApi`]:

- [`get_inference_endpoint`] and [`list_inference_endpoints`] to get information about your Inference Endpoints
- [`create_inference_endpoint`], [`update_inference_endpoint`] and [`delete_inference_endpoint`] to deploy and manage Inference Endpoints
- [`pause_inference_endpoint`] and [`resume_inference_endpoint`] to pause and resume an Inference Endpoint
- [`scale_to_zero_inference_endpoint`] to scale an Endpoint to 0 replicas

## InferenceEndpoint

The main dataclass is [`InferenceEndpoint`]. It contains information about a deployed `InferenceEndpoint`, including its configuration and current state. Once deployed, you can run inference on the Endpoint using the  [`InferenceEndpoint.client`] and [`InferenceEndpoint.async_client`] properties that respectively return an [`InferenceClient`] and an [`AsyncInferenceClient`] object.

[[autodoc]] InferenceEndpoint

## InferenceEndpointStatus

[[autodoc]] InferenceEndpointStatus

## InferenceEndpointType

[[autodoc]] InferenceEndpointType

## InferenceEndpointException

[[autodoc]] InferenceEndpointException