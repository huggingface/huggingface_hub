# Inference Endpoints

Inference Endpoints fournit une solution de production sécurisée pour déployer facilement des modèles sur une infrastructure dédiée et autoscaling gérée par Hugging Face. Un Inference Endpoint est construit à partir d'un modèle du [Hub](https://huggingface.co/models). Cette page est une référence pour l'intégration de `huggingface_hub` avec Inference Endpoints. Pour plus d'informations sur le produit Inference Endpoints, consultez sa [documentation officielle](https://huggingface.co/docs/inference-endpoints/index).

> [!TIP]
> Consultez le [guide associé](../guides/inference_endpoints) pour apprendre comment utiliser `huggingface_hub` pour gérer vos Inference Endpoints par programme.

Les Inference Endpoints peuvent être entièrement gérés via API. Les endpoints sont documentés avec [Swagger](https://api.endpoints.huggingface.cloud/). La classe [`InferenceEndpoint`] est un wrapper simple construit au-dessus de cette API.

## Méthodes

Un sous-ensemble des fonctionnalités Inference Endpoint sont implémentées dans [`HfApi`] :

- [`get_inference_endpoint`] et [`list_inference_endpoints`] pour obtenir des informations sur vos Inference Endpoints
- [`create_inference_endpoint`], [`update_inference_endpoint`] et [`delete_inference_endpoint`] pour déployer et gérer les Inference Endpoints
- [`pause_inference_endpoint`] et [`resume_inference_endpoint`] pour mettre en pause et reprendre un Inference Endpoint
- [`scale_to_zero_inference_endpoint`] pour faire évoluer manuellement un Endpoint vers 0 replicas

## InferenceEndpoint

La dataclass principale est [`InferenceEndpoint`]. Elle contient des informations sur un `InferenceEndpoint` déployé, incluant sa configuration et son état actuel. Une fois déployé, vous pouvez exécuter l'inférence sur l'Endpoint en utilisant les propriétés [`InferenceEndpoint.client`] et [`InferenceEndpoint.async_client`] qui retournent respectivement un objet [`InferenceClient`] et un [`AsyncInferenceClient`].

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
