# Inference Endpoints

Inference Endpoints est une solution permettant de déployer facilement les modèles en production sur une infrastructure gérée par Hugging Face et capable d'autoscaling . Un Inference Endpoint peut être crée sur un modèle depuis le [Hub](https://huggingface.co/models). Cette page est
une référence pour l'intégration d'`huggingface_hub` avec Inference Endpoints. Pour plus d'informations à propos du produit Inference Endpoints, consultez la [documentation officielle](https://huggingface.co/docs/inference-endpoints/index).

<Tip>

Consultez ce [guide](../guides/inference_endpoints) pour apprendre à utiliser `huggingface_hub` pour gérer votre Inference Enpoints depuis le code.

</Tip>

Inference Endpoints peut être entièrement géré depuis une API. Les endpoints sont consultables via [Swagger](https://api.endpoints.huggingface.cloud/).
La classe [`InferenceEndpoint`] est un simple wrapper autour de cette API.

## Méthodes

Un sous ensemble des fonctionnalités de l'Inference Endpoint sont implémentées dans [`HfApi`]: 

- [`get_inference_endpoint`] et [`list_inference_endpoints`] pour obtenir de l'information sur vos Inference Endpoints
- [`create_inference_endpoint`], [`update_inference_endpoint`] et [`delete_inference_endpoint`] pour déployer et gérer les Inference Endpoints
- [`pause_inference_endpoint`] et [`resume_inference_endpoint`] pour mettre en pause et relancer un Inference Enpoint
- [`scale_to_zero_inference_endpoint`] pour scale à la main l'Inference Endpoint à 0 replicas

## InferenceEndpoint

La dataclass principale est [`InferenceEndpoint`]. Elle contient des informations sur un `InferenceEndpoint` déployé, notamment sa configuration et son
état actuel. Une fois déployé, vous pouvez faire des inférences sur l'enpoint en utilisant les propriétés [`InferenceEndpoint.client`] et [`InferenceEndpoint.async_client`] qui retournent respectivement un objet [`InferenceClient`] et [`AsyncInferenceClient`]

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