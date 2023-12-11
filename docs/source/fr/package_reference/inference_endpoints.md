# Inference Endpoints

Inferrence Endpoints est une solutions permettant de déployer facilement les modèles en production sur une infrastructure dédiée et capaable de faire de l'autoscaling gérée par Hugging Face. Un Inference Endpoint peut être crée sur un modèle depuis le [Hub](https://huggingface.co/models). Cette page est
une référence pour l'intégration d'`huggingface_hub` avec Inference Endpoints. Pour plus d'informations à propos du produit Inference Endpoints, consulez
la [documentation officielle](https://huggingface.co/docs/inference-endpoints/index).

<Tip>

Consultez ce [guide](../guides/inference_endpoints) pour apprendre comment utiliser `huggingface_hub` pour gérer votre Inference Enpoints depuis le code.

</Tip>

Inference Endpoints peut-être complètement géré depuis une API. Les enpoints sont consultables via [Swagger](https://api.endpoints.huggingface.cloud/).
La classe [`InferenceEndpoint`] est un simple wrapper autour de cette API.

## Méthodes

Un sous ensemble des fonctionnalités de l'Inference Endpoint sont implémentées dans [`HfApi`]: 

- [`get_inference_endpoint`] et [`list_inference_endpoints`] pour obtenir de l'information sur votre Inference Endpoints
- [`create_inference_endpoint`], [`update_inference_endpoint`] et [`delete_inference_endpoint`] pour déployer et gérer Inference Endpoints
- [`pause_inference_endpoint`] et [`resume_inference_endpoint`] pour mettre en pause et relancer un Inference Enpoint
- [`scale_to_zero_inference_endpoint`] pour scale à la main l'Inference Endpoint à 0 replicas

## InferenceEndpoint

La dataclass principale est [`InferenceEndpoint`]. Elle contient des informations sur un `InferenceEndpoint` déployé, incluant sa configuration et son
état actuel. Une fois déployé, vous pouvez faire des inférences sur l'enpoint en utilisant les propriétés [`InferenceEndpoint.client`] et [`InferenceEndpoint.async_client`] qui return respectivement un objet [`InferenceClient`] et [`AsyncInferenceClient`]

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