<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Inference

L'inférence est le processus d'utilisation d'un modèle entraîné pour faire des prédictions sur de nouvelles données. Parce que ce processus peut être intensif en calculs, l'exécuter sur un service dédié ou externe peut être une option intéressante.
La bibliothèque `huggingface_hub` fournit une interface unifiée pour exécuter l'inférence à travers plusieurs services pour les modèles hébergés sur le Hugging Face Hub :

1.  [Inference Providers](https://huggingface.co/docs/inference-providers/index) : un accès simplifié et unifié à des centaines de modèles de machine learning, alimenté par nos partenaires d'inférence serverless. Cette nouvelle approche s'appuie sur notre précédente API d'inférence Serverless, offrant plus de modèles, des performances améliorées et une plus grande fiabilité grâce à des fournisseurs de classe mondiale. Consultez la [documentation](https://huggingface.co/docs/inference-providers/index#partners) pour une liste des fournisseurs supportés.
2.  [Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index) : un produit pour déployer facilement des modèles en production. L'inférence est exécutée par Hugging Face dans une infrastructure dédiée et entièrement gérée sur un fournisseur cloud de votre choix.
3.  Endpoints locaux : vous pouvez également exécuter l'inférence avec des serveurs d'inférence locaux comme [llama.cpp](https://github.com/ggerganov/llama.cpp), [Ollama](https://ollama.com/), [vLLM](https://github.com/vllm-project/vllm), [LiteLLM](https://docs.litellm.ai/docs/simple_proxy), ou [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference) en connectant le client à ces endpoints locaux.

Ces services peuvent être appelés avec l'objet [`InferenceClient`]. Veuillez vous référer à [ce guide](../guides/inference)
pour plus d'informations sur comment l'utiliser.

## Inference Client

[[autodoc]] InferenceClient

## Async Inference Client

Une version async du client est également fournie, basée sur `asyncio` et `httpx`.

[[autodoc]] AsyncInferenceClient

## InferenceTimeoutError

[[autodoc]] InferenceTimeoutError
