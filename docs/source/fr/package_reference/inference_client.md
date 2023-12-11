<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Inférence

L'inférence est le fait d'utiliser un modèle déjà entrainé pour faire des prédictions sur de nouvelles données. Comme ce
processus peut demander beaucoup de ressources computationnelles, le lancer sur un serveur dédié peut être une option
intéressante. La librairie `huggingface_hub` offre une manière facile d'appeler un service qui fait de l'inférence pour
les modèles hébergés. Il y a plusieurs services auxquels vous pouvez vous connecter:
- [Inference API](https://huggingface.co/docs/api-inference/index): un service qui vous permet de faire des inférences accélérée
sur l'infrastructure Hugging Face gratuitement. Ce service est une manière rapide de commencer, tester différents modèles et
créer des premiers prototypes de produits IA.
-[Inference Endpoints](https://huggingface.co/inference-endpoints): un produit qui permet de déployer facilement des modèles en production.
L'inférence est assurée par Hugging Face dans l'infrastructure dédiée d'un cloud provider de votre choix.

Ces services peuvent être appelés avec l'objet [`InferenceClient`]. Consultez [ce guide](../guides/inference) pour plus
d'informations sur le mode d'utilisation.

## Client d'inférence

[[autodoc]] InferenceClient

## Client d'inférence asynchrone

Une version asychrone du client basée sur `asyncio` et `aihttp` est aussi fournie.
Pour l'utiliser, vous pouvez soit installer `aiohttp` directement ou utiliser l'extra `[inference]`:

```sh
pip install --upgrade huggingface_hub[inference]
# or
# pip install aiohttp
```

[[autodoc]] AsyncInferenceClient

## InferenceTimeoutError

[[autodoc]] InferenceTimeoutError

## Types retournés

Pour la plupart des tâches, la valeur retournée a un type intégré (string, list, image...). Voici une liste de types plus complexes.

### ClassificationOutput

[[autodoc]] huggingface_hub.inference._types.ClassificationOutput

### ConversationalOutputConversation

[[autodoc]] huggingface_hub.inference._types.ConversationalOutputConversation

### ConversationalOutput

[[autodoc]] huggingface_hub.inference._types.ConversationalOutput

### ImageSegmentationOutput

[[autodoc]] huggingface_hub.inference._types.ImageSegmentationOutput

### ModelStatus

[[autodoc]] huggingface_hub.inference._common.ModelStatus

### TokenClassificationOutput

[[autodoc]] huggingface_hub.inference._types.TokenClassificationOutput

### Types pour la génération de texte

La tâche [`~InferenceClient.text_generation`] a un meilleur support que d'autres tâches dans `InferenceClient`.
Plus précisément, les inputs des utilisateurs et les outputs des serveurs sont validés en utilisant [Pydantic](https://docs.pydantic.dev/latest/)
si ce package est installé. Par conséquent, nous vous recommandons de l'installer (`pip install pydantic`) pour
une meilleure expérience.

Vous pouvez trouver ci-dessous, les dataclasses utilisées pour valider des données et en particulier
[`~huggingface_hub.inference._text_generation.TextGenerationParameters`] (input)
[`~huggingface_hub.inference._text_generation.TextGenerationResponse`] (output) et
[`~huggingface_hub.inference._text_generation.TextGenerationStreamResponse`] (streaming output).

[[autodoc]] huggingface_hub.inference._text_generation.TextGenerationParameters

[[autodoc]] huggingface_hub.inference._text_generation.TextGenerationResponse

[[autodoc]] huggingface_hub.inference._text_generation.TextGenerationStreamResponse

[[autodoc]] huggingface_hub.inference._text_generation.InputToken

[[autodoc]] huggingface_hub.inference._text_generation.Token

[[autodoc]] huggingface_hub.inference._text_generation.FinishReason

[[autodoc]] huggingface_hub.inference._text_generation.BestOfSequence

[[autodoc]] huggingface_hub.inference._text_generation.Details

[[autodoc]] huggingface_hub.inference._text_generation.StreamDetails

## InferenceAPI

[`InferenceAPI`] est la méthode historique pour appeler l'API d'inférence. L'interface est plus simpliste et
demande une conaissance des paramètres d'entrées et du format de sortie de chacune des tâches. Cette interface
ne peut pas se connecter à d'autres services tels que Inference Endpoints or AWS SageMaker. [`InferenceAPI`] sera
bientôt deprecated, ainsi, nous recommendons l'utilisation de [`InferenceClient`] quand c'est possible.
Consultez [ce guide](../guides/inference#legacy-inferenceapi-client) pour apprendre comment passer
d'[`InferenceAPI`] à [`InferenceClient`] dans vos scripts.

[[autodoc]] InferenceApi
    - __init__
    - __call__
    - all

