<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Exécuter l'inférence sur des serveurs

L'inférence est le processus d'utilisation d'un modèle entraîné pour faire des prédictions sur de nouvelles données. Ce processus peut être gourmand en calcul, l'exécuter sur un service dédié ou externe peut être une option intéressante.
La bibliothèque `huggingface_hub` fournit une interface unifiée pour exécuter l'inférence à travers plusieurs services pour les modèles hébergés sur le Hugging Face Hub :

1.  [Inference Providers](https://huggingface.co/docs/inference-providers/index) : un accès simplifié et unifié à des centaines de modèles de machine learning, alimenté par nos partenaires d'inférence serverless. Cette nouvelle approche s'appuie sur notre ancienne API d'inférence Serverless, offrant plus de modèles, de meilleures performances et une plus grande fiabilité grâce à des fournisseurs mondiale. Référez-vous à la [documentation](https://huggingface.co/docs/inference-providers/index#partners) pour une liste des fournisseurs supportés.
2.  [Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index) : Déployer facilement des modèles en production. L'inférence est exécutée par Hugging Face dans une infrastructure dédiée et entièrement gérée sur un fournisseur cloud de votre choix.
3.  Endpoints locaux : vous pouvez également exécuter l'inférence avec des serveurs d'inférence locaux comme [llama.cpp](https://github.com/ggerganov/llama.cpp), [Ollama](https://ollama.com/), [vLLM](https://github.com/vllm-project/vllm), [LiteLLM](https://docs.litellm.ai/docs/simple_proxy), ou [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference) en connectant le client à ces endpoints locaux.

> [!TIP]
> [`InferenceClient`] est un client Python qui effectue des appels HTTP vers nos APIs. Si vous souhaitez effectuer les appels HTTP directement en utilisant
> votre outil préféré (curl, postman,...), veuillez vous référer à la documentation [Inference Providers](https://huggingface.co/docs/inference-providers/index)
> ou aux pages de documentation [Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index).
>
> Pour le développement web, un [client JS](https://huggingface.co/docs/huggingface.js/inference/README) a été publié.
> Si vous êtes intéressé par le développement de jeux, vous pourriez jeter un œil à notre [projet C#](https://github.com/huggingface/unity-api).

## Premiers pas

Commençons avec une tâche text-to-image :

```python
>>> from huggingface_hub import InferenceClient

# Exemple avec un fournisseur externe (par ex. replicate)
>>> replicate_client = InferenceClient(
    provider="replicate",
    api_key="my_replicate_api_key",
)
>>> replicate_image = replicate_client.text_to_image(
    "A flying car crossing a futuristic cityscape.",
    model="black-forest-labs/FLUX.1-schnell",
)
>>> replicate_image.save("flying_car.png")

```

Dans l'exemple ci-dessus, nous avons initialisé un [`InferenceClient`] avec un fournisseur tiers, [Replicate](https://replicate.com/). Lors de l'utilisation d'un fournisseur, vous devez spécifier le modèle que vous souhaitez utiliser. L'ID du modèle doit être l'ID du modèle sur le Hugging Face Hub, et non l'ID du modèle du fournisseur tiers.
Dans notre exemple, nous avons généré une image à partir d'un prompt textuel. La valeur renvoyée est un objet `PIL.Image` qui peut être sauvegardé dans un fichier. Pour plus de détails, consultez la documentation [`~InferenceClient.text_to_image`].

Voyons maintenant un exemple utilisant l'API [`~InferenceClient.chat_completion`]. Cette tâche utilise un LLM pour générer une réponse à partir d'une liste de messages :

```python
>>> from huggingface_hub import InferenceClient
>>> messages = [
    {
        "role": "user",
        "content": "What is the capital of France?",
    }
]
>>> client = InferenceClient(
    provider="together",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    api_key="my_together_api_key",
)
>>> client.chat_completion(messages, max_tokens=100)
ChatCompletionOutput(
    choices=[
        ChatCompletionOutputComplete(
            finish_reason="eos_token",
            index=0,
            message=ChatCompletionOutputMessage(
                role="assistant", content="The capital of France is Paris.", name=None, tool_calls=None
            ),
            logprobs=None,
        )
    ],
    created=1719907176,
    id="",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    object="text_completion",
    system_fingerprint="2.0.4-sha-f426a33",
    usage=ChatCompletionOutputUsage(completion_tokens=8, prompt_tokens=17, total_tokens=25),
)
```

Dans l'exemple ci-dessus, nous avons utilisé un fournisseur tiers ([Together AI](https://www.together.ai/)) et spécifié quel modèle nous voulons utiliser (`"meta-llama/Meta-Llama-3-8B-Instruct"`). Nous avons ensuite donné une liste de messages à compléter (ici, une seule question) et passé un paramètre supplémentaire à l'API (`max_token=100`). La sortie est un objet `ChatCompletionOutput` qui suit la spécification OpenAI. Le contenu généré peut être accédé avec `output.choices[0].message.content`. Pour plus de détails, consultez la documentation [`~InferenceClient.chat_completion`].


> [!WARNING]
> L'API est conçue pour être simple. Tous les paramètres et options ne sont pas disponibles ou décrits pour l'utilisateur final. Consultez
> [cette page](https://huggingface.co/docs/api-inference/detailed_parameters) si vous êtes intéressé par en savoir plus sur
> tous les paramètres disponibles pour chaque tâche.

### Utiliser un fournisseur spécifique

Si vous souhaitez utiliser un fournisseur spécifique, vous pouvez le spécifier lors de l'initialisation du client. La valeur par défaut est "auto" qui sélectionnera le premier des fournisseurs disponibles pour le modèle, trié par l'ordre de l'utilisateur dans https://hf.co/settings/inference-providers. Référez-vous à la section [Fournisseurs et tâches supportés](#supported-providers-and-tasks) pour une liste des fournisseurs supportés.

```python
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(provider="replicate", api_key="my_replicate_api_key")
```

### Utiliser un modèle spécifique

Et si vous voulez utiliser un modèle spécifique ? Vous pouvez le spécifier soit comme un paramètre soit directement au niveau de l'instance :

```python
>>> from huggingface_hub import InferenceClient
# Initialiser le client pour un modèle spécifique
>>> client = InferenceClient(provider="together", model="meta-llama/Llama-3.1-8B-Instruct")
>>> client.text_to_image(...)
# Ou utiliser un client générique mais passer votre modèle comme argument
>>> client = InferenceClient(provider="together")
>>> client.text_to_image(..., model="meta-llama/Llama-3.1-8B-Instruct")
```

> [!TIP]
> Lors de l'utilisation du fournisseur "hf-inference", chaque tâche vient avec un modèle recommandé parmi les plus d'1M de modèles disponibles sur le Hub.
> Cependant, cette recommandation peut changer au fil du temps, il est donc préférable de définir explicitement un modèle une fois que vous avez décidé lequel utiliser.
> Pour les fournisseurs tiers, vous devez toujours spécifier un modèle compatible avec ce fournisseur.
>
> Visitez la page [Modèles](https://huggingface.co/models?inference=warm) sur le Hub pour explorer les modèles disponibles via les Inference Providers.

### Utiliser Inference Endpoints

Les exemples que nous avons vus ci-dessus utilisent des inference providers. Bien que ceux-ci s'avèrent très utiles pour prototyper
et tester rapidement des choses. Une fois que vous êtes prêt à déployer votre modèle en production, vous aurez besoin d'une infrastructure dédiée.
C'est là qu'[Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index) entre en jeu. Il vous permet de déployer
n'importe quel modèle et de l'exposer comme une API privée. Une fois déployé, vous obtiendrez une URL que vous pouvez connecter en utilisant exactement le même
code qu'avant, en changeant seulement le paramètre `model` :

```python
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(model="https://uu149rez6gw9ehej.eu-west-1.aws.endpoints.huggingface.cloud/deepfloyd-if")
# ou
>>> client = InferenceClient()
>>> client.text_to_image(..., model="https://uu149rez6gw9ehej.eu-west-1.aws.endpoints.huggingface.cloud/deepfloyd-if")
```

Notez que vous ne pouvez pas spécifier à la fois une URL et un fournisseur - ils sont mutuellement exclusifs. Les URL sont utilisées pour se connecter directement aux endpoints déployés.

### Utiliser des endpoints locaux

Vous pouvez utiliser [`InferenceClient`] pour exécuter une complétion de chat avec des serveurs d'inférence locaux (llama.cpp, vllm, serveur litellm, TGI, mlx, etc.) fonctionnant sur votre propre machine. L'API doit être compatible avec l'API OpenAI.

```python
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(model="http://localhost:8080")

>>> response = client.chat.completions.create(
...     messages=[
...         {"role": "user", "content": "What is the capital of France?"}
...     ],
...     max_tokens=100
... )
>>> print(response.choices[0].message.content)
```

> [!TIP]
> De manière similaire au client Python OpenAI, [`InferenceClient`] peut être utilisé pour exécuter l'inférence Chat Completion avec n'importe quel endpoint compatible avec l'API REST OpenAI.

### Authentification

L'authentification peut se faire de deux manières :

**Routé via Hugging Face** : Utilisez Hugging Face comme proxy pour accéder aux fournisseurs tiers. Les appels seront routés via l'infrastructure de Hugging Face en utilisant nos clés de fournisseur, et l'utilisation sera facturée directement sur votre compte Hugging Face.

Vous pouvez vous authentifier en utilisant un [User Access Token](https://huggingface.co/docs/hub/security-tokens). Vous pouvez fournir votre jeton Hugging Face directement en utilisant le paramètre `api_key` :

```python
>>> client = InferenceClient(
    provider="replicate",
    api_key="hf_****"  # Votre jeton HF
)
```

Si vous *ne passez pas* d'`api_key`, le client tentera de trouver et d'utiliser un jeton stocké localement sur votre machine. Cela se produit généralement si vous vous êtes déjà connecté précédemment. Consultez le [Guide d'authentification](https://huggingface.co/docs/huggingface_hub/quick-start#authentication) pour plus de détails sur la connexion.

```python
>>> client = InferenceClient(
    provider="replicate",
    token="hf_****"  # Votre jeton HF
)
```

**Accès direct au fournisseur** : Utilisez votre propre clé API pour interagir directement avec le service du fournisseur :
```python
>>> client = InferenceClient(
    provider="replicate",
    api_key="r8_****"  # Votre clé API Replicate
)
```

Pour plus de détails, référez-vous à la [documentation sur les prix des Inference Providers](https://huggingface.co/docs/inference-providers/pricing#routed-requests-vs-direct-calls).

## Fournisseurs et tâches supportés

L'objectif d'[`InferenceClient`] est de fournir l'interface la plus simple pour exécuter l'inférence sur les modèles Hugging Face, sur n'importe quel fournisseur. Il dispose d'une API simple qui supporte les tâches les plus courantes. Voici un tableau montrant quels fournisseurs supportent quelles tâches :

| Tâche                                               | Black Forest Labs | Cerebras | Clarifai | Cohere | fal-ai | Featherless AI | Fireworks AI | Groq | HF Inference | Hyperbolic | Nebius AI Studio | Novita AI | Nscale | Public AI | Replicate | Sambanova | Scaleway | Together | Wavespeed | Zai |
| --------------------------------------------------- | ----------------- | -------- | -------- | ------ | ------ | -------------- | ------------ | ---- | ------------ | ---------- | ---------------- | --------- | ------ | ---------- | --------- | --------- | --------- | -------- | --------- | ---- |
| [`~InferenceClient.audio_classification`]           | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        | ❌      | ❌          | ❌         | ❌         | ❌   |
| [`~InferenceClient.audio_to_audio`]                 | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        | ❌      | ❌          | ❌         | ❌         | ❌   |
| [`~InferenceClient.automatic_speech_recognition`]   | ❌                 | ❌        | ❌        | ❌      | ✅      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        | ❌      | ❌          | ❌         | ❌         | ❌   |
| [`~InferenceClient.chat_completion`]                | ❌                 | ✅        | ✅        | ✅      | ❌      | ✅              | ✅            | ✅    | ✅            | ✅          | ✅                | ✅         | ✅      | ✅          | ❌         | ✅         | ✅         | ✅        | ❌         | ✅   |
| [`~InferenceClient.document_question_answering`]    | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        | ❌      | ❌          | ❌         | ❌         | ❌   |
| [`~InferenceClient.feature_extraction`]             | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ✅                | ❌         | ❌      | ❌          | ❌         | ✅         | ✅         | ❌        | ❌         | ❌   |
| [`~InferenceClient.fill_mask`]                      | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        | ❌      | ❌          | ❌         | ❌         | ❌   |
| [`~InferenceClient.image_classification`]           | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        | ❌      | ❌          | ❌         | ❌         | ❌   |
| [`~InferenceClient.image_segmentation`]             | ❌                 | ❌        | ❌        | ❌      | ✅      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        | ❌      | ❌          | ❌         | ❌         | ❌   |
| [`~InferenceClient.image_to_image`]                 | ❌                 | ❌        | ❌        | ❌      | ✅      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌      | ❌          | ✅         | ❌         | ❌         | ❌        | ✅         | ❌   |
| [`~InferenceClient.image_to_video`]                 | ❌                 | ❌        | ❌        | ❌      | ✅      | ❌              | ❌            | ❌    | ❌            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        | ❌      | ❌          | ❌         | ✅         | ❌   |
| [`~InferenceClient.image_to_text`]                  | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        | ❌      | ❌          | ❌         | ❌         | ❌   |
| [`~InferenceClient.object_detection`]               | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        | ❌      | ❌          | ❌         | ❌         | ❌   |
| [`~InferenceClient.question_answering`]             | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        | ❌      | ❌          | ❌         | ❌         | ❌   |
| [`~InferenceClient.sentence_similarity`]            | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        | ❌      | ❌          | ❌         | ❌         | ❌   |
| [`~InferenceClient.summarization`]                  | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        | ❌      | ❌          | ❌         | ❌         | ❌   |
| [`~InferenceClient.table_question_answering`]       | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        | ❌      | ❌          | ❌         | ❌         | ❌   |
| [`~InferenceClient.text_classification`]            | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        | ❌      | ❌          | ❌         | ❌         | ❌   |
| [`~InferenceClient.text_generation`]                | ❌                 | ❌        | ❌        | ❌      | ❌      | ✅              | ❌            | ❌    | ✅            | ✅          | ✅                | ✅         | ❌      | ❌          | ❌         | ❌         | ❌         | ✅        | ❌         | ❌   |
| [`~InferenceClient.text_to_image`]                  | ✅                 | ❌        | ❌        | ❌      | ✅      | ❌              | ❌            | ❌    | ✅            | ✅          | ✅                | ❌         | ✅      | ❌          | ✅         | ❌         | ❌         | ✅        | ✅         | ❌   |
| [`~InferenceClient.text_to_speech`]                 | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌      | ❌          | ✅         | ❌         | ❌         | ❌        | ❌         | ❌   |
| [`~InferenceClient.text_to_video`]                  | ❌                 | ❌        | ❌        | ❌      | ✅      | ❌              | ❌            | ❌    | ❌            | ❌          | ❌                | ✅         | ❌      | ❌          | ✅         | ❌         | ❌         | ❌        | ✅         | ❌   |
| [`~InferenceClient.tabular_classification`]         | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        | ❌      | ❌          | ❌         | ❌         | ❌   |
| [`~InferenceClient.tabular_regression`]             | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        | ❌      | ❌          | ❌         | ❌         | ❌   |
| [`~InferenceClient.token_classification`]           | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        | ❌      | ❌          | ❌         | ❌         | ❌   |
| [`~InferenceClient.translation`]                    | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        | ❌      | ❌          | ❌         | ❌         | ❌   |
| [`~InferenceClient.visual_question_answering`]      | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        | ❌      | ❌          | ❌         | ❌         | ❌   |
| [`~InferenceClient.zero_shot_image_classification`] | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        | ❌      | ❌          | ❌         | ❌         | ❌   |
| [`~InferenceClient.zero_shot_classification`]       | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        | ❌      | ❌          | ❌         | ❌         | ❌   |

> [!TIP]
> Consultez la page [Tâches](https://huggingface.co/tasks) pour en savoir plus sur chaque tâche.

## Compatibilité OpenAI

La tâche `chat_completion` suit la syntaxe du [client Python d'OpenAI](https://github.com/openai/openai-python). Qu'est-ce que cela signifie pour vous ? Cela signifie que si vous avez l'habitude de travailler avec les API d'`OpenAI`, vous pourrez passer à `huggingface_hub.InferenceClient` pour travailler avec des modèles open-source en mettant à jour seulement 2 lignes de code !

```diff
- from openai import OpenAI
+ from huggingface_hub import InferenceClient

- client = OpenAI(
+ client = InferenceClient(
    base_url=...,
    api_key=...,
)


output = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Count to 10"},
    ],
    stream=True,
    max_tokens=1024,
)

for chunk in output:
    print(chunk.choices[0].delta.content)
```

Et c'est tout ! Les seules modifications requises sont de remplacer `from openai import OpenAI` par `from huggingface_hub import InferenceClient` et `client = OpenAI(...)` par `client = InferenceClient(...)`. Vous pouvez choisir n'importe quel modèle LLM du Hugging Face Hub en passant son ID de modèle comme paramètre `model`. [Voici une liste](https://huggingface.co/models?pipeline_tag=text-generation&other=conversational,text-generation-inference&sort=trending) des modèles supportés. Pour l'authentification, vous devez passer un [User Access Token](https://huggingface.co/settings/tokens) valide comme `api_key` ou vous authentifier en utilisant `huggingface_hub` (consultez le [guide d'authentification](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)).

Tous les paramètres d'entrée et le format de sortie sont strictement les mêmes. En particulier, vous pouvez passer `stream=True` pour recevoir les tokens au fur et à mesure qu'ils sont générés. Vous pouvez également utiliser [`AsyncInferenceClient`] pour exécuter l'inférence en utilisant `asyncio` :

```diff
import asyncio
- from openai import AsyncOpenAI
+ from huggingface_hub import AsyncInferenceClient

- client = AsyncOpenAI()
+ client = AsyncInferenceClient()

async def main():
    stream = await client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[{"role": "user", "content": "Say this is a test"}],
        stream=True,
    )
    async for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="")

asyncio.run(main())
```

Vous vous demandez peut-être pourquoi utiliser [`InferenceClient`] au lieu du client OpenAI ? Il y a quelques raisons à cela :
1. [`InferenceClient`] est configuré pour les services Hugging Face. Vous n'avez pas besoin de fournir de `base_url` pour exécuter des modèles avec les Inference Providers. Vous n'avez pas non plus besoin de fournir de `token` ou `api_key` si votre machine est déjà correctement connectée.
2. [`InferenceClient`] est adapté à la fois aux frameworks Text-Generation-Inference (TGI) et à `transformers`, ce qui signifie que vous êtes assuré qu'il sera toujours en phase avec les dernières mises à jour.
3. [`InferenceClient`] est intégré avec notre service Inference Endpoints, ce qui facilite le lancement d'un Inference Endpoint, la vérification de son statut et l'exécution de l'inférence dessus. Consultez le guide [Inference Endpoints](./inference_endpoints.md) pour plus de détails.

> [!TIP]
> `InferenceClient.chat.completions.create` est simplement un alias pour `InferenceClient.chat_completion`. Consultez la référence du package de [`~InferenceClient.chat_completion`] pour plus de détails. Les paramètres `base_url` et `api_key` lors de l'instanciation du client sont également des alias pour `model` et `token`. Ces alias ont été définis pour réduire les frictions lors du passage d'`OpenAI` à `InferenceClient`.

## Function Calling

Le Function Calling permet aux LLM d'interagir avec des outils externes, tels que des fonctions définies ou des API. Cela permet aux utilisateurs de créer facilement des applications adaptées à des cas d'utilisation spécifiques et à des tâches du monde réel.
`InferenceClient` implémente la même interface d'appel de tool que l'API OpenAI Chat Completions. Voici un exemple simple d'appel de tool utilisant [Nebius](https://nebius.com/) comme fournisseur d'inférence :

```python
from huggingface_hub import InferenceClient

tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current temperature for a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and country e.g. Paris, France"
                        }
                    },
                    "required": ["location"],
                },
            }
        }
]

client = InferenceClient(provider="nebius")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-72B-Instruct",
    messages=[
    {
        "role": "user",
        "content": "What's the weather like the next 3 days in London, UK?"
    }
    ],
    tools=tools,
    tool_choice="auto",
)

print(response.choices[0].message.tool_calls[0].function.arguments)

```

> [!TIP]
> Veuillez vous référer à la documentation des fournisseurs pour vérifier quels modèles sont supportés par eux pour le Function/Tool Calling.

## Structured Outputs & JSON Mode

InferenceClient supporte le mode JSON pour des réponses JSON syntaxiquement valides et les Structured Outputs pour des réponses conformes à un schéma. Le mode JSON fournit des données lisibles par machine sans structure stricte, tandis que les Structured Outputs garantissent à la fois un JSON valide et l'adhésion à un schéma prédéfini pour un traitement en aval fiable.

Nous suivons les spécifications de l'API OpenAI pour le mode JSON et les Structured Outputs. Vous pouvez les activer via l'argument `response_format`. Voici un exemple de Structured Outputs utilisant [Cerebras](https://www.cerebras.ai/) comme fournisseur d'inférence :

```python
from huggingface_hub import InferenceClient

json_schema = {
    "name": "book",
    "schema": {
        "properties": {
            "name": {
                "title": "Name",
                "type": "string",
            },
            "authors": {
                "items": {"type": "string"},
                "title": "Authors",
                "type": "array",
            },
        },
        "required": ["name", "authors"],
        "title": "Book",
        "type": "object",
    },
    "strict": True,
}

client = InferenceClient(provider="cerebras")


completion = client.chat.completions.create(
    model="Qwen/Qwen3-32B",
    messages=[
        {"role": "system", "content": "Extract the books information."},
        {"role": "user", "content": "I recently read 'The Great Gatsby' by F. Scott Fitzgerald."},
    ],
    response_format={
        "type": "json_schema",
        "json_schema": json_schema,
    },
)

print(completion.choices[0].message)
```
> [!TIP]
> Veuillez vous référer à la documentation des fournisseurs pour vérifier quels modèles sont supportés par eux pour les Structured Outputs et le JSON Mode.

## Client asynchrone

Une version asynchrone du client est également fournie, basée sur `asyncio` et `httpx`. Tous les endpoints d'API asynchrones sont disponibles via [`AsyncInferenceClient`]. Son initialisation et ses API sont strictement les mêmes que la version sync-only.

```py
# Le code doit être exécuté dans un contexte concurrent asyncio.
# $ python -m asyncio
>>> from huggingface_hub import AsyncInferenceClient
>>> client = AsyncInferenceClient()

>>> image = await client.text_to_image("An astronaut riding a horse on the moon.")
>>> image.save("astronaut.png")

>>> async for token in await client.text_generation("The Huggingface Hub is", stream=True):
...     print(token, end="")
 a platform for sharing and discussing ML-related content.
```

Pour plus d'informations sur le module `asyncio`, veuillez consulter la [documentation officielle](https://docs.python.org/3/library/asyncio.html).

## MCP Client

La bibliothèque `huggingface_hub` inclut maintenant un [`MCPClient`] expérimental, conçu pour permettre aux Large Language Models (LLM) d'interagir avec des outils externes via le [Model Context Protocol](https://modelcontextprotocol.io) (MCP). Ce client étend un [`AsyncInferenceClient`] pour intégrer  l'utilisation des outils.

Le [`MCPClient`] se connecte aux serveurs MCP (soit des scripts `stdio` locaux soit des services `http`/`sse` distants) qui exposent des outils. Il alimente ces outils à un LLM (via [`AsyncInferenceClient`]). Si le LLM décide d'utiliser un outil, [`MCPClient`] gère la requête d'exécution au serveur MCP et relaie la sortie de l'outil au LLM, souvent en streaming les résultats en temps réel.

Dans l'exemple suivant, nous utilisons le modèle [Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) via le fournisseur d'inférence [Nebius](https://nebius.com/). Nous ajoutons ensuite un serveur MCP distant, dans ce cas, un serveur SSE qui a rendu l'outil de génération d'image Flux disponible au LLM.

```python
import os

from huggingface_hub import ChatCompletionInputMessage, ChatCompletionStreamOutput, MCPClient


async def main():
    async with MCPClient(
        provider="nebius",
        model="Qwen/Qwen2.5-72B-Instruct",
        api_key=os.environ["HF_TOKEN"],
    ) as client:
        await client.add_mcp_server(type="sse", url="https://evalstate-flux1-schnell.hf.space/gradio_api/mcp/sse")

        messages = [
            {
                "role": "user",
                "content": "Generate a picture of a cat on the moon",
            }
        ]

        async for chunk in client.process_single_turn_with_tools(messages):
            # Journaliser les messages
            if isinstance(chunk, ChatCompletionStreamOutput):
                delta = chunk.choices[0].delta
                if delta.content:
                    print(delta.content, end="")

            # Ou les appels d'outils
            elif isinstance(chunk, ChatCompletionInputMessage):
                print(
                    f"\nCalled tool '{chunk.name}'. Result: '{chunk.content if len(chunk.content) < 1000 else chunk.content[:1000] + '...'}'"
                )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
```


Pour un développement encore plus simple, nous offrons une classe [`Agent`] de plus haut niveau. Ce 'Tiny Agent' simplifie la création d'Agents conversationnels en gérant la boucle de chat et l'état, agissant essentiellement comme un wrapper autour de [`MCPClient`]. Il est conçu pour être une simple boucle while construite directement sur un [`MCPClient`]. Vous pouvez exécuter ces Agents directement depuis la ligne de commande :


```bash
# installer la dernière version de huggingface_hub avec l'extra mcp
pip install -U huggingface_hub[mcp]
# Exécuter un agent qui utilise l'outil de génération d'image Flux
tiny-agents run julien-c/flux-schnell-generator

```

Lorsqu'il est lancé, l'Agent se chargera, listera les outils qu'il a découverts depuis ses serveurs MCP connectés, et ensuite il est prêt pour vos prompts !

## Conseils avancés

Dans la section ci-dessus, nous avons vu les principaux aspects d'[`InferenceClient`]. Plongeons dans quelques conseils plus avancés.

### Facturation

En tant qu'utilisateur de HF, vous obtenez des crédits mensuels pour exécuter l'inférence via divers fournisseurs sur le Hub. Le montant de crédits que vous obtenez dépend de votre type de compte (Gratuit ou PRO ou Enterprise Hub). Vous êtes facturé pour chaque requête d'inférence, en fonction du tableau des prix du fournisseur. Par défaut, les requêtes sont facturées à votre compte personnel. Cependant, il est possible de définir la facturation pour que les requêtes soient facturées à une organisation dont vous faites partie en passant simplement `bill_to="<your_org_name>"` à `InferenceClient`. Pour que cela fonctionne, votre organisation doit être abonnée à Enterprise Hub. Pour plus de détails sur la facturation, consultez [ce guide](https://huggingface.co/docs/api-inference/pricing#features-using-inference-providers).

```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(provider="fal-ai", bill_to="openai")
>>> image = client.text_to_image(
...     "A majestic lion in a fantasy forest",
...     model="black-forest-labs/FLUX.1-schnell",
... )
>>> image.save("lion.png")
```

Notez qu'il n'est PAS possible de facturer un autre utilisateur ou une organisation dont vous ne faites pas partie. Si vous souhaitez accorder des crédits à quelqu'un d'autre, vous devez créer une organisation conjointe avec eux.


### Timeout

Les appels d'inférence peuvent prendre beaucoup de temps. Par défaut, [`InferenceClient`] attendra "indéfiniment" jusqu'à ce que l'inférence se termine. Si vous voulez plus de contrôle dans votre workflow, vous pouvez définir le paramètre `timeout` à une valeur spécifique en secondes. Si le délai de timeout expire, une [`InferenceTimeoutError`] est levée, que vous pouvez capturer dans votre code :

```python
>>> from huggingface_hub import InferenceClient, InferenceTimeoutError
>>> client = InferenceClient(timeout=30)
>>> try:
...     client.text_to_image(...)
... except InferenceTimeoutError:
...     print("Inference timed out after 30s.")
```

### Entrées binaires

Certaines tâches nécessitent des entrées binaires, par exemple lors du traitement d'images ou de fichiers audio. Dans ce cas, [`InferenceClient`]
essaie d'être aussi permissif que possible et d'accepter différents types :
- des `bytes` bruts
- un objet de type fichier, ouvert en binaire (`with open("audio.flac", "rb") as f: ...`)
- un chemin (`str` ou `Path`) pointant vers un fichier local
- une URL (`str`) pointant vers un fichier distant (par ex. `https://...`). Dans ce cas, le fichier sera téléchargé localement avant
d'être envoyé à l'API.

```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()
>>> client.image_classification("https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg")
[{'score': 0.9779096841812134, 'label': 'Blenheim spaniel'}, ...]
```
