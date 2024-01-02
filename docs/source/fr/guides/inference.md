<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Faire de l'inférence sur des serveurs

L'inférence est le fait d'utiliser un modèle déjà entrainé pour faire des prédictions sur de nouvelles données. Comme ce
processus peut demander beaucoup de ressources computationnelles, le lancer sur un serveur dédié peut être une option
intéressante. La librairie `huggingface_hub` offre une manière facile d'appeler un service qui fait de l'inférence pour
les modèles hébergés. Il y a plusieurs services auxquels vous pouvez vous connecter:
- [Inference API](https://huggingface.co/docs/api-inference/index): un service qui vous permet de faire des inférences accélérées
sur l'infrastructure Hugging Face gratuitement. Ce service est une manière rapide de commencer, tester différents modèles et
créer des premiers prototypes de produits IA.
-[Inference Endpoints](https://huggingface.co/inference-endpoints): un produit qui permet de déployer facilement des modèles en production.
L'inférence est assurée par Hugging Face dans l'infrastructure dédiée d'un cloud provider de votre choix.

Ces services peuvent être appelés avec l'objet [`InferenceClient`]. Ce dernier remplace le client historique [`InferenceApi`],
en ajoutant plus de support pour les tâches et la gestion de l'inférence avec [Inference API](https://huggingface.co/docs/api-inference/index) and [Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index). Apprenez comment migrer vers le nouveau client dans la section
[client historique InferenceAPI](#legacy-inferenceapi-client).

<Tip>

[`InferenceClient`] est un client Python qui fait des appels HTTP à nos APIs. Si vous voulez faire des appels HTTP
directement en utilisant votre outil préféré (curl, postman,...), consultez les pages de documentation
d'[Inference API](https://huggingface.co/docs/api-inference/index) ou d'[Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index)


Pour le développement web, un [JS client](https://huggingface.co/docs/huggingface.js/inference/README) a été créé.
Si vous êtes intéressés par le développement de jeu, consultez notre [projet C#](https://github.com/huggingface/unity-api).

</Tip>

## Commencer avec les inférences

Commençons avec du text-to-image:

```python
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()

>>> image = client.text_to_image("An astronaut riding a horse on the moon.")
>>> image.save("astronaut.png")
```

Nous avons initialisé un [`InferenceClient`] avec les paramètres par défaut. La seule chose que vous avez besoin de savoir est la
[tâche](#supported-tasks) que vous voulez réaliser. Par défaut, le client se connectera à Inference API et sélectionnera un modèle
pour assurer la tâche. Dans notre exemple, nous avons généré une image depuis un prompt textuel. La valeur renvoyée est un objet
`PIL.image` qui peut être enregistré en tant que fichier.

<Tip warning={true}>

L'API est fait pour être simple. Cette paramètres et options ne sont pas disponibles pour l'utilisateur. Consultez
[cette page](https://huggingface.co/docs/api-inference/detailed_parameters) si vous voulez en apprendre plus sur
tous les paramètres disponibles pour chacune des tâches.

</Tip>

### Utiliser un modèle spécifique

Et si vous voulez utiliser un modèle spécifique? Vous pouvez le préciser soit en tant que paramètre ou directement
au niveau de l'instance:

```python
>>> from huggingface_hub import InferenceClient
# Initialise le client pour un modèle spécifique
>>> client = InferenceClient(model="prompthero/openjourney-v4")
>>> client.text_to_image(...)
# Ou bien utiliser un client générique mais mettre son modèle en argument
>>> client = InferenceClient()
>>> client.text_to_image(..., model="prompthero/openjourney-v4")
```

<Tip>

Il y a plus de 200 000 modèles sur le HUb Hugging Face! Chaque tâche dans [`InferenceClient`] a un modèle recommandé
qui lui est assigné. Attention, les recommendantions HF peuvent changer du jour au lendemain sans avertissement préalable.
Par conséquent il vaut mieux définir explicitement un modèle une fois que vous l'avez choisi. En plus dans la plupart des
cas, vous aurez besoin d'un modèle qui  colle à vos besoins spécifiques. Consultez la page [Models](https://huggingface.co/models)
sur le Hub pour explorer vos possibilités.

</Tip>

### Utiliser un URL spécifique

Les exemples vu ci dessis utilisent l'API hébergé gratuitement. Cette API est très utile pour les prototypes
et les tests agiles. Une fois que vous êtes prêts à déployer vos modèles en production, vous aurez besoin d'utiliser
des infrastructures dédiées. C'est là que les [enpoints d'inférence](https://huggingface.co/docs/inference-endpoints/index) 
rentrent en jeu. Cet outil permet de déployer n'importe quel modèle et de l'exposer en tant que'API privé. Une fois déployé,
vous obtiendrez un URL auquel vous pouvez vous connecter en utilisant exactement le même code qu'avant, vous aurez juste
à changer le paramètre `model`:

```python
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(model="https://uu149rez6gw9ehej.eu-west-1.aws.endpoints.huggingface.cloud/deepfloyd-if")
# ou bien
>>> client = InferenceClient()
>>> client.text_to_image(..., model="https://uu149rez6gw9ehej.eu-west-1.aws.endpoints.huggingface.cloud/deepfloyd-if")
```

### Authentification

Les appels faits avec le client [`InferenceClient`] peuvent passer par de l'authentification en utilisant un 
[token d'authentification](https://huggingface.co/docs/hub/security-tokens). Par défaut, le token enregistré sur votre machine sera
utilisé si vous êtes connectés (consultez [comment se connecter](https://huggingface.co/docs/huggingface_hub/quick-start#login)
pour plus de détails). Si vous n'êtes pas connectés, vous pouvez passer votre token comme paramètre d'instance:

```python
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(token="hf_***")
```

<Tip>

L'authentification n'est PAS obligatoire lorsque que vous utilisez Inference API. Cependant, les utilisateur authentifiés
ont droit à un free-tier. Les tokens sont de plus obligatoires si vous voulez faire de l'inférence sur vos modèles privés
ou sur des endpoints privés. 

</Tip>

## Tâches supportés

Le but d'[`InferenceClient`] est de fournir l'interface la plus simple possible pour faire de l'inférence
sur des modèles Hugging Face. Ce client possède une API simple qui supporte la plupart des tâches usuelles.
Voici une liste des tâches actuellement supportées: 

| Domaine | Tâche                           | Supporté     | Documentation                             |
|--------|--------------------------------|--------------|------------------------------------|
| Audio | [Classification audio ](https://huggingface.co/tasks/audio-classification)           | ✅ | [`~InferenceClient.audio_classification`] |
| | [Reconaissance vocal automatique](https://huggingface.co/tasks/automatic-speech-recognition)| ✅ | [`~InferenceClient.automatic_speech_recognition`] |
| | [Text-to-Speech](https://huggingface.co/tasks/text-to-speech)                 | ✅ | [`~InferenceClient.text_to_speech`] |
| Computer Vision | [Classification d'images](https://huggingface.co/tasks/image-classification)        | ✅ | [`~InferenceClient.image_classification`] |
| | [Ségmentation d'images ](https://huggingface.co/tasks/image-segmentation)             | ✅ | [`~InferenceClient.image_segmentation`] |
| | [Image-to-image](https://huggingface.co/tasks/image-to-image)                 | ✅ | [`~InferenceClient.image_to_image`] |
| | [Image-to-text](https://huggingface.co/tasks/image-to-text)                  | ✅ | [`~InferenceClient.image_to_text`] |
| | [Détection d'objets](https://huggingface.co/tasks/object-detection)            | ✅ | [`~InferenceClient.object_detection`] |
| | [Text-to-image](https://huggingface.co/tasks/text-to-image)                  | ✅ | [`~InferenceClient.text_to_image`] |
| | [Classification d'image zero-shot](https://huggingface.co/tasks/zero-shot-image-classification)| ✅ |[`~InferenceClient.zero_shot_image_classification`]|
| Multimodal | [Réponse de questions liées à de la documentation](https://huggingface.co/tasks/document-question-answering) | ✅ | [`~InferenceClient.document_question_answering`] 
| | [Réponse de questions visuelles](https://huggingface.co/tasks/visual-question-answering)      | ✅ | [`~InferenceClient.visual_question_answering`] |
| NLP | [conversationnel](https://huggingface.co/tasks/conversational)                 | ✅ | [`~InferenceClient.conversational`] |
| | [Feature Extraction](https://huggingface.co/tasks/feature-extraction)             | ✅ | [`~InferenceClient.feature_extraction`] |
| | [Fill Mask](https://huggingface.co/tasks/fill-mask)                      | ✅ | [`~InferenceClient.fill_mask`] |
| | [Réponse à des questions](https://huggingface.co/tasks/question-answering)             | ✅ | [`~InferenceClient.question_answering`]
| | [Similarité de phrase](https://huggingface.co/tasks/sentence-similarity)            | ✅ | [`~InferenceClient.sentence_similarity`] |
| | [Création de résumés](https://huggingface.co/tasks/summarization)                  | ✅ | [`~InferenceClient.summarization`] |
| | [Réponse de questions sous forme de tables](https://huggingface.co/tasks/table-question-answering)       | ✅ | [`~InferenceClient.table_question_answering`] |
| | [Classification de texte](https://huggingface.co/tasks/text-classification)            | ✅ | [`~InferenceClient.text_classification`] |
| | [Génération de texte](https://huggingface.co/tasks/text-generation)   | ✅ | [`~InferenceClient.text_generation`] |
| | [Classification de tokens](https://huggingface.co/tasks/token-classification)           | ✅ | [`~InferenceClient.token_classification`] |
| | [Traduction](https://huggingface.co/tasks/translation)       | ✅ | [`~InferenceClient.translation`] |
| | [Classification zero-shot](https://huggingface.co/tasks/zero-shot-classification)       | ✅ | [`~InferenceClient.zero_shot_classification`] |
| Tabular | [Classification tabulaire](https://huggingface.co/tasks/tabular-classification)         | ✅ | [`~InferenceClient.tabular_classification`] |
| | [Régression Tabulaire](https://huggingface.co/tasks/tabular-regression)             | ✅ | [`~InferenceClient.tabular_regression`] |

<Tip>

Consultez la page de [Tâches](https://huggingface.co/tasks) pour en savoir plus sur chaque tâche, comment les utiliser
et les modèles les plus populaires pour chacune des tâches.

</Tip>

## Requêtes personnalisées

Toutefois, il n'est pas toujours possible de couvrir tous les cas d'usages avec ces tâches. Pour faire des requêtes
personnalisées, la méthode [`InferenceClient.post`] vous offre la flexibilité d'envoyer n'importe quelle requête à
l'API d'inférence. Par exemple, vous pouvez spécifier comment parser les entrées et les sorties. Dans l'exemple
ci-dessous, l'image générée est renvoyée en bytes aulieu d'être parsée en tant qu'`image PIL`.
Ceci peut s'avérer utile si vous n'avez pas `Pillow` d'installé sur votre machine et que vous voulez juste avoir
le contenu binaire de l'image.  [`InferenceClient.post`] est aussi utile pour gérer les tâches qui ne sont pas
encore supportées officiellement

```python
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()
>>> response = client.post(json={"inputs": "An astronaut riding a horse on the moon."}, model="stabilityai/stable-diffusion-2-1")
>>> response.content # bytes
b'...'
```

## Client asynchrone

Une version asynchrone du client est aussi fournie, basée sur `asyncio` et `aiohttp`. Vous avez le choix entre installer
`aiohttp` directmeent ou utiliser l'extra `[inference]`:

```sh
pip install --upgrade huggingface_hub[inference]
# ou alors
# pip install aiohttp
```

Après l'installation, toutes les API asynchrones sont disponibles via [`AsyncInferenceClient`]. Son initialisation et
les APIs sont exactement les mêmes que sur la version synchrone.

```py
# Le code doit tourner dans un contexte asyncio
# $ python -m asyncio
>>> from huggingface_hub import AsyncInferenceClient
>>> client = AsyncInferenceClient()

>>> image = await client.text_to_image("An astronaut riding a horse on the moon.")
>>> image.save("astronaut.png")

>>> async for token in await client.text_generation("The Huggingface Hub is", stream=True):
...     print(token, end="")
 a platform for sharing and discussing ML-related content.
```

Pour plus d'informations sur le module `asyncio`, consultez la [documentation officielle](https://docs.python.org/3/library/asyncio.html).

## Astuces avancées

Dans la section ci-dessus, nous avons vu les aspects principaux d' [`InferenceClient`]. Voyons maintenant les
astuces plus avanceées.

### Timeout

Lorsque vous faites de l'inférence, il y a deux causes principales qui causent un timeout:
- Le processus d'inférence prend beaucoup de temps à finir.
- Le modèle n'est pas disponible, par exemple quand Inference API charge le modèle pour la première fois.

[`InferenceClient`] possède un paramètre global `timeout` qui gère ces deux aspects. Par défaut, il a pour valeur
`None`, ce qui signifie que le client attendra indéfiniment pour que l'inférence soit complète. Si vous voulez plus
de controle sur votre workflow, vous pouvez lui donner une valeur spécifique en secondes. Si le délai de timeout
expire, une erreur [`InferenceTimeoutError`] est levée. Vous pouvez la catch et la gérer dans votre code:

```python
>>> from huggingface_hub import InferenceClient, InferenceTimeoutError
>>> client = InferenceClient(timeout=30)
>>> try:
...     client.text_to_image(...)
... except InferenceTimeoutError:
...     print("Inference timed out after 30s.")
```

### Entrée binaires

Certaines tâches demandent des entrées binaire, par exemple, lorsque vous utilisez des images ou des fichiers audio. Dans
ce cas, [`InferenceClient`] essaye d'accepter différent types:
- Des`bytes`
- Un fichier, ouvert en tant que binaire (`with open("audio.flac", "rb") as f: ...`)
- un chemin (`str` ou `path`) qui pointe vers un fichier local
- Un URL (`str`) qui pointe vers un fichier distant (i.e. `https://...`). Dans ce cas là, le fichier sera téléchargé en local
avant d'être envoyé à l'API.

```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()
>>> client.image_classification("https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg")
[{'score': 0.9779096841812134, 'label': 'Blenheim spaniel'}, ...]
```

## Client historique InferenceAPI

[`InferenceClient`] sert de remplacement pour l'approche historique [`InferenceApi`]. Il ajoute des supports spécifiques
pour des tâches et gère l'inférence avec [Inference API](https://huggingface.co/docs/api-inference/index) et [Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index).

Voici un guide très court qui vous aidera à migrer d'[`InferenceApi`] à [`InferenceClient`].

### Initialisation

Remplacez

```python
>>> from huggingface_hub import InferenceApi
>>> inference = InferenceApi(repo_id="bert-base-uncased", token=API_TOKEN)
```

par

```python
>>> from huggingface_hub import InferenceClient
>>> inference = InferenceClient(model="bert-base-uncased", token=API_TOKEN)
```

### Réaliser une tâche spécifique

Remplacez

```python
>>> from huggingface_hub import InferenceApi
>>> inference = InferenceApi(repo_id="paraphrase-xlm-r-multilingual-v1", task="feature-extraction")
>>> inference(...)
```

Par

```python
>>> from huggingface_hub import InferenceClient
>>> inference = InferenceClient()
>>> inference.feature_extraction(..., model="paraphrase-xlm-r-multilingual-v1")
```

<Tip>

C'est la méthode recommandée pour adapter votre code à [`InferenceClient`]. Elle vous permet de bénéficier des
méthodes spécifiques à une tâche telles que `feature_extraction`.

</Tip>

### Faire un requête personnalisée

Remplacez

```python
>>> from huggingface_hub import InferenceApi
>>> inference = InferenceApi(repo_id="bert-base-uncased")
>>> inference(inputs="The goal of life is [MASK].")
[{'sequence': 'the goal of life is life.', 'score': 0.10933292657136917, 'token': 2166, 'token_str': 'life'}]
```

par

```python
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()
>>> response = client.post(json={"inputs": "The goal of life is [MASK]."}, model="bert-base-uncased")
>>> response.json()
[{'sequence': 'the goal of life is life.', 'score': 0.10933292657136917, 'token': 2166, 'token_str': 'life'}]
```

### INférence avec des paramètres

Remplacez

```python
>>> from huggingface_hub import InferenceApi
>>> inference = InferenceApi(repo_id="typeform/distilbert-base-uncased-mnli")
>>> inputs = "Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!"
>>> params = {"candidate_labels":["refund", "legal", "faq"]}
>>> inference(inputs, params)
{'sequence': 'Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!', 'labels': ['refund', 'faq', 'legal'], 'scores': [0.9378499388694763, 0.04914155602455139, 0.013008488342165947]}
```

par

```python
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()
>>> inputs = "Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!"
>>> params = {"candidate_labels":["refund", "legal", "faq"]}
>>> response = client.post(json={"inputs": inputs, "parameters": params}, model="typeform/distilbert-base-uncased-mnli")
>>> response.json()
{'sequence': 'Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!', 'labels': ['refund', 'faq', 'legal'], 'scores': [0.9378499388694763, 0.04914155602455139, 0.013008488342165947]}
```
