# Inference Endpoints

Inference Endpoints fournit une solution viable pour la production et sécurisée pour déployer facilement n'importe quel modèle `transformers`, `sentence-transformers`, et `diffusers` sur une infrastructure dédiée et capable d'autoscaling gérée par Hugging Face. Un endpoint d'inférence est construit à partir d'un modèle du [Hub](https://huggingface.co/models).
Dans ce guide, nous apprendront comment gérer les endpoints d'inférence par le code en utilisant `huggingface_hub`. Pour plus d'informations sur le produit lui même, consultez sa [documentation officielle](https://huggingface.co/docs/inference-endpoints/index).

Ce guide postule que vous avez installé `huggingface_hub` correctement et que votre machine est connectée. Consultez le [guide quick start](https://huggingface.co/docs/huggingface_hub/quick-start#quickstart) si ce n'est pas le cas. La version la plus ancienne supportant l'API d'inference endpoints est `v0.19.0`.


## Créez un endpoint d'inférence

La première étape pour créer un endpoint d'inférence est d'utiliser [`create_inference_endpoint`]:

```py
>>> from huggingface_hub import create_inference_endpoint

>>> endpoint = create_inference_endpoint(
...     "my-endpoint-name",
...     repository="gpt2",
...     framework="pytorch",
...     task="text-generation",
...     accelerator="cpu",
...     vendor="aws",
...     region="us-east-1",
...     type="protected",
...     instance_size="medium",
...     instance_type="c6i"
... )
```

Dans cet exemple, nous avons créé un endpoint d'inférence de type `protected` qui a pour nom `"my-endpoint-name"`, il utilise [gpt2](https://huggingface.co/gpt2) pour faire de la génération de texte (`text-generation`). Le type `protected` signfie que votre token sera demandé pour accéder à l'API. Nous aurons aussi besoin de fournir des informations supplémentaires pour préciser le hardware nécessaire, tel que le provider, la région, l'accélérateur, le type d'instance et la taille. Vous pouvez consulter la liste des ressources disponibles [ici](https://api.endpoints.huggingface.cloud/#/v2%3A%3Aprovider/list_vendors). Par ailleurs, vous pouvez aussi créer un endpoint d'inférence manuellement en utilisant l'[interface web](https://ui.endpoints.huggingface.co/new) si c'est plus pratique pour vous. Consultez ce [guide](https://huggingface.co/docs/inference-endpoints/guides/advanced)  pour des détails sur les paramètres avancés et leur utilisation.

La valeur renvoyée par [`create_inference_endpoint`] est un objet [`InferenceEndpoint`]: 

```py
>>> endpoint
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='pending', url=None)
```

C'est une dataclass qui a des informations sur l'endpoitn. Vous pouvez avoir accès à des attributs importants tels que `name`, `repository`, `status`, `task`, `created_at`, `updated_at`, etc. (respectivement le nom, le dépôt d'origine, le statut, la tâche assignée, la date de création et la date de dernière modification). Si vous en avez besoin, vous pouvez aussi avoir accès à la réponse brute du serveur avec `endpoint.raw`.

Une fois que votre endpoint d'inférence est créé, vous pouvez le retrouver sur votre [dashboard personnel](https://ui.endpoints.huggingface.co/).

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/huggingface_hub/inference_endpoints_created.png)

#### Utiliser une image personnalisée

Par défaut, l'endpoint d'inférence est construit à partir d'une image docker fournie par Hugging Face. Cependant, i lest possible de préciser n'importe quelle image docker en utilisant le paramètre `custom_image`. Un cas d'usage fréquent est l'utilisation des LLM avec le framework [text-generation-inference](https://github.com/huggingface/text-generation-inference). On peut le faire ainsi:

```python
# Créé un endpoint d'inférence utilisant le modèle Zephyr-7b-beta sur une TGI
>>> from huggingface_hub import create_inference_endpoint
>>> endpoint = create_inference_endpoint(
...     "aws-zephyr-7b-beta-0486",
...     repository="HuggingFaceH4/zephyr-7b-beta",
...     framework="pytorch",
...     task="text-generation",
...     accelerator="gpu",
...     vendor="aws",
...     region="us-east-1",
...     type="protected",
...     instance_size="medium",
...     instance_type="g5.2xlarge",
...     custom_image={
...         "health_route": "/health",
...         "env": {
...             "MAX_BATCH_PREFILL_TOKENS": "2048",
...             "MAX_INPUT_LENGTH": "1024",
...             "MAX_TOTAL_TOKENS": "1512",
...             "MODEL_ID": "/repository"
...         },
...         "url": "ghcr.io/huggingface/text-generation-inference:1.1.0",
...     },
... )
```

La valeur à passer dans `custom_image` est un dictionnaire contenant un url vers le conteneur docker et la configuration pour le lancer. Pour plus de détails, consultez la [documentation Swagger](https://api.endpoints.huggingface.cloud/#/v2%3A%3Aendpoint/create_endpoint).

### Obtenir ou lister tous les endpoints d"inférence existants

Dans certains cas, vous aurez besoin de gérer les endpoints d'inférence précédemment créés. Si vous connaissez leur nom, vous pouvez les récupérer en utilisant [`get_inference_endpoint`], qui renvoie un objet [`INferenceEndpoint`]. Sinon, vous pouvez utiliser [`list_inference_endpoints`] pour récupérer une liste de tous les endpoints d'inférence. Les deux méthodes acceptent en paramètre optionnel `namespace`. Vous pouvez mettre en `namespace`  n'importe quelle organisation dont vous faites partie. Si vous ne renseignez pas ce paramètre, votre nom d'utilisateur sera utilisé par défaut.

```py
>>> from huggingface_hub import get_inference_endpoint, list_inference_endpoints

# Obtiens un endpoint
>>> get_inference_endpoint("my-endpoint-name")
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='pending', url=None)

# Liste tous les endpoints d'une organisation
>>> list_inference_endpoints(namespace="huggingface")
[InferenceEndpoint(name='aws-starchat-beta', namespace='huggingface', repository='HuggingFaceH4/starchat-beta', status='paused', url=None), ...]

# Liste tous les endpoints de toutes les organisation dont l'utilisateur fait partie
>>> list_inference_endpoints(namespace="*")
[InferenceEndpoint(name='aws-starchat-beta', namespace='huggingface', repository='HuggingFaceH4/starchat-beta', status='paused', url=None), ...]
```

## Vérifier le statu de déploiement

Dans le reste de ce guide, nous supposons que nous possèdons un objet [`InferenceEndpoint`] appelé `endpoint`. Vous avez peut-être remarqué que l'endpoint a un attribut `status` de type [`InferenceEndpointStatus`]. Lorsque l'endpoint d'inférence est déployé et accessible, le statut est `"running"` et l'attribut `url` est défini:

```py
>>> endpoint
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='running', url='https://jpj7k2q4j805b727.us-east-1.aws.endpoints.huggingface.cloud')
```

Avant d'atteindre l'état `"running"`, l'endpoint d'inférence passe généralement par une phase `"initializing"` ou `"pending"`. Vous pouvez récupérer le nouvel état de l'endpoint en lançant [`~InferenceEndpoint.fetch`]. Comme toutes les autres méthodes d'[`InferenceEndpoint`] qui envoient une requête vers le serveur, les attributs internes d'`endpoint` sont mutés sur place:

```py
>>> endpoint.fetch()
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='pending', url=None)
```

Aulieu de récupérer le statut de l'endpoint d'inférence lorsque vous attendez qu'il soit lancé, vous pouvez directement appeler
[`~InferenceEndpoint.wait`]. Cet helper prend en entrée les paramètres `timeout` et `fetch_every` (en secondes) et bloquera le thread jusqu'à ce que l'endpoint d'inférence soit déployé. Les valeurs par défaut sont respectivement `None` (pas de timeout) et `5` secondes.

```py
# Endpoint en attente
>>> endpoint
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='pending', url=None)

# Attend 10s puis lève une InferenceEndpointTimeoutError
>>> endpoint.wait(timeout=10)
    raise InferenceEndpointTimeoutError("Timeout while waiting for Inference Endpoint to be deployed.")
huggingface_hub._inference_endpoints.InferenceEndpointTimeoutError: Timeout while waiting for Inference Endpoint to be deployed.

# Attend plus
>>> endpoint.wait()
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='running', url='https://jpj7k2q4j805b727.us-east-1.aws.endpoints.huggingface.cloud')
```

Si `timeout` est définit et que l'endpoint d'inférence prend trop de temps à charger, une erreur [`InferenceEndpointTimeouError`] est levée.

## Lancez des inférences

Une fois que votre endpoint d'inférence est fonctionnel, vous pouvez enfin faire de l'inférence avec!

[`InferenceEndpoint`] a deux propriétés `client` et `async_client` qui renvoient respectivement des objets [`InferenceClient`] et [`AsyncInferenceClient`].

```py
# Lancez un tâche de génération de texte:
>>> endpoint.client.text_generation("I am")
' not a fan of the idea of a "big-budget" movie. I think it\'s a'

# Ou dans un contexte asynchrone:
>>> await endpoint.async_client.text_generation("I am")
```

Si l'endpoint d'inférence n'est pas opérationnel, une exception [`InferenceEndpointError`] est levée:

```py
>>> endpoint.client
huggingface_hub._inference_endpoints.InferenceEndpointError: Cannot create a client for this Inference Endpoint as it is not yet deployed. Please wait for the Inference Endpoint to be deployed using `endpoint.wait()` and try again.
```

Pour plus de détails sur l'utilisation d'[`InferenceClient`], consultez le [guide d'inférence](../guides/inference).

## Gérer les cycles de vie


Maintenant que nous avons vu comment créer un endpoint d'inférence et faire de l'inférence avec, regardons comment gérer son cycle de vie.

<Tip>

Dans cette section, nous verrons des méthodes telles que [`~InferenceEndpoint.pause`], [`~InferenceEndpoint.resume`], [`~InferenceEndpoint.scale_to_zero`], [`~InferenceEndpoint.update`] et [`~InferenceEndpoint.delete`]. Toutes ces méthodes sont des alias ajoutés à [`InferenceEndpoint`]. Si vous préférez, vous pouvez aussi utiliser les méthodes génériques définies dans `HfApi`: [`pause_inference_endpoint`], [`resume_inference_endpoint`], [`scale_to_zero_inference_endpoint`], [`update_inference_endpoint`], and [`delete_inference_endpoint`].

</Tip>

### Mettre en pause ou scale à zéro

Pour réduire les coûts lorsque votre endpoint d'inférence n'est pas utilisé, vous pouvez choisir soit de le mettre en pause en utilisant [`~InferenceEndpoint.pause`] ou de réaliser un scaling à zéro en utilisant [`~InferenceEndpoint.scale_to_zero`].

<Tip>

Un endpoint d'inférence qui est *en pause* ou *scalé à zéro* ne coute rien. La différence entre ces deux méthodes est qu'un endpoint *en pause* doit être *relancé* explicitement en utilisant [`~InferenceEndpoint.resume`]. A l'opposé, un endpoint *scalé à zéro* sera automatiquement lancé si un appel d'inférence est fait, avec un délai de "cold start" (temps de démarrage des instances) additionnel. Un endpoint d'inférence peut aussi être configuré pour scale à zero automatiquement après une certaine durée d'inactivité.

</Tip>

```py
# Met en pause et relance un endpoint
>>> endpoint.pause()
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='paused', url=None)
>>> endpoint.resume()
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='pending', url=None)
>>> endpoint.wait().client.text_generation(...)
...

# Scale à zéro
>>> endpoint.scale_to_zero()
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='scaledToZero', url='https://jpj7k2q4j805b727.us-east-1.aws.endpoints.huggingface.cloud')
# L'endpoint n'est pas en mode 'running' mais a tout de même un URL et sera relancé au premier call
```

### Mettre à jour le modèle ou le hardware de l'endpoint

Dans certains cas, vous aurez besoin de mettre à jour votre endpoint d'inférence sans en créer de nouveau. Vous avez le choix entre mettre à jour le modèle hébergé par l'endpoint ou le hardware utilisé pour faire tourner le modèle. Vous pouvez faire ça en utilisant [`~InferenceEndpoint.update`]:

```py
# Change le modèle utilisé
>>> endpoint.update(repository="gpt2-large")
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2-large', status='pending', url=None)

# Met à jour le nombre de replicas
>>> endpoint.update(min_replica=2, max_replica=6)
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2-large', status='pending', url=None)

# Met à jour la taille de l'instance
>>> endpoint.update(accelerator="cpu", instance_size="large", instance_type="c6i")
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2-large', status='pending', url=None)
```

### Supprimer un endpoint

Si vous n'utilisez plus un endpoint d'inférence, vous pouvez simplement appeler la méthode [`~InferenceEndpoint.delete()`].

<Tip warning={true}>

Cette action est irréversible et supprimera complètement l'endpoint, dont sa configuration, ses logs et ses métriques. Vous ne pouvez pas retrouver un endpoint d'inférence supprimé.

</Tip>


## Exemple de A à Z

Un cas d'usage typique d'Hugging Face pour les endpoints d'inférence est des gérer une liste de tâche d'un coup pour limiter les coûts en infrastructure. Vous pouvez automatiser ce processus en utilisant ce que nous avons vu dans ce guide:

```py
>>> import asyncio
>>> from huggingface_hub import create_inference_endpoint

# Lance un endpoint et attend qu'il soit initialisé
>>> endpoint = create_inference_endpoint(name="batch-endpoint",...).wait()

# Fait des inféreces
>>> client = endpoint.client
>>> results = [client.text_generation(...) for job in jobs]

# Ou bien avec asyncio
>>> async_client = endpoint.async_client
>>> results = asyncio.gather(*[async_client.text_generation(...) for job in jobs])

# Met en pause l'endpoint
>>> endpoint.pause()
```

Ou si votre endpoint d'inférence existe et est en pause:

```py
>>> import asyncio
>>> from huggingface_hub import get_inference_endpoint

# Récupère l'endpoint et attend son initialisation
>>> endpoint = get_inference_endpoint("batch-endpoint").resume().wait()

# Fait des inféreces
>>> async_client = endpoint.async_client
>>> results = asyncio.gather(*[async_client.text_generation(...) for job in jobs])

# Met en pause l'endpoint
>>> endpoint.pause()
```