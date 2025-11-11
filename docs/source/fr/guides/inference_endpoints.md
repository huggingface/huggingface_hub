<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Inference Endpoints

Inference Endpoints fournit une solution de production sécurisée pour déployer facilement n'importe quel modèle `transformers`, `sentence-transformers` et `diffusers` sur une infrastructure dédiée et auto-scalable gérée par Hugging Face. Un Inference Endpoint est construit à partir d'un modèle du [Hub](https://huggingface.co/models).
Dans ce guide, nous allons apprendre à gérer les Inference Endpoints avec `huggingface_hub`. Pour plus d'informations sur le produit Inference Endpoints lui-même, consultez sa [documentation officielle](https://huggingface.co/docs/inference-endpoints/index).

Ce guide suppose que `huggingface_hub` est correctement installé. Consultez le [guide de démarrage rapide](https://huggingface.co/docs/huggingface_hub/quick-start#quickstart) si ce n'est pas encore le cas. La version minimale supportant l'API Inference Endpoints est `v0.19.0`.


> [!TIP]
> **Nouveau :** il est maintenant possible de déployer un Inference Endpoint depuis le [catalogue de modèles HF](https://endpoints.huggingface.co/catalog) avec un simple appel d'API. Le catalogue est une liste soigneusement sélectionnée de modèles qui peuvent être déployés avec des paramètres optimisés. Vous n'avez rien besoin de configurer ! Tous les modèles et paramètres sont garantis d'avoir été testés pour fournir le meilleur équilibre coût/performance. [`create_inference_endpoint_from_catalog`] fonctionne de la même manière que [`create_inference_endpoint`], avec beaucoup moins de paramètres à passer. Vous pouvez utiliser [`list_inference_catalog`] pour récupérer le catalogue.
>
> Notez que ceci est encore une fonctionnalité expérimentale. Faites-nous savoir ce que vous en pensez si vous l'utilisez !


## Créer un Inference Endpoint

La première étape consiste à créer un Inference Endpoint en utilisant [`create_inference_endpoint`] :

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
...     instance_size="x2",
...     instance_type="intel-icl"
... )
```

Ou via CLI :

```bash
hf endpoints deploy my-endpoint-name --repo gpt2 --framework pytorch --accelerator cpu --vendor aws --region us-east-1 --instance-size x2 --instance-type intel-icl --task text-generation

# Déployer depuis le catalogue avec une seule commande
hf endpoints catalog deploy my-endpoint-name --repo openai/gpt-oss-120b
```


Dans cet exemple, nous avons créé un Inference Endpoint `protected` nommé `"my-endpoint-name"`, pour servir [gpt2](https://huggingface.co/gpt2) pour la `text-generation`. Un Inference Endpoint `protected` signifie que votre jeton est requis pour accéder à l'API. Nous devons également fournir des informations supplémentaires pour configurer les exigences matérielles, telles que le fournisseur, la région, l'accélérateur, le type d'instance et la taille. Vous pouvez consulter la liste des ressources disponibles [ici](https://api.endpoints.huggingface.cloud/#/v2%3A%3Aprovider/list_vendors). Alternativement, vous pouvez créer un Inference Endpoint manuellement en utilisant l'[interface Web](https://ui.endpoints.huggingface.co/new). Référez-vous à ce [guide](https://huggingface.co/docs/inference-endpoints/guides/advanced) pour plus de détails sur les paramètres et leur utilisation.

La valeur renvoyée par [`create_inference_endpoint`] est un objet [`InferenceEndpoint`] :

```py
>>> endpoint
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='pending', url=None)
```

Ou via CLI :

```bash
hf endpoints describe my-endpoint-name
```

C'est une dataclass qui contient des informations sur l'endpoint. Vous pouvez accéder à des attributs importants tels que `name`, `repository`, `status`, `task`, `created_at`, `updated_at`, etc. Si vous en avez besoin, vous pouvez également accéder à la réponse brute du serveur avec `endpoint.raw`.

Une fois votre Inference Endpoint créé, vous pouvez le trouver sur votre [tableau de bord personnel](https://ui.endpoints.huggingface.co/).

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/huggingface_hub/inference_endpoints_created.png)

#### Utiliser une image personnalisée

Par défaut, l'Inference Endpoint est construit à partir d'une image docker fournie par Hugging Face. Cependant, il est possible de spécifier n'importe quelle image docker en utilisant le paramètre `custom_image`. Un cas d'utilisation courant est d'exécuter des LLM en utilisant le framework [text-generation-inference](https://github.com/huggingface/text-generation-inference). Cela peut être fait comme ceci :

```python
# Démarrer un Inference Endpoint exécutant Zephyr-7b-beta sur TGI
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
...     instance_size="x1",
...     instance_type="nvidia-a10g",
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

La valeur à passer comme `custom_image` est un dictionnaire contenant une url vers le conteneur docker et la configuration pour l'exécuter. Pour plus de détails à ce sujet, consultez la [documentation Swagger](https://api.endpoints.huggingface.cloud/#/v2%3A%3Aendpoint/create_endpoint).

### Obtenir ou lister les Inference Endpoints existants

Dans certains cas, vous pourriez avoir besoin de gérer des Inference Endpoints que vous avez créés précédemment. Si vous connaissez le nom, vous pouvez le récupérer en utilisant [`get_inference_endpoint`], qui renvoie un objet [`InferenceEndpoint`]. Alternativement, vous pouvez utiliser [`list_inference_endpoints`] pour récupérer une liste de tous les Inference Endpoints. Les deux méthodes acceptent un paramètre optionnel `namespace`. Vous pouvez définir le `namespace` sur n'importe quelle organisation dont vous faites partie. Sinon, il est par défaut votre nom d'utilisateur.

```py
>>> from huggingface_hub import get_inference_endpoint, list_inference_endpoints

# Obtenir un
>>> get_inference_endpoint("my-endpoint-name")
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='pending', url=None)

# Lister tous les endpoints d'une organisation
>>> list_inference_endpoints(namespace="huggingface")
[InferenceEndpoint(name='aws-starchat-beta', namespace='huggingface', repository='HuggingFaceH4/starchat-beta', status='paused', url=None), ...]

# Lister tous les endpoints de toutes les organisations auxquelles l'utilisateur appartient
>>> list_inference_endpoints(namespace="*")
[InferenceEndpoint(name='aws-starchat-beta', namespace='huggingface', repository='HuggingFaceH4/starchat-beta', status='paused', url=None), ...]
```

Ou via CLI : 

```bash
hf endpoints describe my-endpoint-name
hf endpoints ls --namespace huggingface
hf endpoints ls --namespace '*'
```

## Vérifier le statut du déploiement

Dans le reste de ce guide, nous supposerons que nous avons un objet [`InferenceEndpoint`] appelé `endpoint`. Vous avez peut-être remarqué que l'endpoint a un attribut `status` de type [`InferenceEndpointStatus`]. Lorsque l'Inference Endpoint est déployé et accessible, le statut doit être `"running"` et l'attribut `url` est défini :

```py
>>> endpoint
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='running', url='https://jpj7k2q4j805b727.us-east-1.aws.endpoints.huggingface.cloud')
```

Avant d'atteindre un état `"running"`, l'Inference Endpoint passe généralement par une phase `"initializing"` ou `"pending"`. Vous pouvez récupérer le nouvel état de l'endpoint en exécutant [`~InferenceEndpoint.fetch`]. Comme toute autre méthode de [`InferenceEndpoint`] qui fait une requête au serveur, les attributs internes d'`endpoint` sont mutés sur place :

```py
>>> endpoint.fetch()
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='pending', url=None)
```

Ou via CLI :

```bash
hf endpoints describe my-endpoint-name
```

Au lieu de récupérer le statut de l'Inference Endpoint en attendant qu'il s'exécute, vous pouvez directement appeler [`~InferenceEndpoint.wait`]. Ce helper prend en entrée un paramètre `timeout` et `fetch_every` (en secondes) et bloquera le thread jusqu'à ce que l'Inference Endpoint soit déployé. Les valeurs par défaut sont respectivement `None` (pas de timeout) et `5` secondes.

```py
# Endpoint en attente
>>> endpoint
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='pending', url=None)

# Attendre 10s => lève une InferenceEndpointTimeoutError
>>> endpoint.wait(timeout=10)
    raise InferenceEndpointTimeoutError("Timeout while waiting for Inference Endpoint to be deployed.")
huggingface_hub._inference_endpoints.InferenceEndpointTimeoutError: Timeout while waiting for Inference Endpoint to be deployed.

# Attendre plus longtemps
>>> endpoint.wait()
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='running', url='https://jpj7k2q4j805b727.us-east-1.aws.endpoints.huggingface.cloud')
```

Si `timeout` est défini et que l'Inference Endpoint prend trop de temps à se charger, une erreur de timeout [`InferenceEndpointTimeoutError`] est levée.

## Exécuter l'inférence

Une fois que votre Inference Endpoint est opérationnel, vous pouvez enfin exécuter l'inférence dessus !

[`InferenceEndpoint`] a deux propriétés `client` et `async_client` renvoyant respectivement des objets [`InferenceClient`] et [`AsyncInferenceClient`].

```py
# Exécuter la tâche text_generation :
>>> endpoint.client.text_generation("I am")
' not a fan of the idea of a "big-budget" movie. I think it\'s a'

# Ou dans un contexte asyncio :
>>> await endpoint.async_client.text_generation("I am")
```

Si l'Inference Endpoint n'est pas en cours d'exécution, une exception [`InferenceEndpointError`] est levée :

```py
>>> endpoint.client
huggingface_hub._inference_endpoints.InferenceEndpointError: Cannot create a client for this Inference Endpoint as it is not yet deployed. Please wait for the Inference Endpoint to be deployed using `endpoint.wait()` and try again.
```

Pour plus de détails sur comment utiliser [`InferenceClient`], consultez le [guide Inference](../guides/inference).

## Gérer le cycle de vie

Maintenant que nous avons vu comment créer un Inference Endpoint et exécuter l'inférence dessus, voyons comment gérer son cycle de vie.

> [!TIP]
> Dans cette section, nous verrons des méthodes comme [`~InferenceEndpoint.pause`], [`~InferenceEndpoint.resume`], [`~InferenceEndpoint.scale_to_zero`], [`~InferenceEndpoint.update`] et [`~InferenceEndpoint.delete`]. Toutes ces méthodes sont des alias ajoutés à [`InferenceEndpoint`]. Si vous préférez, vous pouvez également utiliser les méthodes génériques définies dans `HfApi` : [`pause_inference_endpoint`], [`resume_inference_endpoint`], [`scale_to_zero_inference_endpoint`], [`update_inference_endpoint`], et [`delete_inference_endpoint`].

### Mettre en pause ou scale to zero

Pour réduire les coûts lorsque votre Inference Endpoint n'est pas utilisé, vous pouvez choisir soit de le mettre en pause en utilisant [`~InferenceEndpoint.pause`] soit de le scale to zero en utilisant [`~InferenceEndpoint.scale_to_zero`].

> [!TIP]
> Un Inference Endpoint qui est *en pause* ou *scaled to zero* ne coûte rien. La différence entre les deux est qu'un endpoint *en pause* doit être explicitement *repris* en utilisant [`~InferenceEndpoint.resume`]. Au contraire, un endpoint *scaled to zero* démarrera automatiquement si un appel d'inférence lui est fait, avec un délai de démarrage à froid supplémentaire. Un Inference Endpoint peut également être configuré pour scale to zero automatiquement après une certaine période d'inactivité.

```py
# Mettre en pause et reprendre l'endpoint
>>> endpoint.pause()
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='paused', url=None)
>>> endpoint.resume()
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='pending', url=None)
>>> endpoint.wait().client.text_generation(...)
...

# Scale to zero
>>> endpoint.scale_to_zero()
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='scaledToZero', url='https://jpj7k2q4j805b727.us-east-1.aws.endpoints.huggingface.cloud')
# L'endpoint n'est pas 'running' mais a toujours une URL et redémarrera au premier appel.
```

Ou via CLI :

```bash
hf endpoints pause my-endpoint-name
hf endpoints resume my-endpoint-name
hf endpoints scale-to-zero my-endpoint-name
```

### Mettre à jour le modèle ou les exigences matérielles

Dans certains cas, vous pourriez également vouloir mettre à jour votre Inference Endpoint sans en créer un nouveau. Vous pouvez soit mettre à jour le modèle hébergé soit les exigences matérielles pour exécuter le modèle. Vous pouvez le faire en utilisant [`~InferenceEndpoint.update`] :

```py
# Changer le modèle cible
>>> endpoint.update(repository="gpt2-large")
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2-large', status='pending', url=None)

# Mettre à jour le nombre de répliques
>>> endpoint.update(min_replica=2, max_replica=6)
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2-large', status='pending', url=None)

# Mettre à jour vers une instance plus grande
>>> endpoint.update(accelerator="cpu", instance_size="x4", instance_type="intel-icl")
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2-large', status='pending', url=None)
```

Ou via CLI :

```bash
hf endpoints update my-endpoint-name --repo gpt2-large
hf endpoints update my-endpoint-name --min-replica 2 --max-replica 6
hf endpoints update my-endpoint-name --accelerator cpu --instance-size x4 --instance-type intel-icl
```

### Supprimer l'endpoint

Enfin, si vous n'utiliserez plus l'Inference Endpoint, vous pouvez simplement appeler [`~InferenceEndpoint.delete()`].

> [!WARNING]
> Ceci est une action non réversible qui supprimera complètement l'endpoint, y compris sa configuration, ses logs et ses métriques d'utilisation. Vous ne pouvez pas restaurer un Inference Endpoint supprimé.


## Un exemple de bout en bout

Un cas d'utilisation typique des Inference Endpoints est de traiter un lot de jobs en une seule fois pour limiter les coûts d'infrastructure. Vous pouvez automatiser ce processus en utilisant ce que nous avons vu dans ce guide :

```py
>>> import asyncio
>>> from huggingface_hub import create_inference_endpoint

# Démarrer l'endpoint + attendre jusqu'à initialisation
>>> endpoint = create_inference_endpoint(name="batch-endpoint",...).wait()

# Exécuter l'inférence
>>> client = endpoint.client
>>> results = [client.text_generation(...) for job in jobs]

# Ou avec asyncio
>>> async_client = endpoint.async_client
>>> results = asyncio.gather(*[async_client.text_generation(...) for job in jobs])

# Mettre l'endpoint en pause
>>> endpoint.pause()
```

Ou si votre Inference Endpoint existe déjà et est en pause :

```py
>>> import asyncio
>>> from huggingface_hub import get_inference_endpoint

# Obtenir l'endpoint + attendre jusqu'à initialisation
>>> endpoint = get_inference_endpoint("batch-endpoint").resume().wait()

# Exécuter l'inférence
>>> async_client = endpoint.async_client
>>> results = asyncio.gather(*[async_client.text_generation(...) for job in jobs])

# Mettre l'endpoint en pause
>>> endpoint.pause()
```
