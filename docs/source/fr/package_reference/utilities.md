<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Utilitaires

## Configurer la connexion

Le package `huggingface_hub` expose un utilitaire `logging` qui permet de contrôler le niveau d'authentification
du package. Vous pouvez l'importer ainsi:

```py
from huggingface_hub import logging
```

Ensuite, vous pourrez définir la verbosité afin de changer la quantité de logs que vous
verez:

```python
from huggingface_hub import logging

logging.set_verbosity_error()
logging.set_verbosity_warning()
logging.set_verbosity_info()
logging.set_verbosity_debug()

logging.set_verbosity(...)
```

Les niveau de verbosité doivent être compris comme suit:

- `error`: ne montre que les logs critiques qui pourrait causer une erreur ou un comportement innatendu.
- `warning`: montre des logs qui ne sont pas critiques mais dont l'utilisation pourrait causer un comportement inattendu.
En plus de ça, des informations importantes pourraient être affichées.
- `info`: montre la plupart des logs, dont des informations de logging donnant des informations sur ce que fait la fonction en arrière plan.
Si quelque chose se comporte de manière innatendue, nous recommendons de passer le niveau de verbosité à `info` afin d'avoir plus
d'informations.
- `debug`: montre tous les logs, dont des logs internes qui pourraient utiliser pour suivre en détail tout ce qui se passe en 
arrière plan.

[[autodoc]] logging.get_verbosity
[[autodoc]] logging.set_verbosity
[[autodoc]] logging.set_verbosity_info
[[autodoc]] logging.set_verbosity_debug
[[autodoc]] logging.set_verbosity_warning
[[autodoc]] logging.set_verbosity_error
[[autodoc]] logging.disable_propagation
[[autodoc]] logging.enable_propagation

### Méthodes d'helper spécifiques au dépôt.

Les méthodes montrés ci-dessous sont pertinentes lors de la modification de modules de la librairie `huggingface_hub`.
Utiliser ces méthodes ne devrait pas être nécessaire si vous utilisez les méthodes`huggingface_hub`
et que vous ne les modifiez pas.

[[autodoc]] logging.get_logger

## Configurez la barre de progression

Les barres de progression sont utiles pour afficher des informations à l'utiliser lors de l'exécution d'une longue
tâche (i.e. lors du téléchargement ou de l'upload de fichiers). `huggingface_hub` met à disposition un wrapper
[`~utils.tqdm`] qui permet d'afficher constamment les barres de progressions pour toutes les fonctions de la
librairie.

Par défaut, les barres de progressions sont activées. Vous pouvez les désactiver en définissant la
variable d'environnement `HF_HUB_DISABLE_PROGRESS_BARS`. Vous pouvez aussi les activer/désactiver en
utilisant [`~utils.enable_progress_bars`] et [`~utils.disable_progress_bars`]. Si définie, la variable
d'environnement a la priorité sur les helpers.


```py
>>> from huggingface_hub import snapshot_download
>>> from huggingface_hub.utils import are_progress_bars_disabled, disable_progress_bars, enable_progress_bars

>>> # Supprimez toutes les barres de progression
>>> disable_progress_bars()

>>> # Les barres de progressions ne seront pas affichées !
>>> snapshot_download("gpt2")

>>> are_progress_bars_disabled()
True

>>> # Réactivez toutes les barres de progression
>>> enable_progress_bars()
```

### are_progress_bars_disabled

[[autodoc]] huggingface_hub.utils.are_progress_bars_disabled

### disable_progress_bars

[[autodoc]] huggingface_hub.utils.disable_progress_bars

### enable_progress_bars

[[autodoc]] huggingface_hub.utils.enable_progress_bars

## Configurez un backend HTTPConfigure HTTP backend

Dans certains environnements, vous aurez peut être envie de configurer la manière dont les appels HTTP sont faits,
par exemple, si vous utilisez un proxy. `huggingface_hub` vous permet de configurer ceci en utilisant [`configure_http_backend`].
Toutes les requêtes faites au Hub utiliseront alors vos paramètres.En arrière-plan, `huggingface_hub` utilise
`requests.Session`, vous aurez donc peut être besoin de consultez la [documentation `requests`](https://requests.readthedocs.io/en/latest/user/advanced)
pour en savoir plus sur les paramètres disponibles.

Vu que `requests.Session` n'est pas toujours à l'abri d'un problème de thread, `huggingface_hub` créé une seule
instance de session par thread. L'utilisation de ces sessions permet de garder la connexion ouverte entre les appels HTTP
afin de gagner du temps. Si vous êtes entrain d'intégrer `huggingface_hub` dans une autre librairie tiers et que vous
voulez faire des appels personnalisés vers le Hub, utilisez [`get_session`] pour obtenir une session configurée par
vos utilisateurs (i.e. remplacez tous les appels à `requests.get(...)` par `get_session().get(...)`).

[[autodoc]] configure_http_backend

[[autodoc]] get_session


## Gérez les erreurs HTTP

`huggingface_hub` définit ses propres erreurs HTTP pour améliorer le `HTTPError`
levé par `requests` avec des informations supplémentaires envoyées par le serveur.

### Raise for status

[`~utils.hf_raise_for_status`] est la méthode centrale pour "raise for status" depuis n'importe quelle
reqête faite au Hub. C'est wrapper autour de `requests.raise_for_status` qui fournit des informations
supplémentaires. Toute `HTTPError` envoyée est convertie en `HfHubHTTPError`.

```py
import requests
from huggingface_hub.utils import hf_raise_for_status, HfHubHTTPError

response = requests.post(...)
try:
    hf_raise_for_status(response)
except HfHubHTTPError as e:
    print(str(e)) # message formaté
    e.request_id, e.server_message # détails retourné par le serveur

    # complétez le message d'erreur avec des informations additionnelles une fois que l'erreur est levée
    e.append_to_message("\n`create_commit` expects the repository to exist.")
    raise
```

[[autodoc]] huggingface_hub.utils.hf_raise_for_status

### Erreurs HTTP

Voici une liste des erreurs HTTP levée dans `huggingface_hub`.

#### HfHubHTTPError

`HfHubHTTPError` est la classe parente de toute erreur HTTP venant de HF Hub. Elle
va parser les réponses du serveur et formater le message d'erreur pour fournir le 
plus d'informations possible à l'utilisateur.

[[autodoc]] huggingface_hub.utils.HfHubHTTPError

#### RepositoryNotFoundError

[[autodoc]] huggingface_hub.utils.RepositoryNotFoundError

#### GatedRepoError

[[autodoc]] huggingface_hub.utils.GatedRepoError

#### RevisionNotFoundError

[[autodoc]] huggingface_hub.utils.RevisionNotFoundError

#### EntryNotFoundError

[[autodoc]] huggingface_hub.utils.EntryNotFoundError

#### BadRequestError

[[autodoc]] huggingface_hub.utils.BadRequestError

#### LocalEntryNotFoundError

[[autodoc]] huggingface_hub.utils.LocalEntryNotFoundError

## Télémétrie

`huggingface_hub` inclus un helper pour envoyer de la donnée de télémétrie. Cette information nous aide à debugger des problèmes
et prioriser les nouvelles fonctionnalités à développer. Les utilisateurs peuvent desactiver la collecte télémetrique à n'importe
quel moment et définissant la variable d'environnement `HF_HUB_DISABLE_TELEMETRY=1`. La télémétrie est aussi desactivée en mode
hors ligne (i.e. en définissant `HF_HUB_OFFLINE=1`).

Si vous êtes mainteneur d'une librairie tiercce, envoyer des données de télémetrie est aussi facile que de faire un appel
à [`send_telemetry`]. Les données sont envoyées dans un thread séparé pour réduire autant que possible l'impact sur
l'expérience utilisateur.

[[autodoc]] utils.send_telemetry


## Les validateurs

`huggingface_hub` offre des validateurs personnalisés pour valider la méthode des
arguements automatiquement. La validation est inspirée du travail fait dans
[Pydantic](https://pydantic-docs.helpmanual.io/) pour valider les hints mais
avec des fonctionnalités plus limitées.

### Décorateur générique

[`~utils.validate_hf_hub_args`] est un décorateur générique pour encapsuler
des méthodes qui ont des arguments qui ont le nom d'`huggingface_hub`.
Par défaut, tous les arguments qui ont un validateur implémenté seront
validés.

Si une entrée n'est pas valide, une erreur [`~utils.HFValidationError`]
est levée. Seul la première valeur invalide déclenche une erreur et
interrompt le processus de validation.


Utilisation:

```py
>>> from huggingface_hub.utils import validate_hf_hub_args

>>> @validate_hf_hub_args
... def my_cool_method(repo_id: str):
...     print(repo_id)

>>> my_cool_method(repo_id="valid_repo_id")
valid_repo_id

>>> my_cool_method("other..repo..id")
huggingface_hub.utils._validators.HFValidationError: Cannot have -- or .. in repo_id: 'other..repo..id'.

>>> my_cool_method(repo_id="other..repo..id")
huggingface_hub.utils._validators.HFValidationError: Cannot have -- or .. in repo_id: 'other..repo..id'.

>>> @validate_hf_hub_args
... def my_cool_auth_method(token: str):
...     print(token)

>>> my_cool_auth_method(token="a token")
"a token"

>>> my_cool_auth_method(use_auth_token="a use_auth_token")
"a use_auth_token"

>>> my_cool_auth_method(token="a token", use_auth_token="a use_auth_token")
UserWarning: Both `token` and `use_auth_token` are passed (...). `use_auth_token` value will be ignored.
"a token"
```

#### validate_hf_hub_args

[[autodoc]] utils.validate_hf_hub_args

#### HFValidationError

[[autodoc]] utils.HFValidationError

### Validateurs d'arguments

Les validateurs peuvent aussi être utilisés individuellement. Voici une liste de tous
les arguments qui peuvent être validés.

#### repo_id

[[autodoc]] utils.validate_repo_id

#### smoothly_deprecate_use_auth_token

Pas vraiment un validateur, mais utilisé aussi.

[[autodoc]] utils.smoothly_deprecate_use_auth_token
