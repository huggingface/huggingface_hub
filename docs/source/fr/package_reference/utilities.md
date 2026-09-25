<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Utilitaires

## Configurer le logging

Le package `huggingface_hub` expose un utilitaire `logging` pour contrôler le niveau de logging du package lui-même.
Vous pouvez l'importer comme suit :

```py
from huggingface_hub import logging
```

Ensuite, vous pouvez définir la verbosité afin de mettre à jour la quantité de logs que vous verrez :

```python
from huggingface_hub import logging

logging.set_verbosity_error()
logging.set_verbosity_warning()
logging.set_verbosity_info()
logging.set_verbosity_debug()

logging.set_verbosity(...)
```

Les niveaux doivent être compris comme suit :

- `error` : afficher uniquement les logs critiques concernant l'utilisation qui peut entraîner une erreur ou un comportement inattendu.
- `warning` : afficher les logs qui ne sont pas critiques mais l'utilisation peut entraîner un comportement involontaire.
  De plus, des logs informatifs importants peuvent être affichés.
- `info` : afficher la plupart des logs, y compris certains logs verbeux concernant ce qui se passe sous le capot.
  Si quelque chose se comporte de manière inattendue, nous recommandons de passer le niveau de verbosité à celui-ci afin
  d'obtenir plus d'informations.
- `debug` : afficher tous les logs, y compris certains logs internes qui peuvent être utilisés pour suivre exactement ce qui se passe
  sous le capot.

[[autodoc]] logging.get_verbosity
[[autodoc]] logging.set_verbosity
[[autodoc]] logging.set_verbosity_info
[[autodoc]] logging.set_verbosity_debug
[[autodoc]] logging.set_verbosity_warning
[[autodoc]] logging.set_verbosity_error
[[autodoc]] logging.disable_propagation
[[autodoc]] logging.enable_propagation

### Méthodes helper spécifiques au dépôt

Les méthodes exposées ci-dessous sont pertinentes lors de la modification de modules de la bibliothèque `huggingface_hub` elle-même.
L'utilisation de celles-ci ne devrait pas être nécessaire si vous utilisez `huggingface_hub` et que vous ne les modifiez pas.

[[autodoc]] logging.get_logger

## Configurer les barres de progression

Les barres de progression sont un outil utile pour afficher des informations à l'utilisateur pendant qu'une tâche de longue durée est en cours d'exécution (par exemple
lors du téléchargement ou de l'upload de fichiers). `huggingface_hub` expose un wrapper [`~utils.tqdm`] pour afficher les barres de progression de manière
cohérente à travers la bibliothèque.

Par défaut, les barres de progression sont activées. Vous pouvez les désactiver globalement en définissant la variable d'environnement `HF_HUB_DISABLE_PROGRESS_BARS`.
Vous pouvez également les activer/désactiver en utilisant [`~utils.enable_progress_bars`] et
[`~utils.disable_progress_bars`]. Si définie, la variable d'environnement a priorité sur les helpers.

```py
>>> from huggingface_hub import snapshot_download
>>> from huggingface_hub.utils import are_progress_bars_disabled, disable_progress_bars, enable_progress_bars

>>> # Désactiver les barres de progression globalement
>>> disable_progress_bars()

>>> # La barre de progression ne sera pas affichée !
>>> snapshot_download("gpt2")

>>> are_progress_bars_disabled()
True

>>> # Réactiver les barres de progression globalement
>>> enable_progress_bars()
```

### Contrôle spécifique par groupe des barres de progression

Vous pouvez également activer ou désactiver les barres de progression pour des groupes spécifiques. Cela vous permet de gérer la visibilité des barres de progression de manière plus granulaire dans différentes parties de votre application ou bibliothèque. Lorsqu'une barre de progression est désactivée pour un groupe, tous les sous-groupes sous celui-ci sont également affectés sauf s'ils sont explicitement écrasés.

```py
# Désactiver les barres de progression pour un groupe spécifique
>>> disable_progress_bars("peft.foo")
>>> assert not are_progress_bars_disabled("peft")
>>> assert not are_progress_bars_disabled("peft.something")
>>> assert are_progress_bars_disabled("peft.foo")
>>> assert are_progress_bars_disabled("peft.foo.bar")

# Réactiver les barres de progression pour un sous-groupe
>>> enable_progress_bars("peft.foo.bar")
>>> assert are_progress_bars_disabled("peft.foo")
>>> assert not are_progress_bars_disabled("peft.foo.bar")

# Utiliser les groupes avec tqdm
# Pas de barre de progression pour `name="peft.foo"`
>>> for _ in tqdm(range(5), name="peft.foo"):
...     pass

# La barre de progression sera affichée pour `name="peft.foo.bar"`
>>> for _ in tqdm(range(5), name="peft.foo.bar"):
...     pass
100%|███████████████████████████████████████| 5/5 [00:00<00:00, 117817.53it/s]
```

### are_progress_bars_disabled

[[autodoc]] huggingface_hub.utils.are_progress_bars_disabled

### disable_progress_bars

[[autodoc]] huggingface_hub.utils.disable_progress_bars

### enable_progress_bars

[[autodoc]] huggingface_hub.utils.enable_progress_bars

## Configurer le Backend HTTP

<Tip>

Dans `huggingface_hub` v0.x, les requêtes HTTP étaient gérées avec `requests`, et la configuration se faisait via `configure_http_backend`. Maintenant que nous utilisons `httpx`, la configuration fonctionne différemment : vous devez fournir une fonction factory qui ne prend aucun argument et retourne un `httpx.Client`. Vous pouvez consulter [l'implémentation par défaut ici](https://github.com/huggingface/huggingface_hub/blob/v1.0-release/src/huggingface_hub/utils/_http.py) pour voir quels paramètres sont utilisés par défaut.

</Tip>


Dans certaines configurations, vous devrez peut-être contrôler comment les requêtes HTTP sont effectuées, par exemple lorsque vous travaillez derrière un proxy. La bibliothèque `huggingface_hub` vous permet de configurer cela globalement avec [`set_client_factory`]. Après configuration, toutes les requêtes vers le Hub utiliseront vos paramètres personnalisés. Puisque `huggingface_hub` s'appuie sur `httpx.Client` sous le capot, vous pouvez consulter la [documentation `httpx`](https://www.python-httpx.org/advanced/clients/) pour plus de détails sur les paramètres disponibles.

Si vous construisez une bibliothèque tierce et devez effectuer des requêtes directes vers le Hub, utilisez [`get_session`] pour obtenir un client `httpx` correctement configuré. Remplacez tous les appels directs `httpx.get(...)` par `get_session().get(...)` pour assurer un comportement correct.

[[autodoc]] set_client_factory

[[autodoc]] get_session

Dans de rares cas, vous pourriez vouloir fermer manuellement la session actuelle (par exemple, après une `SSLError` transitoire). Vous pouvez le faire avec [`close_session`]. Une nouvelle session sera automatiquement créée lors du prochain appel à [`get_session`].

Les sessions sont toujours fermées automatiquement lorsque le processus se termine.

[[autodoc]] close_session

Pour le code async, utilisez [`set_async_client_factory`] pour configurer un `httpx.AsyncClient` et [`get_async_session`] pour en récupérer un.

[[autodoc]] set_async_client_factory

[[autodoc]] get_async_session

<Tip>

Contrairement au client synchrone, le cycle de vie du client async n'est pas géré automatiquement. Utilisez un gestionnaire de contexte async pour le gérer correctement.

</Tip>

## Gérer les erreurs HTTP

`huggingface_hub` définit ses propres erreurs HTTP pour affiner la `HTTPError` levée par
`requests` avec des informations supplémentaires renvoyées par le serveur.

### Raise for status

[`~utils.hf_raise_for_status`] est destinée à être la méthode centrale pour "raise for status" à partir de n'importe quelle
requête effectuée vers le Hub. Elle encapsule la base `requests.raise_for_status` pour fournir
des informations supplémentaires. Toute `HTTPError` levée est convertie en `HfHubHTTPError`.

```py
import requests
from huggingface_hub.utils import hf_raise_for_status, HfHubHTTPError

response = requests.post(...)
try:
    hf_raise_for_status(response)
except HfHubHTTPError as e:
    print(str(e)) # message formaté
    e.request_id, e.server_message # détails retournés par le serveur

    # Compléter le message d'erreur avec des informations supplémentaires une fois qu'il est levé
    e.append_to_message("\n`create_commit` attend que le dépôt existe.")
    raise
```

[[autodoc]] huggingface_hub.utils.hf_raise_for_status

### Erreurs HTTP

Voici une liste des erreurs HTTP levées dans `huggingface_hub`.

#### HfHubHTTPError

`HfHubHTTPError` est la classe parente pour toute erreur HTTP HF Hub. Elle s'occupe de l'analyse
de la réponse du serveur et formate le message d'erreur pour fournir autant d'informations que possible
à l'utilisateur.

[[autodoc]] huggingface_hub.errors.HfHubHTTPError

#### RepositoryNotFoundError

[[autodoc]] huggingface_hub.errors.RepositoryNotFoundError

#### GatedRepoError

[[autodoc]] huggingface_hub.errors.GatedRepoError

#### RevisionNotFoundError

[[autodoc]] huggingface_hub.errors.RevisionNotFoundError

#### BadRequestError

[[autodoc]] huggingface_hub.errors.BadRequestError

#### EntryNotFoundError

[[autodoc]] huggingface_hub.errors.EntryNotFoundError

#### RemoteEntryNotFoundError

[[autodoc]] huggingface_hub.errors.RemoteEntryNotFoundError

#### LocalEntryNotFoundError

[[autodoc]] huggingface_hub.errors.LocalEntryNotFoundError

#### OfflineModeIsEnabled

[[autodoc]] huggingface_hub.errors.OfflineModeIsEnabled

## Télémétrie

`huggingface_hub` inclut un helper pour envoyer des données de télémétrie. Ces informations nous aident à déboguer les problèmes et à prioriser les nouvelles fonctionnalités.
Les utilisateurs peuvent désactiver la collecte de télémétrie à tout moment en définissant la variable d'environnement `HF_HUB_DISABLE_TELEMETRY=1`.
La télémétrie est également désactivée en mode hors ligne (c'est-à-dire lors de la définition de HF_HUB_OFFLINE=1).

Si vous êtes mainteneur d'une bibliothèque tierce, envoyer des données de télémétrie est aussi simple que de faire un appel à [`send_telemetry`].
Les données sont envoyées dans un thread séparé pour réduire autant que possible l'impact pour les utilisateurs.

[[autodoc]] utils.send_telemetry


## Validateurs

`huggingface_hub` inclut des validateurs personnalisés pour valider automatiquement les arguments de méthode.
La validation s'inspire du travail effectué dans [Pydantic](https://pydantic-docs.helpmanual.io/)
pour valider les type hints mais avec des fonctionnalités plus limitées.

### Décorateur générique

[`~utils.validate_hf_hub_args`] est un décorateur générique pour encapsuler
les méthodes qui ont des arguments suivant la nomenclature de `huggingface_hub`. Par défaut, tous
les arguments qui ont un validateur implémenté seront validés.

Si une entrée n'est pas valide, une [`~utils.HFValidationError`] est levée. Seule
la première valeur non valide lève une erreur et arrête le processus de validation.

Utilisation :

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
```

#### validate_hf_hub_args

[[autodoc]] utils.validate_hf_hub_args

#### HFValidationError

[[autodoc]] utils.HFValidationError

### Validateurs d'arguments

Les validateurs peuvent également être utilisés individuellement. Voici une liste de tous les arguments qui peuvent être
validés.

#### repo_id

[[autodoc]] utils.validate_repo_id

#### smoothly_deprecate_legacy_arguments

Pas exactement un validateur, mais exécuté également.

[[autodoc]] utils._validators.smoothly_deprecate_legacy_arguments
