<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Migration vers huggingface_hub v1.0

La version v1.0 est une étape majeure pour la bibliothèque `huggingface_hub`. Elle marque un point de stabilité de l'API et un niveau de maturité. Nous avons apporté plusieurs améliorations et changements majeurs pour rendre la bibliothèque plus robuste et plus facile à utiliser.

Ce guide est destiné à vous aider à migrer votre code existant vers la nouvelle version. Si vous avez des questions ou des commentaires, n'hésitez pas à [ouvrir une issue sur GitHub](https://github.com/huggingface/huggingface_hub/issues).

## Python 3.9+

`huggingface_hub` nécessite maintenant Python 3.9 ou plus. Python 3.8 n'est plus supporté.

## Migration vers HTTPX

La bibliothèque `huggingface_hub` utilise désormais [`httpx`](https://www.python-httpx.org/) au lieu de `requests` pour les requêtes HTTP. Ce changement a été effectué pour améliorer les performances et pour supporter les requêtes synchrones et asynchrones de la même manière. Nous avons donc abandonné les dépendances `requests` et `aiohttp`.

### Changements majeurs

Il s'agit d'un changement majeur qui affecte toute la bibliothèque. Bien que nous ayons essayé de rendre ce changement aussi transparent que possible, vous pourriez avoir besoin de mettre à jour votre code dans certains cas. Voici une liste des changements majeurs introduits :

- **Configuration du proxy** : Les proxies "par méthode" ne sont plus supportés. Les proxies doivent être configurés globalement en utilisant les variables d'environnement `HTTP_PROXY` et `HTTPS_PROXY`.
- **Backend HTTP personnalisé** : La fonction `configure_http_backend` a été supprimée. Vous devez maintenant utiliser [`set_client_factory`] et [`set_async_client_factory`] pour configurer les clients HTTP.
- **Gestion des erreurs** : Les erreurs HTTP n'héritent plus de `requests.HTTPError`, mais de `httpx.HTTPError`. Nous recommandons de capturer `huggingface_hub.HfHubHttpError` qui est une sous-classe de `requests.HTTPError` en v0.x et de `httpx.HTTPError` en v1.x. Capturer depuis l'erreur `huggingface_hub` garantit que votre code est compatible avec les anciennes et nouvelles versions de la bibliothèque.
- **SSLError** : `httpx` n'a pas le concept de `SSLError`. C'est maintenant une `httpx.ConnectError` générique.
- **`LocalEntryNotFoundError`** : Cette erreur n'hérite plus de `HTTPError`. Nous définissons maintenant une `EntryNotFoundError` (nouvelle) qui est héritée par [`LocalEntryNotFoundError`] (si le fichier n'est pas trouvé dans le cache local) et [`RemoteEntryNotFoundError`] (si le fichier n'est pas trouvé dans le dépôt sur le Hub). Seule l'erreur distante hérite de `HTTPError`.
- **`InferenceClient`** : Le `InferenceClient` peut maintenant être utilisé comme gestionnaire de contexte. Ceci est particulièrement utile lors du streaming de tokens depuis un modèle de langage pour s'assurer que la connexion est correctement fermée.
- **`AsyncInferenceClient`** : Le paramètre `trust_env` a été supprimé du constructeur de `AsyncInferenceClient`. Les variables d'environnement sont traitées par défaut par `httpx`. Si vous ne souhaitez explicitement pas faire confiance à l'environnement, vous devez le configurer avec [`set_client_factory`].

Pour plus de détails, vous pouvez consulter la [PR #3328](https://github.com/huggingface/huggingface_hub/pull/3328) qui a introduit `httpx`.

### Pourquoi `httpx` ?

La migration de `requests` vers `httpx` apporte plusieurs améliorations clés qui améliorent les performances, la fiabilité et la maintenabilité de la bibliothèque :

**Sécurité des threads et réutilisation des connexions** : `httpx` est thread-safe par conception, nous permettant de réutiliser en toute sécurité le même client sur plusieurs threads. Cette réutilisation des connexions réduit le surcoût de l'établissement de nouvelles connexions pour chaque requête HTTP, améliorant les performances en particulier lors de requêtes fréquentes vers le Hub.

**Support HTTP/2** : `httpx` fournit un support natif HTTP/2, ce qui offre une meilleure efficacité lors de l'exécution de plusieurs requêtes vers le même serveur (exactement notre cas d'usage). Cela se traduit par une latence plus faible et une consommation de ressources réduite par rapport à HTTP/1.1.

**API Sync/Async unifiée** : Contrairement à notre configuration précédente avec des dépendances séparées `requests` (sync) et `aiohttp` (async), `httpx` fournit des clients synchrones et asynchrones avec un comportement identique. Cela garantit que `InferenceClient` et `AsyncInferenceClient` ont des fonctionnalités cohérentes et élimine les différences de comportement qui existaient auparavant entre les deux implémentations.

**Gestion améliorée des erreurs SSL** : `httpx` gère les erreurs SSL plus facilement, rendant le débogage des problèmes de connexion plus facile et plus fiable.

**Architecture pérenne** : `httpx` est activement maintenu et conçu pour les applications Python modernes. En revanche, `requests` est en "maintenance" et ne recevra pas de mises à jour majeures comme des améliorations de sécurité, des threads ou du support HTTP/2.

**Meilleure gestion des variables d'environnement** : `httpx` fournit une gestion plus cohérente des variables d'environnement dans les contextes sync et async, éliminant les incohérences. Là où `requests` lisait les variables d'environnement locales par défaut.

La transition vers `httpx` positionne `huggingface_hub` avec un backend HTTP moderne, efficace et maintenable. Bien que la plupart des utilisateurs devraient bénéficier d'un fonctionnement identique, les améliorations sous-jacentes offrent de meilleures performances et une meilleure fiabilité pour toutes les interactions avec le Hub.

## `hf_transfer`

Maintenant que tous les dépôts sur le Hub sont activés pour Xet et que `hf_xet` est la façon par défaut de télécharger/uploader des fichiers, nous avons supprimé le support du package optionnel `hf_transfer`. La variable d'environnement `HF_HUB_ENABLE_HF_TRANSFER` est donc supprimée. Utilisez plutôt [`HF_XET_HIGH_PERFORMANCE`](../package_reference/environment_variables.md).

## Classe `Repository`

La classe `Repository` a été supprimée dans la v1.0. C'était un wrapper autour du CLI `git` pour gérer les dépôts. Vous pouvez toujours utiliser `git` directement dans le terminal, mais l'approche recommandée est d'utiliser l'API basée sur HTTP dans la bibliothèque `huggingface_hub` pour une expérience plus fluide, en particulier lors de la manipulation de fichiers volumineux.

Voici une correspondance entre la classe `Repository` historique et la nouvelle classe `HfApi` :

| Méthode `Repository`                       | Méthode `HfApi`                                       |
| ------------------------------------------ | ----------------------------------------------------- |
| `repo.clone_from`                          | `snapshot_download`                                   |
| `repo.git_add` + `git_commit` + `git_push` | [`upload_file`], [`upload_folder`], [`create_commit`] |
| `repo.git_tag`                             | `create_tag`                                          |
| `repo.git_branch`                          | `create_branch`                                       |

## Classe `HfFolder`

`HfFolder` était utilisé pour gérer le token d'accès utilisateur. Utilisez [`login`] pour sauvegarder un nouveau token, [`logout`] pour le supprimer et [`whoami`] pour vérifier l'utilisateur connecté au token actuel. Pour finir, utilisez [`get_token`] pour récupérer le token de l'utilisateur.

## Classe `InferenceApi`

`InferenceApi` était une classe pour interagir avec l'API d'inférence. Il est maintenant recommandé d'utiliser la classe [`InferenceClient`] à la place.

## Autres fonctionnalités dépréciées

Certaines méthodes et paramètres ont été supprimés dans la v1.0. Ceux listés ci-dessous ont déjà été dépréciés avec un message d'avertissement dans la v0.x.

- `constants.hf_cache_home` a été supprimé. Veuillez utiliser `HF_HOME` à la place.
- Le paramètre `use_auth_token` a été supprimé de toutes les méthodes. Veuillez utiliser `token` à la place.
- La méthode `get_token_permission` a été supprimée.
- La méthode `update_repo_visibility` a été supprimée. Veuillez utiliser `update_repo_settings` à la place.
- Le paramètre `is_write_action` a été supprimé de `build_hf_headers` ainsi que `write_permission` de `login`. Le concept de "permissions" a été supprimé, il n'était plus pertinent maintenant que les tokens avec des permissions sont existants.
- Le paramètre `new_session` dans `login` a été renommé en `skip_if_logged_in` pour plus de clarté.
- Les paramètres `resume_download`, `force_filename` et `local_dir_use_symlinks` ont été supprimés de `hf_hub_download` et `snapshot_download`.
- Les paramètres `library`, `language`, `tags` et `task` ont été supprimés de `list_models`.

## Commandes CLI de cache

La gestion du cache du CLI a été repensée pour suivre un workflow inspiré de Docker. Le `huggingface-cli` déprécié a été supprimé, `hf` (introduit dans la v0.34) le remplace avec un CLI ressource-action plus clair.
Les commandes historiques `hf cache scan` et `hf cache delete` sont également supprimées dans la v1.0 et sont remplacées par le nouveau trio ci-dessous :

- `hf cache ls` liste les entrées du cache avec une sortie, JSON ou CSV. Utilisez `--revisions` pour inspecter les révisions individuelles, ajoutez des expressions `--filter` telles que `size>1GB` ou `accessed>30d`, et combinez-les avec `--quiet` lorsque vous n'avez besoin que des identifiants.
- `hf cache rm` supprime les entrées de cache sélectionnées. Passez un ou plusieurs IDs de dépôt (par exemple `model/bert-base-uncased`) ou des hashes de révision. Vous pouvez ajouter optionnellement `--dry-run` pour prévisualiser ou `--yes` pour ignorer la confirmation. 
- `hf cache prune` Supprime les révisions non référencées.Vous pouvez Ajouter `--dry-run` ou `--yes` de la même manière qu'avec `hf cache rm`.

Enfin, l'extra `[cli]` a été supprimé - Le CLI est maintenant livré avec le package `huggingface_hub` de base.

## Support de TensorFlow et Keras 2.x

Tout le code et les dépendances liés à TensorFlow ont été supprimés dans la v1.0. Cela inclut les changements majeurs suivants :

- `huggingface_hub[tensorflow]` n'est plus une dépendance extra supportée
- Les fonctions utilitaires `split_tf_state_dict_into_shards` et `get_tf_storage_size` ont été supprimées.
- Les versions `tensorflow`, `fastai` et `fastcore` ne sont plus incluses dans les en-têtes intégrés.

L'intégration Keras 2.x a également été supprimée. Cela inclut la classe `KerasModelHubMixin` et les utilitaires `save_pretrained_keras`, `from_pretrained_keras` et `push_to_hub_keras`. Keras 2.x est une bibliothèque ancienne et non maintenue. L'approche recommandée est d'utiliser Keras 3.x qui est étroitement intégré avec le Hub (c'est-à-dire qu'il contient des méthodes intégrées pour charger/pousser vers le Hub). Si vous souhaitez toujours travailler avec Keras 2.x, vous devez rétrograder `huggingface_hub` vers la version v0.x.

## Valeurs de retour de `upload_file` et `upload_folder`

Les fonctions [`upload_file`] et [`upload_folder`] retournent maintenant l'URL du commit créé sur le Hub. Auparavant, elles retournaient l'URL du fichier ou du dossier. Ceci est pour s'aligner avec la valeur de retour de [`create_commit`], [`delete_file`] et [`delete_folder`].
