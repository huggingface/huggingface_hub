<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Variables d'environnement

`huggingface_hub` peut être configuré en utilisant des variables d'environnement.

Si vous n'êtes pas familier avec le principe de variable d'environnement, voici des
articles assez générique sur ces dernières [sur macOS et Linux](https://linuxize.com/post/how-to-set-and-list-environment-variables-in-linux/)
et sur [Windows](https://phoenixnap.com/kb/windows-set-environment-variable).

Cette page vous guidera à travers toutes les variables d'environnement spécifiques à
`huggingface_hub` et leur signification.

## Les variables génériques

### HF_INFERENCE_ENDPOINT

Pour configurer l'url de base de l'api d'inférence, vous aurez peut-être besoin de définir
cette variable si votre organisation pointe vers un gateway d'API plutôt que directement vers
l'API d'inférence.

Par défaut, l'endpoint sera `"https://api-inference.huggingface.com"`.

### HF_HOME

Pour configurer le chemin vers lequel `huggingface_hub` enregistrera de la donnée par
défaut. En particulier, votre token d'authentification et le cache seront enregistrés
dans ce dossier.

Par défaut, ce chemin sera `"~/.cache/huggingface"` sauf si [XDG_CACHE_HOME](#xdgcachehome)
est définie.

### HF_HUB_CACHE

Pour configurer le chemin vers lequels les dépôts du Hub seront mis en cache en local
(modèles, datasets et spaces).

Par défaut, ce sera `"$HF_HOME/hub"` (i.e. `"~/.cache/huggingface/hub"` par défaut).

### HF_ASSETS_CACHE

Pour configurer le chemin vers lequel les [assets](../guides/manage-cache#caching-assets) créés
par des librairies seront mis en cache en local. Ces assets peuvent être de la donnée pré-traitée,
des fichiers téléchargés depuis GitHub, des logs,...

Par défaut, le chemin sera `"$HF_HOME/assets"` (i.e. `"~/.cache/huggingface/assets"` par défaut).

### HF_TOKEN

Pour configurer le token d'authentification qui permet de vous authentifier sur le Hub.
Si définie, cette valeur écrasera le token enregistré sur la machine (dans `"$HF_HOME/token"`).

Consultez [la référence aux connexions](../package_reference/login) pour plus de détails.

### HF_HUB_VERBOSITY

Définissez le niveau de verbosité du logger `huggingface_hub`. La variable doit
être choisie parmi `{"debug", "info", "warning", "error", "critical"}`.

Par défaut, la variable sera `"warning"`.

Pour plus de détails, consultez [la référence aux connexions](../package_reference/utilities#huggingface_hub.utils.logging.get_verbosity).

### HF_HUB_LOCAL_DIR_AUTO_SYMLINK_THRESHOLD

Valeur entière définissant en dessous de quelle taille un fichier est considéré comme "petit". Lors du téléchargement
de fichiers vers un chemin local, les petits fichiers seront dupliqués pour faciliter l'expérience utilisateur là où
les fichiers plus gros n'auront qu'un symlink vers eux pour conserver de l'espace sur le disque.

Pour plus de détails, consultez le [guide de téléchargement](../guides/download#download-files-to-local-folder).

### HF_HUB_ETAG_TIMEOUT

Valeur entière définissant le nombre de secondes pendant lesquelles il faut attendre la réponse du serveur lors de l'affichage des dernières metadonnées depuis un dépôt avant de télécharger un fichier. En cas de timeout, alors, par défaut, `huggingface_hub` se limitera aux fichiers en cache en local. Définir une valeur plus faible accélère le workflow pour des machines avec une connexion lente qui ont déjà des fichiers en cache. Une plus grande valeur garanti l'appel aux métadonnées pour réussir dans plus de cas. Par défaut, la valeur est de 10s.

### HF_HUB_DOWNLOAD_TIMEOUT

Valeur entière pour définir le nombre de secondes durant lesquelles il faut attendre la réponse du serveur lors du téléchargement d'un fichier. Si la requête tiemout, une TimeoutError sera levée. Définir une valeur grande est bénéfique sur une machine qui a une connexion lente. Une valeur plus faible fait échouer le process plus vite dans le cas d'un réseau complexe. Par défaut, la valeur est de 10s.

## Valeures booléennes

Les variables d'environnement suivantes attendes une valeur booléenne. La variable sera
considérées comme `True` si sa valeur fait partie de la liste`{"1", "ON", "YES", "TRUE"}`.
Toute autre valeur (ou undefined) sera considérée comme `False`.

### HF_HUB_OFFLINE

Si définie, aucun appel HTTP ne sera fait lors de l'ajout de fichiers. Seuls les fichiers
qui sont déjà en cache seront ajoutés. C'est utile si votre réseau est lent et que vous
vous en fichez d'avoir absolument la dernière version d'un fichier.

**Note:** Même si la dernière version d'un fichier est en cache, l'appel de `hf_hub_download`
lancera quand même une requête HTTP pour vérifier qu'une nouvelle version est disponible.
Définir `HF_HUB_OFFLINE=1` évitera cet appel ce qui peut accélérer votre temps de chargement.

### HF_HUB_DISABLE_IMPLICIT_TOKEN

L'authentification n'est pas obligatoire pour toutes les requêtes vers le Hub. Par
exemple, demander des détails sur le modèle `"gpt2"` ne demande pas nécessairement
d'être authentifié. Cependant, si un utilisateur est [connecté](../package_reference/login),
le comportement par défaut sera de toujours envoyer le token pour faciliter l'expérience
utilisateur (cela évite d'avoir une erreur 401 Unauthorized) lors de l'accès de dépôt
privés ou protégé. Vous pouvez supprimer ce comportement en définissant `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`.
Dans ce cas, le token ne sera envoyé que pour des appels de type "write-access" (par exemple, pour créer un commit).

**Note:** supprimer l'envoi implicit de token peut avoir des effets secondaires bizarres.
Par exemple, si vous voulez lister tous les modèles sur le Hub, vos modèles privés ne
seront pas listés. Vous auriez besoin d'un argument explicite `token=True` dans votre
script.

### HF_HUB_DISABLE_PROGRESS_BARS

Pour les tâches longues, `huggingface_hub` affiche une bar de chargement par défaut (en utilisant tqdm).
Vous pouvez désactiver toutes les barres de progression d'un coup en définissant
`HF_HUB_DISABLE_PROGRESS_BARS=1`.

### HF_HUB_DISABLE_SYMLINKS_WARNING

Si vous avez une machine Windows, il est recommandé d'activer le mode développeur ou de
faire tourner `huggingface_hub` en mode admin. Sinon, `huggingface_hub` ne ser pas capable
de créer des symlinks dans votre système de cache. Vous serez capables d'exécuter n'importe
quel script, mais votre expérience utilisateur sera moins bonne vu que certains gros fichiers
pourraient être dupliqués sur votre disque dur. Un message d'avertissement vous préviendra
de ce type de comportement. Définissez `HF_HUB_DISABLE_SYMLINKS_WARNING=1`, pour désactiver
cet avertissement.

Pour plus de détails, consultez [les limitations du cache](../guides/manage-cache#limitations).

### HF_HUB_DISABLE_EXPERIMENTAL_WARNING

Certaines fonctionnalités de `huggingface_hub` sont expérimentales, cela signfie que vous pouvez les utliiser mais
nous ne garantissons pas  qu'elles seront maintenues plus tard. En particulier, nous mettrons peut-être à jour les 
API ou le comportement de telles fonctionnalités sans aucun cycle de deprecation. Un message d'avertissement sera
affiché lorsque vous utilisez une fonctionnalités expérimentale pour vous l'indiquer. Si vous n'êtes pas dérangé par
le fait de devoir debug toute erreur potentielle liée à l'usage d'une fonctionnalité expérimentale, vous pouvez
définir `HF_HUB_DISABLE_EXPERIMENTAL_WARNING=1` pour désactiver l'avertissement.

Si vous utilisez une fonctoinnalité expérimental, faites le nous savoir! Votre retour peut nous aider à l'améliorer.

### HF_HUB_DISABLE_TELEMETRY

Par défaut, des données sont collectées par les librairies HF (`transformers`, `datasets`, `gradio`,..) pour gérer l'utilisation et les problèmes de débug et pour aider à définir les fonctionnalités prioritaires. Chaque librairie définit sa propre politique (i.e. les cas d'usage sur lesquels les données eront collectées), mais l'implémentation principale se passe dans `huggingface_hub` (consultez [`send_telemetry`]).

Vous pouvez définir `HF_HUB_DISABLE_TELEMETRY=1` en tant que variable d'environnement pour désactiver la télémétrie.

### HF_HUB_ENABLE_HF_TRANSFER

Définissez cette valeur à `True` pour des upload et des téléchargements plus rapides depuis le Hub en utilisant `hf_transfer`.

Par défaut, `huggingface_hub` utilise les fonctions basées sur Python `requests.get` et `requests.post`. Même si ces fonctions sont fiables et assez versatiles, elle pourrait ne pas être le choix le plus efficace pour les machines avec une bande passante large. [`hf_transfer`](https://github.com/huggingface/hf_transfer) est un package basé sur Rust développé pour maximiser la bande passante utilisée en divisant les gros fichier en des parties plus petites et les transférer toutes simultanément en utilisant plusieurs threads. Cette approche peut potentiellement doubler la vitesse de transfert. Pour utiliser `hf_transfer`, vous devez l'installer séparément [de PyPI](https://pypi.org/project/hf-transfer/) et définir `HF_HUB_ENABLE_HF_TRANSFER=1` en tant que variable d'environnement.

N'oubliez pas que l'utilisation d'`hf_transfer` a certaines limitations. Vu qu'elle n'est pas purement basé sur Python, le debug d'erreurs peut s'avérer plus compliqué. De plus, `hf_transfer` n'est pas totues les fonctionnalités user-friendly telles que le téléchargement reprenables et les proxys. Ces omissions sont intentionnelles et permettent de maintenir la simplicité et la vitesse de la logique Rust. Par conséquent, `hf_transfer` n'est pas activée par défaut dans `huggingface_hub`.

## Variables d'environnement deprecated

Afin de standardiser toutes les variables d'environnement dans l'écosystème Hugging Face, certaines variable ont été marquée comme deprecated. Même si elle fonctionnent toujours, elles ne sont plus prioritaires par rapport à leur remplacements. La table suivante liste les variables d'environnement deprecated et leurs alternatives respectives:


| Variable deprecated | Alternative |
| --- | --- |
| `HUGGINGFACE_HUB_CACHE` | `HF_HUB_CACHE` |
| `HUGGINGFACE_ASSETS_CACHE` | `HF_ASSETS_CACHE` |
| `HUGGING_FACE_HUB_TOKEN` | `HF_TOKEN` |
| `HUGGINGFACE_HUB_VERBOSITY` | `HF_HUB_VERBOSITY` |

## Depuis un outil extérieur

Certaines variables d'environnement ne sont pas spécifiques à `huggingface_hub` mais sont tout de même prises en compte
lorsqu'elles sont définies.

### NO_COLOR

Valeur booléenne, lorsque définie à `True`, l'outil `huggingface-cli` n'affichera
aucune couleur ANSI.
Consultez [no-color.org](https://no-color.org/).

### XDG_CACHE_HOME

Utilisé uniqueemnt si `HF_HOME` n'est pas défini!

C'est la manière par défaut de configurer l'endroit où [les données en cache non essentielles devraient être écrites](https://wiki.archlinux.org/title/XDG_Base_Directory) sur les machines linux.

Si `HF_HOME` n'est pas définie, le chemin par défaut sera `"$XDG_CACHE_HOME/huggingface"`
aulieu de `"~/.cache/huggingface"`.
