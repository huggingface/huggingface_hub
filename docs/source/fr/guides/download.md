<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Télécharger des fichiers du Hub

La librairie `huggingface_hub` fournit des fonctions pour télécharger des fichiers depuis
les dépôts stockés sur le Hub. Vous pouvez utiliser ces fonctions directement ou les intégrer
dans votre propre librairie, pour rendre l'intéraction entre vos utilisateurs et le Hub
plus simple. Ce guide vous montrera comment:

* Télécharger et mettre en cache un fichier
* Télécharger et mettre en cache un dépôt entier
* Télécharger des fichiers dans un dossier local

## Télécharger un fichier

La fonction [`hf_hub_download`] est la fonction principale pour télécharger des fichiers du Hub.
Elle télécharge le fichier, le met en cache sur le disque (en prenant en compte les versions)
et retourne le chemin vers le fichier local téléchargé.

<Tip>

Le chemin retourné est un pointeur vers le cache local HF. Par conséquent, il est important de ne pas modifier le fichier
pour éviter de corrompre le cache. Si vous voulez en apprendre plus sur la manière dont les fichiers sont mis en cache,
consultez notre [guide dédié au cache](./manage-cache).

</Tip>

### Télécharger la dernière version

Sélectionnez le fichier à télécharger en utilisant les paramètres `repo_id`, `repo_type` et `filename`. Par défaut,
le fichier sera considéré comme appartenant à un dépôt contenant des objets de type `model`.

```python
>>> from huggingface_hub import hf_hub_download
>>> hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json")
'/root/.cache/huggingface/hub/models--lysandre--arxiv-nlp/snapshots/894a9adde21d9a3e3843e6d5aeaaf01875c7fade/config.json'

# Télécharge un dataset
>>> hf_hub_download(repo_id="google/fleurs", filename="fleurs.py", repo_type="dataset")
'/root/.cache/huggingface/hub/datasets--google--fleurs/snapshots/199e4ae37915137c555b1765c01477c216287d34/fleurs.py'
```

### Télécharger une version spécifique

Par défaut, la dernière version de la branche `main` est téléchargée. Cependant, dans certains cas, vous aurez besoin
de télécharger un fichier ayant une version particulière (i.e. d'une branche spécifique, une pull request, un tag,
ou un hash de commit).
Pour ce faire, utilisez le paramètre `revision`:

```python
# Télécharge à partir du tag `v1.0`
>>> hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json", revision="v1.0")

# Télécharge à partir de la branche `test-branch`
>>> hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json", revision="test-branch")

# Télécharge à partir de la pull request #3
>>> hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json", revision="refs/pr/3")

# Télécharge à partir d'un hash de commit spécifique
>>> hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json", revision="877b84a8f93f2d619faa2a6e514a32beef88ab0a")
```

**Note:** Lorsque vous utilisez le hash de commit, vous devez renseigner le hash complet et pas le hash de commit à 7 caractères.

### Générer un URL de téléchargement

Si vous voulez générer l'URL utilisé pour télécharger un fichier depuis un dépôt, vous pouvez utiliser [`hf_hub_url`]
qui renvoie un URL. Notez que cette méthode est utilisée en arrière plan par  [`hf_hub_download`].

## Télécharger un dépôt entier

[`snapshot_download`] télécharge un dépôt entier à une révision donnée. Cette méthode utilise en arrière-plan
[`hf_hub_download`] ce qui signifie que tous les fichiers téléchargés sont aussi mis en cache sur votre disque en local.
Les téléchargements sont faits en parallèle pour rendre le processus plus rapide.

Pour télécharger un dépôt entier, passez simplement le `repo_id` et le `repo_type`:

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="lysandre/arxiv-nlp")
'/home/lysandre/.cache/huggingface/hub/models--lysandre--arxiv-nlp/snapshots/894a9adde21d9a3e3843e6d5aeaaf01875c7fade'

# Ou pour un dataset
>>> snapshot_download(repo_id="google/fleurs", repo_type="dataset")
'/home/lysandre/.cache/huggingface/hub/datasets--google--fleurs/snapshots/199e4ae37915137c555b1765c01477c216287d34'
```

[`snapshot_download`]  télécharge la dernière révision par défaut. Si vous voulez une révision spécifique, utilisez
le paramètre `revision`:

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="lysandre/arxiv-nlp", revision="refs/pr/1")
```

### Filtrer les fichiers à télécharger

[`snapshot_download`] offre une manière simple de télécharger un dépôt. Cependant, vous ne voudrez peut être pas
télécharger tout le contenu d'un dépôt à chaque fois. Par exemple, vous n'aurez peut-être pas envie de télécharger
tous les fichiers `.bin` si vous savez que vous utiliserez uniquement les poids du `.safetensors`. Vous pouvez
faire ceci en utilisant les paramètres `allow_patterns` et `ignore_patterns`.

Ces paramètres acceptent un pattern ou une liste de patterns. Les patterns sont des wildcards standards, comme précisé
[ici](https://tldp.org/LDP/GNU-Linux-Tools-Summary/html/x11655.htm). Le matching de pattern utilise [`fnmatch`](https://docs.python.org/3/library/fnmatch.html).

Par exemple, vous pouvez utiliser `allow_patterns` pour ne télécharger que les fichiers de configuration JSON:

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="lysandre/arxiv-nlp", allow_patterns="*.json")
```

A l'opposé, `ignore_patterns` empêche certains fichiers d'être téléchargés. L'exemple
suivant ignore les fichiers ayant pour extension `.msgpack` et `.h5`:

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="lysandre/arxiv-nlp", ignore_patterns=["*.msgpack", "*.h5"])
```

Enfin, vous pouvez combiner les deux pour filtrer avec précision vos téléchargements. voici un exemple pour télécharger
tous les fichiers en .md et en .json à l'exception de `vocab.json`

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="gpt2", allow_patterns=["*.md", "*.json"], ignore_patterns="vocab.json")
```

## Télécharger un ou plusieurs fichier(s) vers un dossier local

La manière recommandée (et utilisée par défaut) pour télécharger des fichiers depuis les Hub est d'utiliser
le [cache-system](./manage-cache). Vous pouvez définir le chemin vers votre cache en définissant le
paramètre `cache_dir` (dans [`hf_hub_download`] et [`snapshot_download`]).

Toutefois, dans certains cas, vous aurez besoin de télécharger des fichiers et de les déplacer dans un dossier spécifique.
C'est une pratique utile pour créer un workflow plus proche de ce qu'on peut retrouver avec les commande `git`. Vous
pouvez faire ceci en utilisant les paramètres `local_dir` et `local_dir_use_symlinks`:
- `local_dir` doit être un chemin vers un dossier de votre système. Les fichiers téléchargés garderont la même structure
de fichier que dans le dépôt. Par exemple, si `filename="data/train.csv"` et `local_dir="path/to/folder"`, alors le
chemin renvoyé sera `"path/to/folder/data/train.csv"`.
- `local_dir_use_symlinks` renseigne comment le fichier doit être enregistré sur votre dossier local.
  - Le comportement par défaut (`"auto"`), dupliquera les fichiers peu volumineux (<5MB) et utilisera les symlinks pour
    les fichiers plus gros. Les symlinks permettent d'optimiser à la fois la bande passante et l'utilisation du disque.
    Cependant, éditer manuellement un fichier sous symlink pourrait corrompre le cache, d'où la duplication pour des
    petits fichiers. Le seuil de 5MB peut être configuré avec la variable d'environnement`HF_HUB_LOCAL_DIR_AUTO_SYMLINK_THRESHOLD`.
  - Si `local_dir_use_symlinks=True` est passé, alors tous les fichiers seront sous symlink pour une utilisation
    optimal de l'espace disque. C'est par exemple utile lors du téléchargement d'un dataset très volumineux contenant
    des milliers de petits fichiers.
  - Enfin, si vous ne voulez pas utiliser de symlink du tout, vous pouvez les désactier (`local_dir_use_symlinks=False`).
    Le chemin du cache sera toujours utilisé afin de vérifier si le fichier est déjà en cache ou pas. Si ce dernier
    n'est pas déjà en cache, il sera téléchargé et déplacé directement vers le chemin local. Ce qui signifie que si
    vous avez besoin de le réutiliser ailleurs, il sera **retéléchargé**

Voici une table qui résume les différentes options pour vous aider à choisir les paramètres qui collent le mieux à votre situation.

<!-- Generated with https://www.tablesgenerator.com/markdown_tables -->
| Paramètre | Fichier déjà en cahce | Chemin renvoyé | Peut-on lire le chemin | Pouvez vous sauvegarder le chemin | Bande passante optimisée | Utilisation du disque optimisée |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `local_dir=None` |  | symlink en cache | ✅ | ❌<br>_(sauvegarder corromprait le cache)_ | ✅ | ✅ |
| `local_dir="path/to/folder"`<br>`local_dir_use_symlinks="auto"` |  | fichier ou symlink dans un dossier | ✅ | ✅ _(pour les petits fichiers)_ <br> ⚠️ _(pour les gros fichiers, ne resolve pas le path avant l'enregistrement)_ | ✅ | ✅ |
| `local_dir="path/to/folder"`<br>`local_dir_use_symlinks=True` |  | symlink dans un dossier | ✅ | ⚠️<br>_(ne resolve pas le paht avant l'enregistrement)_ | ✅ | ✅ |
| `local_dir="path/to/folder"`<br>`local_dir_use_symlinks=False` | Non | fichier dans un dossier | ✅ | ✅ | ❌<br>_(en cas de re-run, le fichier est retéléchargé)_ | ⚠️<br>(plusieurs copies si lancé dans plusieurs dossiers) |
| `local_dir="path/to/folder"`<br>`local_dir_use_symlinks=False` | oui | fichier dans un dossier | ✅ | ✅ | ⚠️<br>_(le fichier doit être mis en cache d'abord)_ | ❌<br>_(le fichier est dupliqué)_ |

**Note:** si vous utilisez une machien Windows, vous devez activer le mode développeur ou lancer `huggingface_hub` en tant qu'administrateur pour activer les syymlinks. Consultez la section [limitations du cache](../guides/manage-cache#limitations)

## Télécharger depuis le CLI

Vous pouvez utiliser la commande `huggingface-cli download` depuis un terminal pour télécharger directement des
fichiers du Hub. En interne, cette commande utilise les même helpers [`hf_hub_download`] et [`snapshot_download`]
décrits ci-dessus et affiche le chemin renvoyé dans le terminal.

```bash
>>> huggingface-cli download gpt2 config.json
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10/config.json
```

Vous pouvez télécharger plusieurs fichiers d'un coup, ce qui affiche une barre de chargement et renvoie le chemin de
la snapshot dans lequel les fichiers sont localisés.

```bash
>>> huggingface-cli download gpt2 config.json model.safetensors
Fetching 2 files: 100%|████████████████████████████████████████████| 2/2 [00:00<00:00, 23831.27it/s]
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10
```

Pour plus de détails sur la commande download du CLI, veuillez consulter le [guide CLI](./cli#huggingface-cli-download).

## Téléchargements plus rapides

Si vous utilisez une machine avec une bande passante plus large, vous pouvez augmenter votre vitesse de téléchargement en utilisant [`hf_transfer`],
une librairie basée sur Rust développée pour accélérer le transfer de fichiers avec le Hub. Pour l'activer, installez le package (`pip install hf_transfer`) et définissez set `HF_HUB_ENABLE_HF_TRANSFER=1` en tant que variable d'environnement 

<Tip>

Les barres de chargement ne fonctionnent avec `hf_transfer` qu'à partir de la version `0.1.4`. Mettez à jour la version (`pip install -U hf_transfer`)
si vous comptez utiliser cette librairie.

</Tip>

<Tip warning={true}>

`hf_transfer` est un outil très puissant! Il a été testé et est prêt à être utilisé en production, mais il lui manque certaines fonctionnalités user friendly, telles que la gestion d'erreurs avancée ou les proxys. Pour plus de détails, consultez cette [section](https://huggingface.co/docs/huggingface_hub/hf_transfer).

</Tip>