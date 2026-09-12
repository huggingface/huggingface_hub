<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Télécharger des fichiers depuis le Hub

La bibliothèque `huggingface_hub` fournit des fonctions pour télécharger des fichiers depuis les dépôts
stockés sur le Hub. Vous pouvez utiliser ces fonctions indépendamment ou les intégrer dans votre
propre bibliothèque, Dans tous les cas, cela rend plus pratique l'interaction avec le Hub. Ce
guide vous montrera comment :

* Télécharger et mettre en cache un seul fichier.
* Télécharger et mettre en cache un dépôt entier.
* Télécharger des fichiers dans un dossier local.

## Télécharger un seul fichier

La fonction [`hf_hub_download`] est la fonction principale pour télécharger des fichiers depuis le Hub.
Elle télécharge le fichier distant, le met en cache sur le disque et retourne son chemin de fichier local.

> [!TIP]
> Le chemin de fichier retourné est un pointeur vers le cache local HF. Par conséquent, il est important de ne pas modifier le fichier pour éviter
> d'avoir un cache corrompu. Si vous souhaitez en savoir plus sur la façon dont les fichiers sont mis en cache, veuillez consulter notre
> [guide de mise en cache](./manage-cache).

### Depuis la dernière version

Sélectionnez le fichier à télécharger en utilisant les paramètres `repo_id`, `repo_type` et `filename`. Par défaut, le fichier sera
considéré comme faisant partie d'un dépôt `model`.

```python
>>> from huggingface_hub import hf_hub_download
>>> hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json")
'/root/.cache/huggingface/hub/models--lysandre--arxiv-nlp/snapshots/894a9adde21d9a3e3843e6d5aeaaf01875c7fade/config.json'

# Télécharger depuis un jeu de données
>>> hf_hub_download(repo_id="google/fleurs", filename="fleurs.py", repo_type="dataset")
'/root/.cache/huggingface/hub/datasets--google--fleurs/snapshots/199e4ae37915137c555b1765c01477c216287d34/fleurs.py'
```

### Depuis une version spécifique

Par défaut, la dernière version de la branche `main` est téléchargée. Cependant, dans certains cas, vous souhaitez télécharger un fichier
à une version particulière (par exemple depuis une branche spécifique, une PR, un tag ou un hash de commit).
Pour ce faire, utilisez le paramètre `revision` :

```python
# Télécharger depuis le tag `v1.0`
>>> hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json", revision="v1.0")

# Télécharger depuis la branche `test-branch`
>>> hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json", revision="test-branch")

# Télécharger depuis la Pull Request #3
>>> hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json", revision="refs/pr/3")

# Télécharger depuis un hash de commit spécifique
>>> hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json", revision="877b84a8f93f2d619faa2a6e514a32beef88ab0a")
```

**Note :** Lors de l'utilisation du hash de commit, il doit s'agir du hash complet et non d'un hash de commit à 7 caractères.

### Construire une URL de téléchargement

Au cas où vous voudriez construire l'URL utilisée pour télécharger un fichier depuis un dépôt, vous pouvez utiliser [`hf_hub_url`] qui retourne une URL.
Notez qu'elle est utilisée en interne par [`hf_hub_download`].

## Télécharger un dépôt entier

[`snapshot_download`] télécharge un dépôt entier à une révision donnée. Elle utilise en interne [`hf_hub_download`] ce qui
signifie que tous les fichiers téléchargés sont également mis en cache sur votre disque local. Les téléchargements sont effectués de manière concurrente pour accélérer le processus.

Pour télécharger un dépôt complet, passez simplement le `repo_id` et le `repo_type` :

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="lysandre/arxiv-nlp")
'/home/lysandre/.cache/huggingface/hub/models--lysandre--arxiv-nlp/snapshots/894a9adde21d9a3e3843e6d5aeaaf01875c7fade'

# Ou depuis un jeu de données
>>> snapshot_download(repo_id="google/fleurs", repo_type="dataset")
'/home/lysandre/.cache/huggingface/hub/datasets--google--fleurs/snapshots/199e4ae37915137c555b1765c01477c216287d34'
```

[`snapshot_download`] télécharge la dernière révision par défaut. Si vous voulez une révision de dépôt spécifique, utilisez le
paramètre `revision` :

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="lysandre/arxiv-nlp", revision="refs/pr/1")
```

### Filtrer les fichiers à télécharger

[`snapshot_download`] offre un moyen facile de télécharger un dépôt. Cependant, vous ne voulez pas toujours télécharger le
contenu entier d'un dépôt. Par exemple, vous pourriez vouloir éviter de télécharger tous les fichiers `.bin` si vous savez que vous n'utiliserez
que les poids `.safetensors`. Vous pouvez le faire en utilisant les paramètres `allow_patterns` et `ignore_patterns`.

Ces paramètres acceptent soit un seul motif, soit une liste de motifs. Les motifs sont des wildcards standard (motifs de globbing)
comme documenté [ici](https://tldp.org/LDP/GNU-Linux-Tools-Summary/html/x11655.htm). La correspondance de motifs est
basée sur [`fnmatch`](https://docs.python.org/3/library/fnmatch.html).

Par exemple, vous pouvez utiliser `allow_patterns` pour télécharger uniquement les fichiers de configuration JSON :

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="lysandre/arxiv-nlp", allow_patterns="*.json")
```

D'un autre côté, `ignore_patterns` peut exclure certains fichiers du téléchargement. L'
exemple suivant ignore les extensions de fichier `.msgpack` et `.h5` :

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="lysandre/arxiv-nlp", ignore_patterns=["*.msgpack", "*.h5"])
```

Enfin, vous pouvez combiner les deux pour filtrer précisément votre téléchargement. Voici un exemple pour télécharger tous les fichiers json et markdown
sauf `vocab.json`.

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="gpt2", allow_patterns=["*.md", "*.json"], ignore_patterns="vocab.json")
```

## Télécharger des fichier(s) dans un dossier local

Par défaut, nous recommandons d'utiliser le [système de cache](./manage-cache) pour télécharger des fichiers depuis le Hub. Vous pouvez spécifier un emplacement de cache personnalisé en utilisant le paramètre `cache_dir` dans [`hf_hub_download`] et [`snapshot_download`], ou en définissant la variable d'environnement [`HF_HOME`](../package_reference/environment_variables#hf_home).

Cependant, si vous devez télécharger des fichiers dans un dossier spécifique, vous pouvez passer un paramètre `local_dir` à la fonction de téléchargement. Ceci est utile pour obtenir un workflow plus proche de ce que la commande `git` offre. Les fichiers téléchargés conserveront leur structure de fichiers d'origine dans le dossier spécifié. Par exemple, si `filename="data/train.csv"` et `local_dir="path/to/folder"`, le chemin de fichier résultant sera `"path/to/folder/data/train.csv"`.

Un dossier `.cache/huggingface/` est créé à la racine de votre répertoire local contenant des métadonnées sur les fichiers téléchargés. Cela évite de re-télécharger des fichiers s'ils sont déjà à jour. Si les métadonnées ont changé, alors la nouvelle version du fichier est téléchargée. Avec cette fonctionnalité, le `local_dir` optimisé pour récupérer uniquement les dernières modifications.

Après avoir terminé le téléchargement, vous pouvez supprimer en toute sécurité le dossier `.cache/huggingface/` si vous n'en avez plus besoin. Cependant, sachez que ré-exécuter votre script sans ce dossier peut entraîner des temps de télechargement plus longs, car les métadonnées seront perdues. Rassurez-vous, vos données locales resteront intactes et non affectées, même si vous supprimez le dossier de cache.

> [!TIP]
> Ne vous inquiétez pas du dossier `.cache/huggingface/` lors du commit de modifications vers le Hub ! Ce dossier est automatiquement ignoré par `git` et [`upload_folder`].

## Télécharger depuis le CLI

Vous pouvez utiliser la commande `hf download` depuis le terminal pour télécharger directement des fichiers depuis le Hub.
En interne, elle utilise les mêmes helpers [`hf_hub_download`] et [`snapshot_download`] décrits ci-dessus et affiche le
chemin retourné dans le terminal.

```bash
>>> hf download gpt2 config.json
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10/config.json
```

Vous pouvez télécharger plusieurs fichiers à la fois, ce qui affiche une barre de progression et retourne le chemin du snapshot dans lequel les fichiers
sont situés :

```bash
>>> hf download gpt2 config.json model.safetensors
Fetching 2 files: 100%|████████████████████████████████████████████| 2/2 [00:00<00:00, 23831.27it/s]
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10
```

Pour plus de détails sur la commande de téléchargement du CLI, veuillez consulter le [guide CLI](./cli#hf-download).

## Mode dry-run

Dans certains cas, vous souhaiteriez vérifier quels fichiers seraient téléchargés avant de les télécharger réellement. Vous pouvez vérifier cela en utilisant le paramètre `--dry-run`. Il liste tous les fichiers à télécharger dans le dépôt et vérifie s'ils sont déjà téléchargés ou non. Ce paramètre donne une idée du nombre de fichiers à télécharger et de leurs tailles.

Voici un exemple, vérifiant un seul fichier :

```sh
>>> hf download openai-community/gpt2 onnx/decoder_model_merged.onnx --dry-run
[dry-run] Will download 1 files (out of 1) totalling 655.2M
File                           Bytes to download
------------------------------ -----------------
onnx/decoder_model_merged.onnx 655.2M
```

Et si le fichier est déjà en cache :

```sh
>>> hf download openai-community/gpt2 onnx/decoder_model_merged.onnx --dry-run
[dry-run] Will download 0 files (out of 1) totalling 0.0.
File                           Bytes to download
------------------------------ -----------------
onnx/decoder_model_merged.onnx -
```

Vous pouvez également exécuter un dry-run sur un dépôt entier :

```sh
>>> hf download openai-community/gpt2 --dry-run
[dry-run] Fetching 26 files: 100%|█████████████| 26/26 [00:04<00:00,  6.26it/s]
[dry-run] Will download 11 files (out of 26) totalling 5.6G.
File                              Bytes to download
--------------------------------- -----------------
.gitattributes                    -
64-8bits.tflite                   125.2M
64-fp16.tflite                    248.3M
64.tflite                         495.8M
README.md                         -
config.json                       -
flax_model.msgpack                497.8M
generation_config.json            -
merges.txt                        -
model.safetensors                 548.1M
onnx/config.json                  -
onnx/decoder_model.onnx           653.7M
onnx/decoder_model_merged.onnx    655.2M
onnx/decoder_with_past_model.onnx 653.7M
onnx/generation_config.json       -
onnx/merges.txt                   -
onnx/special_tokens_map.json      -
onnx/tokenizer.json               -
onnx/tokenizer_config.json        -
onnx/vocab.json                   -
pytorch_model.bin                 548.1M
rust_model.ot                     702.5M
tf_model.h5                       497.9M
tokenizer.json                    -
tokenizer_config.json             -
vocab.json                        -
```

Et avec le filtrage de fichiers :

```sh
>>> hf download openai-community/gpt2 --include "*.json"  --dry-run
[dry-run] Fetching 11 files: 100%|█████████████| 11/11 [00:00<00:00, 80518.92it/s]
[dry-run] Will download 0 files (out of 11) totalling 0.0.
File                         Bytes to download
---------------------------- -----------------
config.json                  -
generation_config.json       -
onnx/config.json             -
onnx/generation_config.json  -
onnx/special_tokens_map.json -
onnx/tokenizer.json          -
onnx/tokenizer_config.json   -
onnx/vocab.json              -
tokenizer.json               -
tokenizer_config.json        -
vocab.json                   -
```

Enfin, vous pouvez également effectuer un dry-run à chaque fois en passant `dry_run=True` aux méthodes [`hf_hub_download`] et [`snapshot_download`]. Cela retournera un [`DryRunFileInfo`] (respectivement une liste de [`DryRunFileInfo`]) avec pour chaque fichier contenant leur hash de commit, le nom et la taille du fichier, si le fichier est en cache et si le fichier serait téléchargé. En pratique, le fichier sera téléchargé s'il n'est pas en cache ou si `force_download=True` est passé.

## Téléchargements plus rapides

Profitez de téléchargements plus rapides grâce à `hf_xet`, la liaison Python vers la bibliothèque [`xet-core`](https://github.com/huggingface/xet-core) qui permet
la déduplication basée sur les chunks pour des téléchargements et uploads plus rapides. `hf_xet` s'intègre parfaitement avec `huggingface_hub`, mais utilise la bibliothèque Rust `xet-core` et le stockage Xet au lieu de LFS.

`hf_xet` utilise le système de stockage Xet, qui décompose les fichiers en chunks immuables, stockant des collections de ces chunks (appelés blocks ou xorbs) à distance et les récupérant pour ré-assembler le fichier lorsque demandé. Lors du téléchargement, après avoir confirmé que l'utilisateur est autorisé à accéder aux fichiers, `hf_xet` interrogera le service d'adressage de contenu Xet (CAS) avec le hash SHA256 LFS pour ce fichier pour recevoir les métadonnées de reconstruction (plages dans les xorbs) pour assembler ces fichiers, ainsi que des URLs pré-signées pour télécharger les xorbs directement. Ensuite, `hf_xet` téléchargera efficacement les plages de xorb nécessaires et écrira les fichiers sur le disque.

Pour l'activer, installez simplement la dernière version de `huggingface_hub` :

```bash
pip install -U "huggingface_hub"
```

À partir de `huggingface_hub` 0.32.0, cela installera également `hf_xet`.

Toutes les autres APIs `huggingface_hub` continueront à fonctionner sans aucune modification. Pour en savoir plus sur les avantages du stockage Xet et `hf_xet`, consultez cette [section](https://huggingface.co/docs/hub/xet/index).

Note : `hf_transfer` était anciennement utilisé avec le backend de stockage LFS et est maintenant déprécié ; utilisez `hf_xet` à la place.
