<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Ligne de commande (CLI)

L'interface en ligne de commande `huggingface-cli` vous permet d'interagir directement avec le Hub depuis votre terminal. Vous pouvez créer et gérer des dépôts, télécharger et uploader des fichiers, et effectuer d'autres opérations directement depuis votre terminal de commande.

## Installation

L'outil CLI `huggingface-cli` est inclus dans le package Python `huggingface_hub` :

```bash
pip install -U "huggingface_hub[cli]"
```

Si vous utilisez `pip install huggingface_hub` sans le composant `[cli]`, seules les commandes basiques sont installées par défaut. Pour installer toutes les fonctionnalités CLI (par exemple, la gestion du cache, la validation), il est recommandé d'installer le package avec le composant `[cli]`.

Pour mettre à jour le package vers la dernière version, exécutez :

```bash
pip install -U "huggingface_hub[cli]"
```

Pour vérifier que la CLI est correctement installée, vous pouvez exécuter :

```bash
huggingface-cli --help
```

Alternative : Installation avec curl (Linux/macOS)

```bash
curl -L https://hf.co/install-cli.sh | sh
```

Alternative : Installation avec `uv`

```bash
uv tool install "huggingface_hub[cli]"
```

Alternative : Installation avec Homebrew (macOS/Linux)

```bash
brew install huggingface-cli
```

> [!TIP]
> Dans toute la documentation, vous verrez des exemples utilisant `huggingface-cli` ou `hf`. Ces deux commandes sont équivalentes, `hf` étant simplement un alias pour le même outil. Utilisez celui que vous préférez, mais nous utiliserons généralement `hf` car il est plus court.

## hf auth login

Dans de nombreux cas, vous devrez vous connecter avec un compte Hugging Face pour interagir avec le Hub, que ce soit pour télécharger des dépôts privés, uploader des fichiers, créer des PRs, etc. Utilisez la commande suivante dans votre terminal pour vous connecter :

```bash
hf auth login
```

Cette commande vous indiquera si vous êtes déjà connecté et vous invitera à saisir votre jeton d'accès. Vous pouvez créer un jeton d'accès depuis vos [Paramètres de compte](https://huggingface.co/settings/tokens). Une fois connecté, le jeton d'accès sera stocké dans votre répertoire de cache (`~/.cache/huggingface/token` par défaut) et sera automatiquement utilisé lors de l'exécution de toute commande ou script Python appelant `huggingface_hub`.

Vous pouvez également passer votre jeton en utilisant l'option `--token` :

```bash
hf auth login --token YOUR_TOKEN
```

### Se connecter via une variable d'environnement

Vous pouvez définir votre jeton en tant que variable d'environnement `HF_TOKEN`. Cela permet au système d'authentification de récupérer le jeton même sans passer par `hf auth login`. Ceci est particulièrement utile pour les serveurs ou les environnements CI/CD qui ne permettent pas l'interaction avec les commandes.

```bash
export HF_TOKEN="YOUR_TOKEN"
```

### Se connecter avec git credentials

Alternativement, vous pouvez vous connecter en utilisant git credentials. Ceci est utile si vous souhaitez accéder aux dépôts Hugging Face depuis git directement et non via des scripts Python.

```bash
hf auth login --git-credential
```

Par défaut, cela configurera git pour utiliser le helper `store` qui stockera vos credentials en texte clair sur votre machine. Si vous préférez utiliser un keyring pour stocker vos credentials de manière sécurisée, utilisez `--git-credential-with-keyring` (nécessite `keyring` : `pip install keyring`) :

```bash
hf auth login --git-credential-with-keyring
```

Dans ce cas, git sera configuré pour utiliser le helper `huggingface` qui interagit avec votre keyring pour stocker et récupérer vos credentials. Voir [Git credentials](../package_reference/environment_variables#hfhubgitcredential) pour plus de détails.

Enfin, vous pouvez également ajouter votre jeton directement au git remote :

```bash
# Utilisez votre nom d'utilisateur et le jeton comme mot de passe
git clone https://USER:TOKEN@huggingface.co/my-username/my-model
```

> [!TIP]
> Configurez `--add-to-git-credential` en plus de `--token` pour vous connecter avec un jeton et le stocker dans git en une seule commande.

## hf auth whoami

Si vous souhaitez savoir si vous êtes connecté, vous pouvez utiliser `hf auth whoami`. Cette commande ne nécessite aucune authentification et n'est donc utile que pour vérifier si vous êtes connecté ou pour obtenir votre nom d'utilisateur et les organisations auxquelles vous appartenez :

```bash
hf auth whoami
```

Exemple de sortie :

```bash
Wauplin
orgs: huggingface,eu-test,hf-accelerate
```

## hf auth logout

Enfin, vous pouvez vous déconnecter en utilisant `hf auth logout`. Cette commande supprimera le jeton d'accès de votre cache (`~/.cache/huggingface/token`). Notez que votre jeton pourrait encore être disponible si vous l'avez défini via la variable d'environnement `HF_TOKEN`.

```bash
hf auth logout
```

## hf download

Utilisez la commande `hf download` pour télécharger des fichiers depuis le Hub. En interne, elle utilise les mêmes helpers [`hf_hub_download`] et [`snapshot_download`] décrits dans le guide [Télécharger](./download). Dans les exemples ci-dessous, nous passerons en revue les cas d'utilisation les plus courants. Pour une liste complète des options disponibles, vous pouvez exécuter :

```bash
hf download --help
```

### Télécharger un fichier unique

Pour télécharger un fichier unique depuis un dépôt, utilisez simplement la commande `hf download repo_id filename`.

```bash
hf download gpt2 config.json
```

Par défaut, le fichier sera téléchargé dans le répertoire de cache défini par la variable d'environnement `HF_HOME`. Cependant, dans la plupart des cas, vous souhaiterez probablement définir où le fichier va être téléchargé. La manière la plus simple de le faire est d'utiliser l'option `--local-dir`. Le chemin renvoyé sera alors "human-readable" :

```bash
>>> hf download gpt2 config.json --local-dir=./models/gpt2
./models/gpt2/config.json
```

### Télécharger un dépôt entier

Dans certains cas, vous souhaiterez simplement télécharger tous les fichiers d'un dépôt. Pour ce faire, omettez simplement l'argument `filename` :

```bash
>>> hf download HuggingFaceH4/zephyr-7b-beta
/home/wauplin/.cache/huggingface/hub/models--HuggingFaceH4--zephyr-7b-beta/snapshots/3bac358730f8806e5c3dc7c7e19eb36e045bf720
```

### Télécharger plusieurs fichiers

Vous pouvez également télécharger un sous-ensemble de fichiers d'un dépôt avec un seul appel. Il existe deux manières de le faire. Si vous avez une liste précise de fichiers à télécharger, vous pouvez simplement fournir une liste d'arguments `filename` :

```bash
>>> hf download gpt2 config.json model.safetensors
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10
```

Cependant, dans la plupart des cas d'utilisation, vous souhaiterez probablement filtrer les fichiers que vous souhaitez télécharger en utilisant un pattern (par exemple, télécharger tous les safetensors weights mais pas les sharded PyTorch weights). Vous pouvez le faire en utilisant les options `--include` et `--exclude`. Par exemple, pour télécharger tous les fichiers JSON et Markdown sauf `vocab.json` :

```bash
>>> hf download gpt2 --include="*.json" --include="*.md" --exclude="vocab.json"
Fetching 5 files: 100%|████████████████████████████████████████████| 5/5 [00:00<00:00, 41662.15it/s]
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10
```

### Télécharger un dataset ou un Space

Les exemples ci-dessus montrent comment télécharger depuis un dépôt de modèles. Pour télécharger un dataset ou un Space, utilisez les options `--repo-type=dataset` et `--repo-type=space` :

```bash
# Téléchargez un dataset unique
>>> hf download --repo-type=dataset lhoestq/custom_squad --include="*.json" --local-dir=./datasets/custom_squad
Fetching 9 files: 100%|████████████████████████████████████████████| 9/9 [00:00<00:00, 87664.16it/s]
./datasets/custom_squad

# Téléchargez le code d'un Gradio Space
>>> hf download --repo-type=space Wauplin/my-cool-training-space --include="*.py" --include="requirements.txt" --local-dir=./spaces/my-cool-training-space
Fetching 3 files: 100%|████████████████████████████████████████████| 3/3 [00:00<00:00, 24125.05it/s]
./spaces/my-cool-training-space
```

### Télécharger une révision spécifique

L'argument ci-dessus télécharge les derniers fichiers depuis la branche `main`. Pour télécharger depuis une autre branche ou une révision de référence (par exemple, d'une PR), utilisez l'option `--revision` :

```bash
>>> hf download bigcode/the-stack --repo-type=dataset --revision=v1.1 --include="data/python/*" --local-dir=./datasets/the-stack-python
Fetching 206 files: 100%|████████████████████████████████████████████| 206/206 [02:31<00:00,  1.36it/s]
./datasets/the-stack-python
```

### Dry-run mode

Si vous souhaitez avoir un aperçu des fichiers qui seront téléchargés avant que cela ne se produise réellement, utilisez l'option `--dry-run`. Cela s'avère utile lorsque vous souhaitez télécharger un dépôt entier avec des patterns `--include` et `--exclude` mais que vous n'êtes pas sûr que le pattern est correct. L'exemple suivant liste tous les fichiers du dépôt _adept/fuyu-8b_ sans télécharger quoi que ce soit :

```bash
>>> hf download adept/fuyu-8b --dry-run
config.json                        -
generation_config.json             -
handler.py                         -
model-00001-of-00002.safetensors   4.96G
model-00002-of-00002.safetensors   543.5M
model.safetensors.index.json       -
onnx/config.json                   -
onnx/decoder_model.onnx            653.7M
onnx/decoder_model_merged.onnx     655.2M
onnx/decoder_with_past_model.onnx  653.7M
pytorch_model.bin.index.json       -
pytorch_model-00001-of-00002.bin   5.0G
pytorch_model-00002-of-00002.bin   548.1M
requirements.txt                   -
special_tokens_map.json            -
tokenizer.json                     -
tokenizer.model                    -
tokenizer_config.json              -
```

Pour plus de détails, consultez le [guide de téléchargement](./download.md#dry-run-mode).

### Spécifier le répertoire de cache

Si vous n'utilisez pas `--local-dir`, tous les fichiers seront téléchargés par défaut dans le répertoire de cache défini par la variable d'environnement `HF_HOME` [environment variable](../package_reference/environment_variables#hfhome). Vous pouvez spécifier un cache personnalisé en utilisant `--cache-dir` :

```bash
>>> hf download adept/fuyu-8b --cache-dir ./path/to/cache
...
./path/to/cache/models--adept--fuyu-8b/snapshots/ddcacbcf5fdf9cc59ff01f6be6d6662624d9c745
```

### Spécifier un jeton

Pour accéder aux dépôts privés ou à accès restreint, vous devez utiliser un jeton. Par défaut, le cli utilise le jeton enregistré localement (en utilisant `hf auth login`). Si vous souhaitez vous authentifier explicitement, utilisez l'option `--token` :

```bash
>>> hf download gpt2 config.json --token=hf_****
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10/config.json
```

### Mode silencieux

Par défaut, la commande `hf download` sera verbeuse. Elle affichera des détails tels que des messages d'avertissement, des informations sur les fichiers téléchargés et des barres de progression. Si vous souhaitez masquer tout cela, utilisez l'option `--quiet`. Seule la dernière ligne (c'est-à-dire le chemin vers les fichiers téléchargés) est affichée. Cela peut s'avérer utile si vous souhaitez passer la sortie à une autre commande dans un script.

```bash
>>> hf download gpt2 --quiet
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10
```

### Timeout de téléchargement

Sur les machines avec des connexions lentes, vous pourriez rencontrer des problèmes de timeout comme celui-ci :
```bash
`httpx.TimeoutException: (TimeoutException("HTTPSConnectionPool(host='cdn-lfs-us-1.huggingface.co', port=443): Read timed out. (read timeout=10)"), '(Request ID: a33d910c-84c6-4514-8362-c705e2039d38)')`
```

Pour atténuer ce problème, vous pouvez définir la variable d'environnement `HF_HUB_DOWNLOAD_TIMEOUT` avec une valeur plus élevée (la valeur par défaut est 10) :
```bash
export HF_HUB_DOWNLOAD_TIMEOUT=30
```

Pour plus de détails, consultez la [référence des variables d'environnement](../package_reference/environment_variables#hfhubdownloadtimeout). Et relancez votre commande de téléchargement.

## hf upload

Utilisez la commande `hf upload` pour uploader des fichiers vers le Hub directement. En interne, elle utilise les mêmes helpers [`upload_file`] et [`upload_folder`] décrits dans le guide [Upload](./upload). Dans les exemples ci-dessous, nous passerons en revue les cas d'utilisation les plus courants. Pour une liste complète des options disponibles, vous pouvez exécuter :

```bash
>>> hf upload --help
```

### Uploader un dossier entier

L'utilisation par défaut pour cette commande est :

```bash
# Usage:  hf upload [repo_id] [local_path] [path_in_repo]
```

Pour uploader le répertoire actuel à la racine du dépôt, utilisez :

```bash
>>> hf upload my-cool-model . .
https://huggingface.co/Wauplin/my-cool-model/tree/main/
```

> [!TIP]
> Si le dépôt n'existe pas encore, il sera créé automatiquement.

Vous pouvez également uploader un dossier spécifique :

```bash
>>> hf upload my-cool-model ./models .
https://huggingface.co/Wauplin/my-cool-model/tree/main/
```

Enfin, vous pouvez uploader un dossier vers une destination spécifique sur le dépôt :

```bash
>>> hf upload my-cool-model ./path/to/curated/data /data/train
https://huggingface.co/Wauplin/my-cool-model/tree/main/data/train
```

### Uploader un fichier unique

Vous pouvez également uploader un fichier unique en configurant `local_path` pour pointer vers un fichier sur votre machine. Si c'est le cas, `path_in_repo` est optionnel et sera par défaut le nom de votre fichier local :

```bash
>>> hf upload Wauplin/my-cool-model ./models/model.safetensors
https://huggingface.co/Wauplin/my-cool-model/blob/main/model.safetensors
```

Si vous souhaitez uploader un fichier unique vers un répertoire spécifique, configurez `path_in_repo` en conséquence :

```bash
>>> hf upload Wauplin/my-cool-model ./models/model.safetensors /vae/model.safetensors
https://huggingface.co/Wauplin/my-cool-model/blob/main/vae/model.safetensors
```

### Uploader plusieurs fichiers

Pour uploader plusieurs fichiers depuis un dossier en une seule fois sans uploader le dossier entier, utilisez les patterns `--include` et `--exclude`. Cela peut également être combiné avec l'option `--delete` pour supprimer des fichiers sur le dépôt tout en uploadant de nouveaux. Dans l'exemple ci-dessous, nous synchronisons le Space local en supprimant les fichiers distants et en uploadant tous les fichiers sauf ceux du répertoire `/logs` :

```bash
# Synchroniser le Space local avec le Hub (uploader de nouveaux fichiers sauf depuis logs/, supprimer les fichiers retirés)
>>> hf upload Wauplin/space-example --repo-type=space --exclude="/logs/*" --delete="*" --commit-message="Sync local Space with Hub"
...
```

### Uploader vers un dataset ou un Space

Pour uploader vers un dataset ou un Space, utilisez l'option `--repo-type` :

```bash
>>> hf upload Wauplin/my-cool-dataset ./data /train --repo-type=dataset
...
```

### Uploader vers une organisation

Pour uploader du contenu vers un dépôt appartenant à une organisation plutôt qu'un dépôt personnel, vous devez le spécifier explicitement dans le `repo_id` :

```bash
>>> hf upload MyCoolOrganization/my-cool-model . .
https://huggingface.co/MyCoolOrganization/my-cool-model/tree/main/
```

### Uploader vers une révision spécifique

Par défaut, les fichiers sont uploadés vers la branche `main`. Si vous souhaitez uploader des fichiers vers une autre branche ou référence, utilisez l'option `--revision` :

```bash
# Uploader des fichiers vers une PR
>>> hf upload bigcode/the-stack . . --repo-type dataset --revision refs/pr/104
...
```

**Note :** si `revision` n'existe pas et que `--create-pr` n'est pas défini, une branche sera créée automatiquement depuis la branche `main`.

### Uploader et créer une PR

Si vous n'avez pas la permission de pousser vers un dépôt, vous devez ouvrir une PR et informer les auteurs des modifications que vous souhaitez apporter. Cela peut être fait en configurant l'option `--create-pr` :

```bash
# Créer une PR et uploader les fichiers vers celle-ci
>>> hf upload bigcode/the-stack . . --repo-type dataset --revision refs/pr/104
https://huggingface.co/datasets/bigcode/the-stack/blob/refs%2Fpr%2F104/
```

### Uploader à intervalles réguliers

Dans certains cas, vous pourriez vouloir pousser des mises à jour régulières vers un dépôt. Par exemple, cela est utile si vous entraînez un modèle et que vous souhaitez uploader le dossier de logs toutes les 10 minutes. Vous pouvez le faire en utilisant l'option `--every` :

```bash
# Uploader de nouveaux logs toutes les 10 minutes
hf upload training-model logs/ --every=10
```

### Spécifier un message de commit

Utilisez `--commit-message` et `--commit-description` pour définir un message et une description personnalisés pour votre commit au lieu de ceux par défaut

```bash
>>> hf upload Wauplin/my-cool-model ./models . --commit-message="Epoch 34/50" --commit-description="Val accuracy: 68%. Check tensorboard for more details."
...
https://huggingface.co/Wauplin/my-cool-model/tree/main
```

### Spécifier un jeton

Pour uploader des fichiers, vous devez utiliser un jeton. Par défaut, le jeton enregistré localement (en utilisant `hf auth login`) sera utilisé. Si vous souhaitez vous authentifier explicitement, utilisez l'option `--token` :

```bash
>>> hf upload Wauplin/my-cool-model ./models . --token=hf_****
...
https://huggingface.co/Wauplin/my-cool-model/tree/main
```

### Mode silencieux

Par défaut, la commande `hf upload` sera verbeuse. Elle affichera des détails tels que des messages d'avertissement, des informations sur les fichiers uploadés et des barres de progression. Si vous souhaitez masquer tout cela, utilisez l'option `--quiet`. Seule la dernière ligne (c'est-à-dire l'URL vers les fichiers uploadés) est affichée. Cela peut s'avérer utile si vous souhaitez passer la sortie à une autre commande dans un script.

```bash
>>> hf upload Wauplin/my-cool-model ./models . --quiet
https://huggingface.co/Wauplin/my-cool-model/tree/main
```

## hf repo

`hf repo` vous permet de créer, supprimer, déplacer des dépôts et mettre à jour leurs paramètres sur le Hugging Face Hub. Elle inclut également des sous-commandes pour gérer les branches et les tags.

### Créer un dépôt

```bash
>>> hf repo create Wauplin/my-cool-model
Successfully created Wauplin/my-cool-model on the Hub.
Your repo is now available at https://huggingface.co/Wauplin/my-cool-model
```

Créer un dataset privé ou un Space :

```bash
>>> hf repo create my-cool-dataset --repo-type dataset --private
>>> hf repo create my-gradio-space --repo-type space --space-sdk gradio
```

Utilisez `--exist-ok` si le dépôt peut déjà exister, et `--resource-group-id` pour cibler un groupe de ressources Enterprise.

### Supprimer un dépôt

```bash
>>> hf repo delete Wauplin/my-cool-model
```

Datasets et Spaces :

```bash
>>> hf repo delete my-cool-dataset --repo-type dataset
>>> hf repo delete my-gradio-space --repo-type space
```

### Déplacer un dépôt

```bash
>>> hf repo move old-namespace/my-model new-namespace/my-model
```

### Mettre à jour les paramètres du dépôt

```bash
>>> hf repo settings Wauplin/my-cool-model --gated auto
>>> hf repo settings Wauplin/my-cool-model --private true
>>> hf repo settings Wauplin/my-cool-model --private false
```

- `--gated` : l'un de `auto`, `manual`, `false`
- `--private true|false` : définir la confidentialité du dépôt

### Gérer les branches

```bash
>>> hf repo branch create Wauplin/my-cool-model dev
>>> hf repo branch create Wauplin/my-cool-model release-1 --revision refs/pr/104
>>> hf repo branch delete Wauplin/my-cool-model dev
```

> [!TIP]
> Toutes les commandes acceptent `--repo-type` (l'un de `model`, `dataset`, `space`) et `--token` si vous devez vous authentifier explicitement. Utilisez `--help` sur n'importe quelle commande pour voir toutes les options.


## hf repo-files

Si vous souhaitez supprimer des fichiers d'un dépôt Hugging Face, utilisez la commande `hf repo-files`.

### Supprimer des fichiers

La sous-commande `hf repo-files delete <repo_id>` vous permet de supprimer des fichiers d'un dépôt. Voici quelques exemples d'utilisation.

Supprimer un dossier :
```bash
>>> hf repo-files delete Wauplin/my-cool-model folder/
Files correctly deleted from repo. Commit: https://huggingface.co/Wauplin/my-cool-mo...
```

Supprimer plusieurs fichiers :
```bash
>>> hf repo-files delete Wauplin/my-cool-model file.txt folder/pytorch_model.bin
Files correctly deleted from repo. Commit: https://huggingface.co/Wauplin/my-cool-mo...
```

Utiliser des wildcards de style Unix pour supprimer des ensembles de fichiers :
```bash
>>> hf repo-files delete Wauplin/my-cool-model "*.txt" "folder/*.bin"
Files correctly deleted from repo. Commit: https://huggingface.co/Wauplin/my-cool-mo...
```

### Spécifier un jeton

Pour supprimer des fichiers d'un dépôt, vous devez être authentifié et autorisé. Par défaut, le jeton enregistré localement (en utilisant `hf auth login`) sera utilisé. Si vous souhaitez vous authentifier explicitement, utilisez l'option `--token` :

```bash
>>> hf repo-files delete --token=hf_**** Wauplin/my-cool-model file.txt
```

## hf cache ls

Utilisez `hf cache ls` pour inspecter ce qui est stocké localement dans votre cache Hugging Face. Par défaut, elle agrège les informations par dépôt :

```bash
>>> hf cache ls
ID                          SIZE     LAST_ACCESSED LAST_MODIFIED REFS        
--------------------------- -------- ------------- ------------- ----------- 
dataset/nyu-mll/glue          157.4M 2 days ago    2 days ago    main script 
model/LiquidAI/LFM2-VL-1.6B     3.2G 4 days ago    4 days ago    main        
model/microsoft/UserLM-8b      32.1G 4 days ago    4 days ago    main  

Found 3 repo(s) for a total of 5 revision(s) and 35.5G on disk.
```

Ajoutez `--revisions` pour descendre jusqu'aux snapshots spécifiques, et enchaînez les filtres pour vous concentrer sur ce qui compte :

```bash
>>> hf cache ls --filter "size>30g" --revisions
ID                        REVISION                                 SIZE     LAST_MODIFIED REFS 
------------------------- ---------------------------------------- -------- ------------- ---- 
model/microsoft/UserLM-8b be8f2069189bdf443e554c24e488ff3ff6952691    32.1G 4 days ago    main 

Found 1 repo(s) for a total of 1 revision(s) and 32.1G on disk.
```

La commande prend en charge plusieurs formats de sortie pour les scripts : `--format json` affiche des objets structurés, `--format csv` écrit des lignes séparées par des virgules, et `--quiet` affiche uniquement les ID. Utilisez `--sort` pour ordonner les entrées par `accessed`, `modified`, `name`, ou `size` (ajoutez `:asc` ou `:desc` pour contrôler l'ordre), et `--limit` pour restreindre les résultats aux N premières entrées. Combinez-les avec `--cache-dir` pour cibler des emplacements de cache alternatifs. Consultez le guide [Gérer votre cache](./manage-cache) pour des workflows avancés.

Supprimez les entrées de cache sélectionnées avec `hf cache ls --q` en pipant les ID dans `hf cache rm` :

```bash
>>> hf cache rm $(hf cache ls --filter "accessed>1y" -q) -y
About to delete 2 repo(s) totalling 5.31G.
  - model/meta-llama/Llama-3.2-1B-Instruct (entire repo)
  - model/hexgrad/Kokoro-82M (entire repo)
Delete repo: ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct
Delete repo: ~/.cache/huggingface/hub/models--hexgrad--Kokoro-82M
Cache deletion done. Saved 5.31G.
Deleted 2 repo(s) and 2 revision(s); freed 5.31G.
```

## hf cache rm

`hf cache rm` supprime les dépôts en cache ou les révisions individuelles. Passez un ou plusieurs ID de dépôt (`model/bert-base-uncased`) ou hashes de révision :

```bash
>>> hf cache rm model/LiquidAI/LFM2-VL-1.6B
About to delete 1 repo(s) totalling 3.2G.
  - model/LiquidAI/LFM2-VL-1.6B (entire repo)
Proceed with deletion? [y/N]: y
Delete repo: ~/.cache/huggingface/hub/models--LiquidAI--LFM2-VL-1.6B
Cache deletion done. Saved 3.2G.
Deleted 1 repo(s) and 2 revision(s); freed 3.2G.
```

Mélangez des dépôts et des révisions spécifiques dans le même appel. Utilisez `--dry-run` pour prévisualiser l'impact, ou `--yes` pour ignorer le message de confirmation dans les scripts automatisés :

```bash
>>> hf cache rm model/t5-small 8f3ad1c --dry-run
About to delete 1 repo(s) and 1 revision(s) totalling 1.1G.
  - model/t5-small:
      8f3ad1c [main] 1.1G
Dry run: no files were deleted.
```

Lors de travaux en dehors de l'emplacement de cache par défaut, associez la commande avec `--cache-dir PATH`.

## hf cache prune

`hf cache prune` est un raccourci qui supprime toutes les révisions détachées (non référencées) dans votre cache. Cela ne conserve que les révisions qui sont toujours accessibles via une branche ou un tag :

```bash
>>> hf cache prune
About to delete 3 unreferenced revision(s) (2.4G total).
  - model/t5-small:
      1c610f6b [refs/pr/1] 820.1M
      d4ec9b72 [(detached)] 640.5M
  - dataset/google/fleurs:
      2b91c8dd [(detached)] 937.6M
Proceed? [y/N]: y
Deleted 3 unreferenced revision(s); freed 2.4G.
```

Comme avec les autres commandes de cache, `--dry-run`, `--yes`, et `--cache-dir` sont disponibles. Référez-vous au guide [Gérer votre cache](./manage-cache) pour plus d'exemples.

## hf cache verify

Utilisez `hf cache verify` pour valider les fichiers locaux par rapport à leurs checksums sur le Hub. Vous pouvez vérifier soit un snapshot du cache soit un répertoire local normal.

Exemples :

```bash
# Vérifier la révision main d'un modèle dans le cache
>>> hf cache verify deepseek-ai/DeepSeek-OCR

# Vérifier une révision spécifique
>>> hf cache verify deepseek-ai/DeepSeek-OCR --revision refs/pr/5
>>> hf cache verify deepseek-ai/DeepSeek-OCR --revision ef93bf4a377c5d5ed9dca78e0bc4ea50b26fe6a4

# Vérifier un dépôt privé
>>> hf cache verify me/private-model --token hf_***

# Vérifier un dataset
>>> hf cache verify karpathy/fineweb-edu-100b-shuffle --repo-type dataset

# Vérifier les fichiers dans un répertoire local
>>> hf cache verify deepseek-ai/DeepSeek-OCR --local-dir /path/to/repo
```

Par défaut, la commande avertit sur les fichiers manquants ou supplémentaires. Utilisez des drapeaux pour transformer ces avertissements en erreurs :

```bash
>>> hf cache verify deepseek-ai/DeepSeek-OCR --fail-on-missing-files --fail-on-extra-files
```

En cas de succès, vous verrez un résumé :

```text
✅ Verified 13 file(s) for 'deepseek-ai/DeepSeek-OCR' (model) in ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6
  All checksums match.
```

Si des non-correspondances sont détectées, la commande affiche une liste détaillée et se termine avec un statut non nul.

## hf repo tag create

La commande `hf repo tag create` vous permet de tagger, untagger et lister les tags pour les dépôts.

### Tagger un modèle

Pour tagger un dépôt, vous devez fournir le `repo_id` et le nom du `tag` :

```bash
>>> hf repo tag create Wauplin/my-cool-model v1.0
You are about to create tag v1.0 on model Wauplin/my-cool-model
Tag v1.0 created on Wauplin/my-cool-model
```

### Tagger un modèle à une révision spécifique

Si vous souhaitez tagger une révision spécifique, vous pouvez utiliser l'option `--revision`. Par défaut, le tag sera créé sur la branche `main` :

```bash
>>> hf repo tag create Wauplin/my-cool-model v1.0 --revision refs/pr/104
You are about to create tag v1.0 on model Wauplin/my-cool-model
Tag v1.0 created on Wauplin/my-cool-model
```

### Tagger un dataset ou un Space

Si vous souhaitez tagger un dataset ou Space, vous devez spécifier l'option `--repo-type` :

```bash
>>> hf repo tag create bigcode/the-stack v1.0 --repo-type dataset
You are about to create tag v1.0 on dataset bigcode/the-stack
Tag v1.0 created on bigcode/the-stack
```

### Lister les tags

Pour lister tous les tags d'un dépôt, utilisez l'option `-l` ou `--list` :

```bash
>>> hf repo tag create Wauplin/gradio-space-ci -l --repo-type space
Tags for space Wauplin/gradio-space-ci:
0.2.2
0.2.1
0.2.0
0.1.2
0.0.2
0.0.1
```

### Supprimer un tag

Pour supprimer un tag, utilisez l'option `-d` ou `--delete` :

```bash
>>> hf repo tag create -d Wauplin/my-cool-model v1.0
You are about to delete tag v1.0 on model Wauplin/my-cool-model
Proceed? [Y/n] y
Tag v1.0 deleted on Wauplin/my-cool-model
```

Vous pouvez également passer `-y` pour ignorer l'étape de confirmation.

## hf env

La commande `hf env` affiche des détails sur la configuration de votre machine. Ceci est utile lorsque vous ouvrez un problème sur [GitHub](https://github.com/huggingface/huggingface_hub) pour aider les mainteneurs à enquêter sur votre problème.

```bash
>>> hf env

Copy-and-paste the text below in your GitHub issue.

- huggingface_hub version: 1.0.0.rc6
- Platform: Linux-6.8.0-85-generic-x86_64-with-glibc2.35
- Python version: 3.11.14
- Running in iPython ?: No
- Running in notebook ?: No
- Running in Google Colab ?: No
- Running in Google Colab Enterprise ?: No
- Token path ?: /home/wauplin/.cache/huggingface/token
- Has saved token ?: True
- Who am I ?: Wauplin
- Configured git credential helpers: store
- Installation method: unknown
- Torch: N/A
- httpx: 0.28.1
- hf_xet: 1.1.10
- gradio: 5.41.1
- tensorboard: N/A
- pydantic: 2.11.7
- ENDPOINT: https://huggingface.co
- HF_HUB_CACHE: /home/wauplin/.cache/huggingface/hub
- HF_ASSETS_CACHE: /home/wauplin/.cache/huggingface/assets
- HF_TOKEN_PATH: /home/wauplin/.cache/huggingface/token
- HF_STORED_TOKENS_PATH: /home/wauplin/.cache/huggingface/stored_tokens
- HF_HUB_OFFLINE: False
- HF_HUB_DISABLE_TELEMETRY: False
- HF_HUB_DISABLE_PROGRESS_BARS: None
- HF_HUB_DISABLE_SYMLINKS_WARNING: False
- HF_HUB_DISABLE_EXPERIMENTAL_WARNING: False
- HF_HUB_DISABLE_IMPLICIT_TOKEN: False
- HF_HUB_DISABLE_XET: False
- HF_HUB_ETAG_TIMEOUT: 10
- HF_HUB_DOWNLOAD_TIMEOUT: 10
```

## hf jobs

Exécutez des jobs de calcul sur l'infrastructure Hugging Face avec une interface familière de type Docker.

`hf jobs` est un outil en ligne de commande qui vous permet d'exécuter n'importe quoi sur l'infrastructure de Hugging Face (y compris les GPU et TPU !) avec des commandes simples. Pensez à `docker run`, mais pour exécuter du code sur des A100.

```bash
# Exécuter directement du code Python
>>> hf jobs run python:3.12 python -c 'print("Hello from the cloud!")'

# Utiliser des GPU sans aucune configuration
>>> hf jobs run --flavor a10g-small pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
... python -c "import torch; print(torch.cuda.get_device_name())"

# Exécuter dans un compte d'organisation
>>> hf jobs run --namespace my-org-name python:3.12 python -c "print('Running in an org account')"

# Exécuter depuis des Hugging Face Spaces
>>> hf jobs run hf.co/spaces/lhoestq/duckdb duckdb -c "select 'hello world'"

# Exécuter un script Python avec `uv` (expérimental)
>>> hf jobs uv run my_script.py
```

### ✨ Fonctionnalités clés

- 🐳 **CLI de type Docker** : Commandes familières (`run`, `ps`, `logs`, `inspect`) pour exécuter et gérer les jobs
- 🔥 **N'importe quel matériel** : Des CPU aux GPU A100 et pods TPU - changez avec un simple drapeau
- 📦 **Exécutez n'importe quoi** : Utilisez des images Docker, des HF Spaces, ou vos conteneurs personnalisés
- 🔐 **Authentification simple** : Il suffit d'utiliser votre jeton HF
- 📊 **Surveillance en direct** : Streamer les logs en temps réel, comme si vous exécutiez localement
- 💰 **Paiement à l'utilisation** : Ne payez que pour les secondes que vous utilisez

> [!TIP]
> Les **Hugging Face Jobs** ne sont disponibles que pour les [utilisateurs Pro](https://huggingface.co/pro) et les [organisations Team ou Enterprise](https://huggingface.co/enterprise). Mettez à niveau votre abonnement pour commencer !

### Démarrage rapide

#### 1. Exécuter votre premier job

```bash
# Exécuter un simple script Python
>>> hf jobs run python:3.12 python -c "print('Hello from HF compute!')"
```

Cette commande exécute le job et affiche les logs. Vous pouvez passer `--detach` pour exécuter le Job en arrière-plan et n'afficher que l'ID du Job.

#### 2. Vérifier le statut du job

```bash
# Lister vos jobs en cours d'exécution
>>> hf jobs ps

# Inspecter le statut d'un job
>>> hf jobs inspect <job_id>

# Afficher les logs d'un job
>>> hf jobs logs <job_id>

# Annuler un job
>>> hf jobs cancel <job_id>
```

#### 3. Exécuter sur GPU

Vous pouvez également exécuter des jobs sur des GPU ou TPU avec l'option `--flavor`. Par exemple, pour exécuter un job PyTorch sur un GPU A10G :

```bash
# Utiliser un GPU A10G pour vérifier PyTorch CUDA
>>> hf jobs run --flavor a10g-small pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
... python -c "import torch; print(f"This code ran with the following GPU: {torch.cuda.get_device_name()}")"
```

L'exécution de ceci affichera la sortie suivante !

```bash
This code ran with the following GPU: NVIDIA A10G
```

Vous exécutez maintenant du code sur l'infrastructure de Hugging Face.

### Cas d'utilisation courants

- **Entraînement de modèles** : Affinez ou entraînez des modèles sur des GPU (T4, A10G, A100) sans gérer d'infrastructure
- **Génération de données synthétiques** : Générez des datasets à grande échelle en utilisant des LLM sur du matériel puissant
- **Traitement de données** : Traitez des datasets massifs avec des configurations haute-CPU pour des charges de travail parallèles
- **Inférence par lots** : Exécutez des inférences hors ligne sur des milliers d'échantillons en utilisant des configurations GPU optimisées
- **Expériences & Benchmarks** : Exécutez des expériences ML sur du matériel cohérent pour des résultats reproductibles
- **Développement & Débogage** : Testez du code GPU sans configuration CUDA locale

### Passer des variables d'environnement et des secrets

Vous pouvez passer des variables d'environnement à votre job en utilisant 

```bash
# Passer des variables d'environnement
>>> hf jobs run -e FOO=foo -e BAR=bar python:3.12 python -c "import os; print(os.environ['FOO'], os.environ['BAR'])"
```

```bash
# Passer un environnement depuis un fichier .env local
>>> hf jobs run --env-file .env python:3.12 python -c "import os; print(os.environ['FOO'], os.environ['BAR'])"
```

```bash
# Passer des secrets - ils seront chiffrés côté serveur
>>> hf jobs run -s MY_SECRET=psswrd python:3.12 python -c "import os; print(os.environ['MY_SECRET'])"
```

```bash
# Passer des secrets depuis un fichier .env.secrets local - ils seront chiffrés côté serveur
>>> hf jobs run --secrets-file .env.secrets python:3.12 python -c "import os; print(os.environ['MY_SECRET'])"
```

> [!TIP]
> Utilisez `--secrets HF_TOKEN` pour passer votre jeton Hugging Face local implicitement.
> Avec cette syntaxe, le secret est récupéré depuis la variable d'environnement.
> Pour `HF_TOKEN`, il peut lire le fichier de jeton situé dans le dossier home de Hugging Face si la variable d'environnement n'est pas définie.

### Matériel

Options `--flavor` disponibles :

- CPU : `cpu-basic`, `cpu-upgrade`
- GPU : `t4-small`, `t4-medium`, `l4x1`, `l4x4`, `a10g-small`, `a10g-large`, `a10g-largex2`, `a10g-largex4`,`a100-large`
- TPU : `v5e-1x1`, `v5e-2x2`, `v5e-2x4`

(mis à jour en 07/2025 depuis la [documentation suggested_hardware](https://huggingface.co/docs/hub/en/spaces-config-reference) de Hugging Face)

### Scripts UV (Expérimental)

Exécutez des scripts UV (scripts Python avec dépendances inline) sur l'infrastructure HF :

```bash
# Exécuter un script UV (crée un dépôt temporaire)
>>> hf jobs uv run my_script.py

# Exécuter avec un dépôt persistant
>>> hf jobs uv run my_script.py --repo my-uv-scripts

# Exécuter avec GPU
>>> hf jobs uv run ml_training.py --flavor gpu-t4-small

# Passer des arguments au script
>>> hf jobs uv run process.py input.csv output.parquet

# Ajouter des dépendances
>>> hf jobs uv run --with transformers --with torch train.py

# Exécuter un script directement depuis une URL
>>> hf jobs uv run https://huggingface.co/datasets/username/scripts/resolve/main/example.py

# Exécuter une commande
>>> hf jobs uv run --with lighteval python -c "import lighteval"
```

Les scripts UV sont des scripts Python qui incluent leurs dépendances directement dans le fichier en utilisant une syntaxe de commentaire spéciale. Cela les rend parfaits pour les tâches autonomes qui ne nécessitent pas de configurations complexes. En savoir plus sur les scripts UV dans la [documentation UV](https://docs.astral.sh/uv/guides/scripts/).

### Jobs planifiés

Planifiez et gérez des jobs qui s'exécuteront sur l'infrastructure HF.

Le planning doit être l'un de `@annually`, `@yearly`, `@monthly`, `@weekly`, `@daily`, `@hourly`, ou une expression CRON (par exemple, `"0 9 * * 1"` pour 9h tous les lundis).

```bash
# Planifier un job qui s'exécute toutes les heures
>>> hf jobs scheduled run @hourly python:3.12 python -c 'print("This runs every hour!")'

# Utiliser la syntaxe CRON
>>> hf jobs scheduled run "*/5 * * * *" python:3.12 python -c 'print("This runs every 5 minutes!")'

# Planifier avec GPU
>>> hf jobs scheduled run @hourly --flavor a10g-small pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
... python -c "import torch; print(f"This code ran with the following GPU: {torch.cuda.get_device_name()}")"

# Planifier un script UV
>>> hf jobs scheduled uv run @hourly my_script.py
```

Utilisez les mêmes paramètres que `hf jobs run` pour passer des variables d'environnement, des secrets, un timeout, etc.

Gérez les jobs planifiés en utilisant

```bash
# Lister vos jobs planifiés actifs
>>> hf jobs scheduled ps

# Inspecter le statut d'un job
>>> hf jobs scheduled inspect <scheduled_job_id>

# Suspendre (mettre en pause) un job planifié
>>> hf jobs scheduled suspend <scheduled_job_id>

# Reprendre un job planifié
>>> hf jobs scheduled resume <scheduled_job_id>

# Supprimer un job planifié
>>> hf jobs scheduled delete <scheduled_job_id>
```

## hf endpoints

Utilisez `hf endpoints` pour lister, déployer, décrire et gérer les Inference Endpoints directement depuis le terminal. L'alias hérité
`hf inference-endpoints` reste disponible pour la compatibilité.

```bash
# Lister les endpoints dans votre namespace
>>> hf endpoints ls

# Déployer un endpoint depuis le Model Catalog
>>> hf endpoints catalog deploy --repo openai/gpt-oss-120b --name my-endpoint

# Déployer un endpoint depuis le Hugging Face Hub 
>>> hf endpoints deploy my-endpoint --repo gpt2 --framework pytorch --accelerator cpu --instance-size x2 --instance-type intel-icl

# Lister les entrées du catalogue
>>> hf endpoints catalog ls

# Afficher le statut et les métadonnées
>>> hf endpoints describe my-endpoint

# Mettre l'endpoint en pause
>>> hf endpoints pause my-endpoint

# Supprimer sans invite de confirmation
>>> hf endpoints delete my-endpoint --yes
```

> [!TIP]
> Ajoutez `--namespace` pour cibler une organisation, `--token` pour remplacer l'authentification.
