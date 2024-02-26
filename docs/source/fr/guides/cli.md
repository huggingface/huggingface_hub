<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Invite de commande (CLI)

Le module Python `huggingface_hub` offre un CLI intégré appelé `huggingface-cli`. Cet outil vous permet d'intéragir avec le Hub Hugging
Face directement depuis un terminal. Par exemple, vous pouvez vous connecter à votre compte, créer un dépot, upload/download des fichiers, etc.
Le CLI contient aussi des fonctionnalités pratiques pour configurer votre machine ou gérer votre cache. Dans ce guide, nous regarderons les
fonctionnalités principales du CLI et comment les utiliser.


## Installation

Tout d'abord, installons le CLI:

```
>>> pip install -U "huggingface_hub[cli]"
```

<Tip>

Dans les snippet ci dessus, nous avons aussi installé les dépendances `[cli]` pour que vous ayez une meilleure expérience utilisateur, en particulier lors de l'utilisation de la commande `delete-cache`.

</Tip>

Une fois cette première étape terminée, vous pouvez vérifier que le CLI est correctement installé :

```
>>> huggingface-cli --help
usage: huggingface-cli <command> [<args>]

positional arguments:
  {env,login,whoami,logout,repo,upload,download,lfs-enable-largefiles,lfs-multipart-upload,scan-cache,delete-cache}
                        huggingface-cli command helpers
    env                 Print information about the environment.
    login               Log in using a token from huggingface.co/settings/tokens
    whoami              Find out which huggingface.co account you are logged in as.
    logout              Log out
    repo                {create} Commands to interact with your huggingface.co repos.
    upload              Upload a file or a folder to a repo on the Hub
    download            Download files from the Hub
    lfs-enable-largefiles
                        Configure your repository to enable upload of files > 5GB.
    scan-cache          Scan cache directory.
    delete-cache        Delete revisions from the cache directory.

options:
  -h, --help            show this help message and exit
```

Si le CLI est installé correctement, vous devriez voir une liste de toutes les options disponibles dans le CLI. Si vous obtenez un message d'erreur
tel que `command not found: huggingface-cli`, veuillez  vous référer au guide disponible ici : [Installation](../installation) 

<Tip>

L'option `--help` est assez pratique pour obtenir plus de détails sur une commande. Vous pouvez l'utiliser n'importe quand pour lister toutes les options
disponibles et leur détail. Par exemple, `huggingface-cli upload --help` fournira des informations permettant d'upload des fichiers en utilisant le
CLI.

</Tip>

## Connexion à huggingface-cli

Dans la plupart des cas, vous devez être connectés à un compte Hugging Face pour intéragir avec le Hub (par exempel pour télécharger des dépôts privés, upload des fichiers, créer des pull requests, etc.). Pour ce faire, vous avez besoin d'un [token d'authentification](https://huggingface.co/docs/hub/security-tokens) généré depuis vos [paramètres](https://huggingface.co/settings/tokens). Ce token est utilisé pour authentifier votre identité au Hub.
Vérifiez bien que vous générez un token avec les accès write si vous voulez upload ou modifier du contenu.

Une fois que vous avez votre token d'authentification, lancez la commande suivante dans votre terminal :

```bash
>>> huggingface-cli login
```

Cette commande devrait vous demaner un token. Copiez-collez le vôtre et appuyez sur *Entrée*. Ensuite, le CLI devrait vous demander si le token
doit aussi être sauvegardé en tant que credential git. Appuyez sur *Entrée* encore (oui par défaut) si vous comptez utiliser `git` en local.
Enfin, le CLI vérifiera auprès du Hub si votre token est valide et l'enregistra en local.

```
_|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
_|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
_|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
_|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
_|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|

To login, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens .
Token: 
Add token as git credential? (Y/n) 
Token is valid (permission: write).
Your token has been saved in your configured git credential helpers (store).
Your token has been saved to /home/wauplin/.cache/huggingface/token
Login successful
```

Alternativement, si vous souhaitez vous connecter sans que le CLI vous demande quoi que ce soit, vous pouvez passer votre token directement depuis
l'invite de commande. Pour que le processus soit plus sécurisé, nous vous recommandons de mettre votre token dans une variable d'environnement
pour éviter de le laisser dans l'historique de votre invite de commande.

```bash
# Utilisation d'une variable d'environnement
>>> huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential 
Token is valid (permission: write).
Your token has been saved in your configured git credential helpers (store).
Your token has been saved to /home/wauplin/.cache/huggingface/token
Login successful
```

## huggingface-cli whoami

Si vous voulez vérifier que vous êtes bien connecté, vous pouvez utiliser `huggingface-cli whoami`. Cette commande ne prend aucune option en entré et print votre nom d'utilisateur et l'organisation dont vous faites parti dans le Hub :

```bash
huggingface-cli whoami                                                                     
Wauplin
orgs:  huggingface,eu-test,OAuthTesters,hf-accelerate,HFSmolCluster
```

Si vous n'êtes pas connecté, un message d'erreur sera renvoyé.

## huggingface-cli logout

Cette commande vous déconnecte. En pratique, elle supprime le token enregistré sur votre machine.

Cette commande ne vous déconnectera pas si vous vous êtes connecté en utilisant la variable d'environnement `HF_TOKEN` (voir [référence](../package_reference/environment_variables#hftoken)). Si c'est le cas, vous devez désactiver la variable dans les paramètres de votre machine.

## huggingface-cli download

Utilisez la commande `huggingface-cli download` pour télécharger des fichiers directement depuis le Hub. En arrière-plan, cette commande utilise
les mêmes helpers [`hf_hub_download`] et [`snapshot_download`] décrits dans le guide [Téléchargement](./download) et affiche le chemin
renvoyé dans le terminal. Dans les exemples ci-dessous, nous verrons les cas d'usage les plus communs. Pour afficher la liste des
options disponibles, vous pouvez lancer la commande:

```bash
huggingface-cli download --help
```

### Télécharger un fichier

Pour télécharger un unique fichier d'un dépôt, mettez le repo_id et le nom du fichier ainsi :

```bash
>>> huggingface-cli download gpt2 config.json
downloading https://huggingface.co/gpt2/resolve/main/config.json to /home/wauplin/.cache/huggingface/hub/tmpwrq8dm5o
(…)ingface.co/gpt2/resolve/main/config.json: 100%|██████████████████████████████████| 665/665 [00:00<00:00, 2.49MB/s]
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10/config.json
```

La commande affichera toujours sur la dernière ligne le chemin vers le fichier.

### Télécharger un dépôt entier

Dans le cas où vous voudriez télécharger tous les fichier d'un dépôt, vous pouvez ne mettre que l'id du dépôt:

```bash
>>> huggingface-cli download HuggingFaceH4/zephyr-7b-beta
Fetching 23 files:   0%|                                                | 0/23 [00:00<?, ?it/s]
...
...
/home/wauplin/.cache/huggingface/hub/models--HuggingFaceH4--zephyr-7b-beta/snapshots/3bac358730f8806e5c3dc7c7e19eb36e045bf720
```

### Télécharger plusieurs fichiers

Vous pouvez aussi télécharger un sous ensemble des fichiers d'un dépôt en une seule commande. Vous pouvez le faire de deux manières. Si vous avez
déjà une liste précise des fichiers à télécharger, vous pouvez simplement les mettre un par un: 

```bash
>>> huggingface-cli download gpt2 config.json model.safetensors
Fetching 2 files:   0%|                                                                        | 0/2 [00:00<?, ?it/s]
downloading https://huggingface.co/gpt2/resolve/11c5a3d5811f50298f278a704980280950aedb10/model.safetensors to /home/wauplin/.cache/huggingface/hub/tmpdachpl3o
(…)8f278a7049802950aedb10/model.safetensors: 100%|██████████████████████████████| 8.09k/8.09k [00:00<00:00, 40.5MB/s]
Fetching 2 files: 100%|████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  3.76it/s]
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10
```

L'autre approche est de fournir des modèles en filtrant les fichiers que vous voulez télécharger en utilisant `--include` et `--exclude`.
Par exemple, si vous voulez télécharger tous les fichiers safetensors depuis [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) à l'exception des fichiers en précision P16:

```bash
>>> huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 --include "*.safetensors" --exclude "*.fp16.*"*
Fetching 8 files:   0%|                                                                         | 0/8 [00:00<?, ?it/s]
...
...
Fetching 8 files: 100%|█████████████████████████████████████████████████████████████████████████| 8/8 (...)
/home/wauplin/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b
```

### Télécharger un dataset ou un space

Les exemples ci-dessus montrent comment télécharger des fichiers depuis un dépôt. Pour télécharger un dataset ou un space, utilisez
l'option `--repo-type`:

```bash
# https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
>>> huggingface-cli download HuggingFaceH4/ultrachat_200k --repo-type dataset

# https://huggingface.co/spaces/HuggingFaceH4/zephyr-chat
>>> huggingface-cli download HuggingFaceH4/zephyr-chat --repo-type space

...
```

### Télécharger une version spécifique

Les exemples ci-dessus montrent comment télécharger le dernier commit sur la branche main. Pour télécharger une version spécifique (hash de commit,
nom de la branche ou tag), utilisez l'option `revision`:

```bash
>>> huggingface-cli download bigcode/the-stack --repo-type dataset --revision v1.1
...
```

### Télécharger vers un dossier local

La manière recommandée (et appliquée par défaut) pour télécharger des fichiers depuis le Hub est d'utiliser le cache-system. Toutefois, dans certains cas, vous aurez besoin de télécharger des fichiers et de les déplacer dans un dossier spécifique. C'est utile pour avoir un fonctionnement proche de celui de git. Vous pouvez le faire en utilisant l'option `--local_dir`.

<Tip warning={true}>

Le téléchargement vers un chemin local a des désavantages. Veuillez vérifier les limitations dans le guide
[Téléchargement](./download#download-files-to-local-folder) avant d'utiliser `--local-dir`.

</Tip>

```bash
>>> huggingface-cli download adept/fuyu-8b model-00001-of-00002.safetensors --local-dir .
...
./model-00001-of-00002.safetensors
```

### Spécifier le chemin du cache

Par défaut, tous les fichiers seront téléchargés dans le chemin du cache définit par la [variable d'environnement](../package_reference/environment_variables#hfhome) `HF_HOME`. Vous pouvez spécifier un cache personnalisé en utilisant `--cache-dir`:

```bash
>>> huggingface-cli download adept/fuyu-8b --cache-dir ./path/to/cache
...
./path/to/cache/models--adept--fuyu-8b/snapshots/ddcacbcf5fdf9cc59ff01f6be6d6662624d9c745
```

### Préciser un token

Pour avoir accès à des dépôts privés ou sécurisés, vous aurez besoin d'un token. Par défaut, le token enregistré en local
(lors de l'utilsation de `huggingface-cli login`) sera utilisé. Si vous voulez vous authentifier de manière explicite, utilisez l'option `--token`:

```bash
>>> huggingface-cli download gpt2 config.json --token=hf_****
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10/config.json
```

### Mode silencieux

Par défaut, la commande `huggingface-cli download` affichera des détails tels que des avertissements, des informations sur les fichiers téléchargés
et des barres de progression. Si vous ne voulez pas de ce type de message, utilisez l'option `--quiet`. Seule la dernière ligne (i.e. le chemin vers le fichier téléchargé) sera affiché. Ceci peut-être utile si vous voulez utiliser l'output d'une autre commande dans un script.

```bash
>>> huggingface-cli download gpt2 --quiet
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10
```

## huggingface-cli upload

Utilisez la commande `huggingface-cli upload` pour upload des fichiers directement dans le Hub. En arrière plan, cette commande utilise les mêmes helpers
que [`upload_file`] et [`upload_folder`] décrits dans le guide [upload](./upload). Dans les exemples ci-dessous, nous verrons les cas d'utilisation les plus communs.
Pour une liste exhaustive des options disponibles, vous pouvez lancer:

```bash
>>> huggingface-cli upload --help
```

### Upload un fichier

L'utilisation par défaut de cette commande est:

```bash
# Utilisation:  huggingface-cli upload [repo_id] [local_path] [path_in_repo]
```

Pour upload le chemin actuel à la racine du dépôt, utilisez:

```bash
>>> huggingface-cli upload mon-super-modele . .
https://huggingface.co/Wauplin/mon-super-modele/tree/main/
```

<Tip>

Si le dépôt n'existe pas encore, il sera créé automatiquement.

</Tip>

Vous pouvez aussi upload un dossier spécifique:

```bash
>>> huggingface-cli upload mon-super-modele ./models .
https://huggingface.co/Wauplin/mon-super-modele/tree/main/
```
Enfin, vous pouvez upload un dossier dans une destination spécifique dans le dépot:

```bash
>>> huggingface-cli upload mon-super-modele ./path/to/curated/data /data/train
https://huggingface.co/Wauplin/mon-super-modele/tree/main/data/train
```

### Upload un seul fichier

Vous pouvez aussi upload un seul fichier en définissant `local_path` pour qu'il pointe vers ce fichier dans votre machine. Si c'est le cas, `path_in_repo` est optionnel et aura, comme valeur par défaut, le nom de votre fichier local:

```bash
>>> huggingface-cli upload Wauplin/mon-super-modele ./models/model.safetensors
https://huggingface.co/Wauplin/mon-super-modele/blob/main/model.safetensors
```

Si vous voulez upload un seul fichier vers un chemin spécifique, choisissez `path_in_repo` de la bonne manière:

```bash
>>> huggingface-cli upload Wauplin/mon-super-modele ./models/model.safetensors /vae/model.safetensors
https://huggingface.co/Wauplin/mon-super-modele/blob/main/vae/model.safetensors
```

### Upload plusieurs fichiers

Pour upload plusieurs fichiers d'un coup depuis un dossier sans upload tout le dossier, vous pouvez utiliser `--include` et `--exclude`. Cette méthode peut aussi être combinée avec l'option `--delete` pour supprimer des fichiers du dépôt tout en uploadant les nouveaux fichiers. Dans l'exemple ci-dessous, nous synchronisons le space local en supprimant les fichiers distants et en uploadant tous les fichiers sauf ceux dans `/logs`:

```bash
# Synchronisation du space local avec le Hub (upload des nouveaux fichier excepté ceux de logs/, supression des fichiers retirés)
>>> huggingface-cli upload Wauplin/space-example --repo-type=space --exclude="/logs/*" --delete="*" --commit-message="Synchronisation du space local avec le Hub"
...
```

### Upload vers un dataset ou un space

Pour upload vers un dataset ou un space, utilisez l'option `--repo-type`:

```bash
>>> huggingface-cli upload Wauplin/mon-super-dataset ./data /train --repo-type=dataset
...
```

### Upload vers une organisation

Pour upload du contenu vers un dépôt appartenant à une organisation aulieu d'un dépôt personnel, vous devez le spécifier explicitement dans le `repo_id`:

```bash
>>> huggingface-cli upload MonOrganisation/mon-super-modele . .
https://huggingface.co/MonOrganisation/mon-super-modele/tree/main/
```

### Upload vers une version spécifique

Par défaut, les fichiers sont upload vers la branche `main`. Si vous voulez upload des fichiers vers une autre branche, utilisez l'option `--revision`

```bash
# Upload des fichiers vers une pull request
>>> huggingface-cli upload bigcode/the-stack . . --repo-type dataset --revision refs/pr/104
...
```

**Note:** Si la `revision` n'existe pas et que `--create-pr` n'est pas utilisé, une branche sera créé automatiquement à partir de la branche `main`.

### Upload et créer une pull request

Si vous n'avez pas la permission de push vers un dépôt, vous devez ouvrir une pull request et montrer aux propriétaires les changements que vous voulez faire. Vous pouvez le faire en utilisant l'option `--create-pr`:

```bash
# Création d'une pull request et upload des fichiers dessus
>>> huggingface-cli upload bigcode/the-stack . . --repo-type dataset --revision refs/pr/104
https://huggingface.co/datasets/bigcode/the-stack/blob/refs%2Fpr%2F104/
```

### Upload a des intervalles réguliers

Dans certains cas, vous aurez peut-être besoin de push des mise à jour régulières vers un dépôt. Par exemple, si vous entrainez un modèle et que vous voulez upload le fichier log toutes les dix minutes. Pour faire ceci, utilisez l'option `--every`:

```bash
# Upload de nouveaux logs toutes les dix minutes
huggingface-cli upload training-model logs/ --every=10
```

### Mettre un message de commit

Utilisez les options `--commit-message` et `--commit-description` pour mettre votre propre message et description pour votre commit aulieu de ceux par défaut.

```bash
>>> huggingface-cli upload Wauplin/mon-super-modele ./models . --commit-message="Epoch 34/50" --commit-description="Accuracy sur le set de validation : 68%, vérifiez le tensorboard pour plus de détails"
...
https://huggingface.co/Wauplin/mon-super-modele/tree/main
```

### Préciser un token

Pour upload des fichiers, vous devez utiliser un token. Par défaut, le token enregistré en local (lors de l'utilisation de `huggingface-cli login`) sera utilisé. Si vous voulez vous authentifier de manière explicite, utilisez l'option `--token`:

```bash
>>> huggingface-cli upload Wauplin/mon-super-modele ./models . --token=hf_****
...
https://huggingface.co/Wauplin/mon-super-modele/tree/main
```

### Mode silencieux

Par défaut, la commande `huggingface-cli upload` affichera des détails tels que des avertissements, des informations sur les fichiers téléchargés
et des barres de progression.Si vous ne voulez pas de ce type de message, utilisez l'option `--quiet`. Seule la dernière ligne (i.e. l'url vers le fichier uploadé) sera affiché. Cette option peut-être utile si vous voulez utiliser l'output d'une autre commande dans un script.

```bash
>>> huggingface-cli upload Wauplin/mon-super-modele ./models . --quiet
https://huggingface.co/Wauplin/mon-super-modele/tree/main
```

## huggingface-cli scan-cache

Scanner le chemin de votre cache peut être utile si vous voulez connaître les dépôts que vous avez téléchargé et l'espace qu'ils prennent sur votre disque. Vous pouvez faire ceci en utilisant `huggingface-cli scan-cache`:

```bash
>>> huggingface-cli scan-cache
REPO ID                     REPO TYPE SIZE ON DISK NB FILES LAST_ACCESSED LAST_MODIFIED REFS                LOCAL PATH
--------------------------- --------- ------------ -------- ------------- ------------- ------------------- -------------------------------------------------------------------------
glue                        dataset         116.3K       15 4 days ago    4 days ago    2.4.0, main, 1.17.0 /home/wauplin/.cache/huggingface/hub/datasets--glue
google/fleurs               dataset          64.9M        6 1 week ago    1 week ago    refs/pr/1, main     /home/wauplin/.cache/huggingface/hub/datasets--google--fleurs
Jean-Baptiste/camembert-ner model           441.0M        7 2 weeks ago   16 hours ago  main                /home/wauplin/.cache/huggingface/hub/models--Jean-Baptiste--camembert-ner
bert-base-cased             model             1.9G       13 1 week ago    2 years ago                       /home/wauplin/.cache/huggingface/hub/models--bert-base-cased
t5-base                     model            10.1K        3 3 months ago  3 months ago  main                /home/wauplin/.cache/huggingface/hub/models--t5-base
t5-small                    model           970.7M       11 3 days ago    3 days ago    refs/pr/1, main     /home/wauplin/.cache/huggingface/hub/models--t5-small

Done in 0.0s. Scanned 6 repo(s) for a total of 3.4G.
Got 1 warning(s) while scanning. Use -vvv to print details.
```

Pour plus de détails sur comment scanner le chemin vers votre cache, consultez le guide [Gérez votre cache](./manage-cache#scan-cache-from-the-terminal).

## huggingface-cli delete-cache

`huggingface-cli delete-cache` est un outil qui vous permet de supprimer des parties de votre cache que vous n'utilisez plus. Cette commande permet de libérer de la mémoire disque. Pour en apprendre plus sur cette commande, consultez le guide [Gérez votre cache](./manage-cache#clean-cache-from-the-terminal). 

## huggingface-cli env

La commande `huggingface-cli env` affiche des détails sur le setup de votre machine. C'est particulièrement utile si vous ouvrez une issue sur [GitHub](https://github.com/huggingface/huggingface_hub) pour aider les mainteneurs à enquêter sur vos problèmes.

```bash
>>> huggingface-cli env

Copy-and-paste the text below in your GitHub issue.

- huggingface_hub version: 0.19.0.dev0
- Platform: Linux-6.2.0-36-generic-x86_64-with-glibc2.35
- Python version: 3.10.12
- Running in iPython ?: No
- Running in notebook ?: No
- Running in Google Colab ?: No
- Token path ?: /home/wauplin/.cache/huggingface/token
- Has saved token ?: True
- Who am I ?: Wauplin
- Configured git credential helpers: store
- FastAI: N/A
- Tensorflow: 2.11.0
- Torch: 1.12.1
- Jinja2: 3.1.2
- Graphviz: 0.20.1
- Pydot: 1.4.2
- Pillow: 9.2.0
- hf_transfer: 0.1.3
- gradio: 4.0.2
- tensorboard: 2.6
- numpy: 1.23.2
- pydantic: 2.4.2
- aiohttp: 3.8.4
- ENDPOINT: https://huggingface.co
- HF_HUB_CACHE: /home/wauplin/.cache/huggingface/hub
- HF_ASSETS_CACHE: /home/wauplin/.cache/huggingface/assets
- HF_TOKEN_PATH: /home/wauplin/.cache/huggingface/token
- HF_HUB_OFFLINE: False
- HF_HUB_DISABLE_TELEMETRY: False
- HF_HUB_DISABLE_PROGRESS_BARS: None
- HF_HUB_DISABLE_SYMLINKS_WARNING: False
- HF_HUB_DISABLE_EXPERIMENTAL_WARNING: False
- HF_HUB_DISABLE_IMPLICIT_TOKEN: False
- HF_HUB_ENABLE_HF_TRANSFER: False
- HF_HUB_ETAG_TIMEOUT: 10
- HF_HUB_DOWNLOAD_TIMEOUT: 10
```