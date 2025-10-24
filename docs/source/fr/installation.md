<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Installation

Avant de commencer, vous allez avoir besoin de préparer votre environnement
en installant les packages appropriés.

`huggingface_hub` est testée sur **Python 3.9+**.

## Installation avec pip

Il est fortement recommandé d'installer `huggingface_hub` dans un [environnement virtuel](https://docs.python.org/3/library/venv.html).
Si vous n'êtes pas familier avec les environnements virtuels Python, suivez ce [guide](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/). Un environnement virtuel sera utile lorsque vous devrez gérer des plusieurs projets en parallèle
afin d'éviter les problèmes de compatibilité entre les différentes dépendances.

Commencez par créer un environnement virtuel à l'emplacement de votre projet:

```bash
python -m venv .env
```

Activez l'environnement virtuel sur Linux et macOS:

```bash
source .env/bin/activate
```

Activez l'environnement virtuel sur Windows:

```bash
.env/Scripts/activate
```

Maintenant, vous êtes prêts à installer `hugginface_hub` [depuis PyPi](https://pypi.org/project/huggingface-hub/):

```bash
pip install --upgrade huggingface_hub
```

Une fois l'installation terminée, rendez-vous à la section [vérification](#verification-de-l-installation) pour s'assurer que tout fonctionne correctement.

### Installation des dépendances optionnelles

Certaines dépendances de `huggingface_hub` sont [optionnelles](https://setuptools.pypa.io/en/latest/userguide/dependency_management.html#optional-dependencies) car elles ne sont pas nécessaire pour faire marcher les fonctionnalités principales de `huggingface_hub`.
Toutefois, certaines fonctionnalités de `huggingface_hub` ne seront pas disponibles si les dépendances optionnelles ne sont pas installées

Vous pouvez installer des dépendances optionnelles via `pip`:
```bash
#Installation des dépendances spécifiques à Pytorch et au CLI.
pip install 'huggingface_hub[cli,torch]'
```

Voici une liste des dépendances optionnelles dans `huggingface_hub`:
- `cli` fournit une interface d'invite de commande plus pratique pour `huggingface_hub`.
- `fastai`, `torch` sont des dépendances pour utiliser des fonctionnalités spécifiques à un framework.
- `dev` permet de contribuer à la librairie. Cette dépendance inclut `testing` (pour lancer des tests), `typing` (pour lancer le vérifieur de type) et `quality` (pour lancer des linters).



### Installation depuis le code source

Dans certains cas, il est intéressant d'installer `huggingface_hub` directement depuis le code source.
Ceci vous permet d'utiliser la version `main`, contenant les dernières mises à jour, plutôt que
d'utiliser la dernière version stable. La version `main` est utile pour rester à jour sur les
derniers développements, par exemple si un bug est corrigé depuis la dernière version officielle
mais que la nouvelle version n'a pas encore été faite.

Toutefois, cela signifie que la version `main` peut ne pas être stable. Nous travaillons
afin de rendre la version `main` aussi stable que possible, et la plupart des problèmes sont résolus
en quelques heures ou jours. Si vous avez un problème, ouvrez une issue afin que
nous puissions la régler au plus vite !

```bash
pip install git+https://github.com/huggingface/huggingface_hub
```

Lorsque vous installez depuis le code source, vous pouvez préciser la branche depuis laquelle installer. Cela permet de tester une nouvelle fonctionnalité ou un bug-fix qui n'a pas encore été merge:

```bash
pip install git+https://github.com/huggingface/huggingface_hub@ma-branche
```

Une fois l'installation terminée, rendez-vous à la section [vérification](#verification-de-l-installation) pour s'assurer que tout fonctionne correctement.

### Installation éditable

L'installation depuis le code source vous permet de mettre en place une [installation éditable](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs). Cette installation sert surtout si vous comptez contribuer à `huggingface_hub`
et que vous avez besoin de tester rapidement des changements dans le code. Pour cela, vous devez cloner le projet `huggingface_hub` sur votre machine.

```bash
# Commencez par cloner le dépôt en local
git clone https://github.com/huggingface/huggingface_hub.git

# Ensuite, installez-le avec le flag -e
cd huggingface_hub
pip install -e .
```

Python regardera maintenant à l'intérieur du dossier dans lequel vous avez cloné le dépôt en
plus des chemins de librairie classiques. Par exemple, si vos packages Python sont installés dans
`./.venv/lib/python3.13/site-packages/`, Python regardera aussi dans le dossier que vous avez
cloné `./huggingface_hub/`.

## Installation avec conda

Si vous avez plutôt l'habitude d'utiliser conda, vous pouvez installer `huggingface_hub` en utilisant le [channel conda-forge](https://anaconda.org/conda-forge/huggingface_hub):


```bash
conda install -c conda-forge huggingface_hub
```

Une fois l'installation terminée, rendez-vous à la section [vérification](#verification-de-l-installation) pour s'assurer que tout fonctionne correctement.

## Vérification de l'installation

Une fois installée, vérifiez que `huggingface_hub` marche correctement en lançant la commande suivante:

```bash
python -c "from huggingface_hub import model_info; print(model_info('gpt2'))"
```

Cette commande va récupérer des informations sur le modèle [gpt2](https://huggingface.co/gpt2) depuis le Hub.
La sortie devrait ressembler à ça:

```text
Model Name: gpt2
Tags: ['pytorch', 'tf', 'jax', 'tflite', 'rust', 'safetensors', 'gpt2', 'text-generation', 'en', 'doi:10.57967/hf/0039', 'transformers', 'exbert', 'license:mit', 'has_space']
Task: text-generation
```

## Les limitations Windows

Afin de démocratiser le machine learning au plus grand nombre, nous avons développé `huggingface_hub`
de manière cross-platform et en particulier, pour qu'elle fonctionne sur une maximum de systèmes d'exploitation différents. Toutefois
`huggingface_hub` connaît dans certains cas des limitations sur Windows.
Nous avons listé ci-dessous les problèmes connus. N'hésitez pas à nous signaler si vous rencontrez un problème
non documenté en ouvrant une [issue sur Github](https://github.com/huggingface/huggingface_hub/issues/new/choose).

- Le cache de `huggingface_hub` a besoin des symlinks pour mettre en cache les fichiers installés depuis le Hub.
Sur windows, vous devez activer le mode développeur pour lancer ou lancer votre script en tant qu'administrateur
afin de faire fonctionner les symlinks. S'ils ne sont pas activés, le système de cache fonctionnera toujours mais
de manière sous-optimale. Consultez les [limitations du cache](./guides/manage-cache#limitations) pour plus de détails.
- Les noms de fichiers sur le Hub peuvent avoir des caractères spéciaux (par exemple `"path/to?/my/file"`).
Windows est plus restrictif sur les [caractères spéciaux](https://learn.microsoft.com/en-us/windows/win32/intl/character-sets-used-in-file-names)
ce qui rend ces fichiers ininstallables sur Windows. Heureusement c'est un cas assez rare.
Contactez le propriétaire du dépôt si vous pensez que c'est une erreur ou contactez nous
pour que nous cherchions une solution.


## Prochaines étapes

Une fois que `huggingface_hub` est installé correctement sur votre machine, vous aurez peut-être besoin de
[configurer les variables d'environnement](package_reference/environment_variables) ou de [lire un de nos guides](guides/overview)
pour vous lancer.
