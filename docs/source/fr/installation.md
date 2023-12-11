<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Installation

Avant de commmencer l'installation, vous allez avoir besoin de préparer votre environnement
en installant les packages appropriés.

`huggingface_hub` est testé sur **Python 3.8+**.

## Installation avec pip

Il est fortement recommandé d'installer `huggingface_hub` dans un [environnement virtuel](https://docs.python.org/3/library/venv.html).
Si vous êtes familier avec les environnements virtuels python, regardez plutôt ce [guide](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/). Un environnement virtuel sera utile lorsque vous devrez gérer des projets différents
et éviter les problèmes de compatibilité entre les différetnes dépendances.

Commencez par créer un environnement virtuel dans le chemin de votre projet:

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

Maintenant, vous êtes prêts à installer `hugginface_hub` [depuis PyPi ](https://pypi.org/project/huggingface-hub/):

```bash
pip install --upgrade huggingface_hub
```

Une fois l'installation finie, [vérifiez](#check-installation) que tout marche correctement.

### Installation des dépendances optionnelles

Certaines dépendances de `huggingface_hub` sont [optionnelles](https://setuptools.pypa.io/en/latest/userguide/dependency_management.html#optional-dependencies) car elles ne sont pas nécessaire pour faire marcher les fonctionnalités principales de `huggingface_hub`.
Toutefois, certaines fonctionnalités de `huggingface_hub` ne seront pas disponibles si les dépendancces optionnelles ne sont pas installées

Vous pouvez installer des dépendances optionelles via `pip`:
```bash
#Installation des dépendances pour les fonctionnalités spécifiques à Tensorflow.
#/!\ Attention : cette commande n'est pas équivalente à `pip install tensorflow`.
pip install 'huggingface_hub[tensorflow]'

#Installation des dépendances spécifiques à Pytorch et au CLI.
pip install 'huggingface_hub[cli,torch]'
```

Voici une liste des dépendances optionnelles dans `huggingface_hub`:
- `cli` fournit une interface d'invite de commande plus pratique pour `huggingface_hub`.
- `fastai`, `torch` et `tensorflow` sont des dépendances pour utiliser des fonctionnalités spécifique à un framework.
- `dev` permet de contribuer à la librairie. Cette dépendance inclut `testing` (pour lancer des tests), `typing` (pour lancer le vérifieur de type) et `quality` (pour lancer des linters). 



### Installation depuis la source

Dans certains cas, il est intéressant d'installer `huggingface_hub` directement depuis la source.
Ceci vous permet d'utiliser la version `main`, contenant les dernières mises à jour, plutôt que
d'utiliser la dernière version stable. La version `main` est utile pour rester à jour sur les
derniers développements, par exemple si un bug est réglé depuis la dernière sortie officielle
mais que la nouvelle sortie n'a pas encore été faite.

Toutefois, cela signifie que la version `main` pourrait ne pas stable. Nous travaillons
afin de rendre la version `main` optionnelle, et la pluspart des problèmes sont résolus
en quelques heure ou en une journée. Si vous avez un problème, ouvrez une Issue afin que
nous puissions la régler encore plus vite !

```bash
pip install git+https://github.com/huggingface/huggingface_hub
```

Lorsque vous faites l'installation depuis le code source, vous pouvez aussi préciser une 
branche spécifique. C'est utile si vous voulez tester une nouvelle fonctionnalité ou un
nouveau bug-fix qui n'a pas encore été merge:

```bash
pip install git+https://github.com/huggingface/huggingface_hub@ma-branche
```

Une fois fini, [vérifiez l'installation](#check-installation).

### Installation éditable

L'installation depuis le code source vous permet de mettre en place une [installation éditable](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs). Cette installation est plus avancée et sert surtout si vous comptez contribuer à `huggingface_hub`
et que vous avez besoin de tester des changements dans le code. Vous devez cloner une copie locale de `huggingface_hub` sur votre machine.

```bash
#D'abord, clonez le dépôt en local
git clone https://github.com/huggingface/huggingface_hub.git

# Ensuite, installez avec le flag -e
cd huggingface_hub
pip install -e .
```

Ces commandes lieront le dossier que vous avez cloné avec le dépôt et vos chemins de librairies Python.
Python regardera maintenant à l'intérieur du dossier dans lequel vous avez cloné le dépôt et
These commands will link the folder you cloned the repository to and your Python library paths.
Python will now look inside the folder you cloned to in addition to the normal library paths.
For example, if your Python packages are typically installed in `./.venv/lib/python3.11/site-packages/`,
Python will also search the folder you cloned `./huggingface_hub/`.

## Install with conda

If you are more familiar with it, you can install `huggingface_hub` using the [conda-forge channel](https://anaconda.org/conda-forge/huggingface_hub):


```bash
conda install -c conda-forge huggingface_hub
```

Once done, [check installation](#check-installation) is working correctly.

## Check installation

Once installed, check that `huggingface_hub` works properly by running the following command:

```bash
python -c "from huggingface_hub import model_info; print(model_info('gpt2'))"
```

This command will fetch information from the Hub about the [gpt2](https://huggingface.co/gpt2) model.
Output should look like this:

```text
Model Name: gpt2
Tags: ['pytorch', 'tf', 'jax', 'tflite', 'rust', 'safetensors', 'gpt2', 'text-generation', 'en', 'doi:10.57967/hf/0039', 'transformers', 'exbert', 'license:mit', 'has_space']
Task: text-generation
```

## Windows limitations

With our goal of democratizing good ML everywhere, we built `huggingface_hub` to be a
cross-platform library and in particular to work correctly on both Unix-based and Windows
systems. However, there are a few cases where `huggingface_hub` has some limitations when
run on Windows. Here is an exhaustive list of known issues. Please let us know if you
encounter any undocumented problem by opening [an issue on Github](https://github.com/huggingface/huggingface_hub/issues/new/choose).

- `huggingface_hub`'s cache system relies on symlinks to efficiently cache files downloaded
from the Hub. On Windows, you must activate developer mode or run your script as admin to
enable symlinks. If they are not activated, the cache-system still works but in an non-optimized
manner. Please read [the cache limitations](./guides/manage-cache#limitations) section for more details.
- Filepaths on the Hub can have special characters (e.g. `"path/to?/my/file"`). Windows is
more restrictive on [special characters](https://learn.microsoft.com/en-us/windows/win32/intl/character-sets-used-in-file-names)
which makes it impossible to download those files on Windows. Hopefully this is a rare case.
Please reach out to the repo owner if you think this is a mistake or to us to figure out
a solution.


## Next steps

Once `huggingface_hub` is properly installed on your machine, you might want
[configure environment variables](package_reference/environment_variables) or [check one of our guides](guides/overview) to get started.