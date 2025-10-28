<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/huggingface_hub-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/huggingface_hub.svg">
    <img alt="logo de la bibliothèque huggingface_hub" src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/huggingface_hub.svg" width="352" height="59" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p> 

<p align="center">
    <i>Le client Python officiel pour le Huggingface Hub.</i>
</p>

<p align="center">
    <a href="https://huggingface.co/docs/huggingface_hub/en/index"><img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/huggingface_hub/index.svg?down_color=red&down_message=offline&up_message=online&label=doc"></a>
    <a href="https://github.com/huggingface/huggingface_hub/releases"><img alt="Version GitHub" src="https://img.shields.io/github/release/huggingface/huggingface_hub.svg"></a>
    <a href="https://github.com/huggingface/huggingface_hub"><img alt="Version PyPI" src="https://img.shields.io/pypi/pyversions/huggingface_hub.svg"></a>
    <a href="https://pypi.org/project/huggingface-hub"><img alt="PyPI - Téléchargements" src="https://img.shields.io/pypi/dm/huggingface_hub"></a>
    <a href="https://codecov.io/gh/huggingface/huggingface_hub"><img alt="Couverture du code" src="https://codecov.io/gh/huggingface/huggingface_hub/branch/main/graph/badge.svg?token=RXP95LE2XL"></a>
</p>

<h4 align="center">
    <p>
        <a href="https://github.com/huggingface/huggingface_hub/blob/main/README.md">English</a> |
        <a href="https://github.com/huggingface/huggingface_hub/blob/main/i18n/README_de.md">Deutsch</a> |
        <b>Français</b> |
        <a href="https://github.com/huggingface/huggingface_hub/blob/main/i18n/README_hi.md">हिंदी</a> |
        <a href="https://github.com/huggingface/huggingface_hub/blob/main/i18n/README_ko.md">한국어</a> |
        <a href="https://github.com/huggingface/huggingface_hub/blob/main/i18n/README_cn.md">中文 (简体)</a>
</h4>

---

**Documentation** : <a href="https://hf.co/docs/huggingface_hub" target="_blank">https://hf.co/docs/huggingface_hub</a>

**Code source** : <a href="https://github.com/huggingface/huggingface_hub" target="_blank">https://github.com/huggingface/huggingface_hub</a>

---

## Bienvenue sur la bibliothèque huggingface_hub

La bibliothèque `huggingface_hub` vous permet d’interagir avec le [Hugging Face Hub](https://huggingface.co/), cette plateforme démocratise le Machine Learning en open source pour les créateurs et les collaborateurs du monde entier. Découvrez des modèles et des jeux de données pré-entraînés pour vos projets, ou amusez-vous avec les milliers d’applications de machine learning hébergées sur le Hub. Vous pouvez aussi créer et partager vos propres modèles, vos jeux de données, vos démos avec la communauté. La bibliothèque `huggingface_hub` vous permet de faire ça simplement avec Python.

## Fonctionnalités clés

- [Télécharger des fichiers](https://huggingface.co/docs/huggingface_hub/en/guides/download) depuis le Hub.
- [Téléverser des fichiers](https://huggingface.co/docs/huggingface_hub/en/guides/upload) vers le Hub.
- [Gérer vos dépôts](https://huggingface.co/docs/huggingface_hub/en/guides/repository).
- [Exécuter l’inférence](https://huggingface.co/docs/huggingface_hub/en/guides/inference) sur des modèles déployés.
- [Rechercher](https://huggingface.co/docs/huggingface_hub/en/guides/search) des modèles, des jeux de données et des Spaces.
- [Partager des Model Cards](https://huggingface.co/docs/huggingface_hub/en/guides/model-cards) pour documenter vos modèles.
- [Interagir avec la communauté](https://huggingface.co/docs/huggingface_hub/en/guides/community) via des PR et des commentaires.

## Installation

Installez le paquet `huggingface_hub` avec [pip](https://pypi.org/project/huggingface-hub/) :

```bash
pip install huggingface_hub
```

Si vous préférez, vous pouvez aussi l’installer avec [conda](https://huggingface.co/docs/huggingface_hub/en/installation#install-with-conda).

Afin de garder le paquet minimal par défaut, `huggingface_hub` vient avec des dépendances optionnelles utiles pour certains cas d’usage. Par exemple, si vous voulez faire de l’inférence, exécutez :

```bash
pip install "huggingface_hub[inference]"
```

Pour en savoir plus sur l'installation et les dépendances optionnelles, consultez le [guide d’installation](https://huggingface.co/docs/huggingface_hub/en/installation).

## Démarrage rapide

### Télécharger des fichiers

Télécharger un seul fichier

```py
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="tiiuae/falcon-7b-instruct", filename="config.json")
```

Ou un dépôt entier

```py
from huggingface_hub import snapshot_download

snapshot_download("stabilityai/stable-diffusion-2-1")
```

Les fichiers seront téléchargés dans un dossier de cache local. Pour plus d'informations, consultez [ce guide](https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache).

### Connexion

Le Hugging Face Hub utilise des jetons d'authentification pour les applications (voir [docs](https://huggingface.co/docs/hub/security-tokens)). Pour vous connecter sur votre machine, exécutez la commande CLI suivante :

```bash
hf auth login
# ou en utilisant une variable d'environnement
hf auth login --token $HUGGINGFACE_TOKEN
```

### Créer un dépôt

```py
from huggingface_hub import create_repo

create_repo(repo_id="super-cool-model")
```

### Téléverser des fichiers

Téléverser un seul fichier

```py
from huggingface_hub import upload_file

upload_file(
    path_or_fileobj="/home/lysandre/dummy-test/README.md",
    path_in_repo="README.md",
    repo_id="lysandre/test-model",
)
```

Ou un dossier entier

```py
from huggingface_hub import upload_folder

upload_folder(
    folder_path="/path/to/local/space",
    repo_id="username/my-cool-space",
    repo_type="space",
)
```

Pour plus de détails, consultez le [guide de téléversement](https://huggingface.co/docs/huggingface_hub/en/guides/upload).

## Intégrer votre bibliothèque au Hub

Nous collaborons avec des bibliothèques de machine learning open source pour offrir gratuitement de l’hébergement et du versionnage de modèles. Vous pouvez trouver les intégrations existantes [ici](https://huggingface.co/docs/hub/libraries).

Les avantages sont :

- Hébergement gratuit de modèles et des jeux de données pour les bibliothèques et leurs utilisateurs.
- Versionnage des fichiers — y compris très volumineux — grâce à Git.
- Widgets interactifs dans le navigateur pour essayer les modèles mis en ligne.
- N’importe qui peut téléverser un nouveau modèle pour votre bibliothèque, il suffit d’ajouter l’étiquette correspondante pour que le modèle soit disponible.
- Téléchargements rapides ! Nous utilisons CloudFront (un CDN) pour géo-répliquer les téléchargements afin qu’ils soient ultra rapides partout dans le monde.
- Statistiques d’utilisation et d’autres fonctionnalités à venir.

Si vous souhaitez intégrer votre bibliothèque, n’hésitez pas à ouvrir une issue pour lancer la discussion. Nous avons écrit un [guide pas à pas](https://huggingface.co/docs/hub/adding-a-library) avec ❤️ pour vous accompagner.

## Les contributions (demandes de fonctionnalité, bugs, etc.) sont les bienvenues 💙💚💛💜🧡❤️

Tout le monde est le bienvenu pour contribuer, et nous valorisons la contribution de chacun. Le code n’est pas la seule façon d’aider la communauté.
Répondre aux questions, aider les autres, prendre contact et améliorer la documentation sont immensément précieux pour la communauté.
Nous avons écrit un [guide de contribution](https://github.com/huggingface/huggingface_hub/blob/main/CONTRIBUTING.md) pour vous aider à comprendre comment commencer à contribuer à ce dépôt.