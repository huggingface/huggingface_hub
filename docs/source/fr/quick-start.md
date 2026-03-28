<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Démarrage rapide

Le [Hugging Face Hub](https://huggingface.co/) est le meilleur endroit pour partager des
modèles de machine learning, des démos, des datasets et des métriques. La librairie
`huggingface_hub` vous aide à intéragir avec le Hub sans sortir de votre environnement de
développement. Vous pouvez: créer et gérer des dépôts facilement, télécharger et upload des
fichiers, et obtenir des modèles et des métadonnées depuis le Hub.

## Installation

Pour commencer, installez la librairie `huggingface_hub`:

```bash
pip install --upgrade huggingface_hub
```

Pour plus de détails, vérifiez le guide d'[installation](installation)

## Télécharger des fichiers

Les dépôts sur le Hub utilisent le versioning Git, les utilisateurs peuvent
télécharger un fichier, ou un dépôt entier. Vous pouvez utiliser la fonction [`hf_hub_download`]
pour télécharger des fichiers. Cette fonction téléchargera et mettra dans le cache un fichier
sur votre disque local. La prochaine fois que vous aurez besoin de ce fichier, il sera chargé
depuis votre cache de façon à ce que vous n'ayez pas besoin de le retélécharger.

Vous aurez besoin de l'id du dépôt et du nom du fichier que vous voulez télécharger.
Par exemple, pour télécharger le fichier de configuration du
modèle [Pegasus](https://huggingface.co/google/pegasus-xsum):

```py
>>> from huggingface_hub import hf_hub_download
>>> hf_hub_download(repo_id="google/pegasus-xsum", filename="config.json")
```

Pour télécharger une version spécifique du fichier, utilisez le paramètre `revision` afin
de spécifier le nom de la branche, le tag ou le hash de commit. Si vous décidez d'utiliser
le hash de commit, vous devez renseigner le hash entier et pas le hash court de 7 caractères:

```py
>>> from huggingface_hub import hf_hub_download
>>> hf_hub_download(
...     repo_id="google/pegasus-xsum", 
...     filename="config.json", 
...     revision="4d33b01d79672f27f001f6abade33f22d993b151"
... )
```

Pour plus de détails et d'options, consultez la réference de l'API pour [`hf_hub_download`].

## Connexion

Dans la plupart des cas, vous devez être connectés avec un compte Hugging Face pour interagir
avec le Hub: pour télécharger des dépôts privés, upload des fichiers, créer des pull
requests...
[Créez un compte](https://huggingface.co/join) si vous n'en avez pas déjà un et connectez
vous pour obtenir votre [token d'authentification](https://huggingface.co/docs/hub/security-tokens)
depuis vos [paramètres](https://huggingface.co/settings/tokens). Le token
est utilisé pour authentifier votre identité au Hub.

Une fois que vous avez votre token d'authentification, lancez la commande suivante
dans votre terminal:

```bash
hf auth login
# ou en utilisant une varible d'environnement:
hf auth login --token $HUGGINGFACE_TOKEN
```

Sinon, vous pouvez vous connecter en utilisant [`login`] dans un notebook ou
un script:

```py
>>> from huggingface_hub import login
>>> login()
```

Il est aussi possible de se connecter automatiquement sans qu'on vous demande votre token en
passant le token dans [`login`] de cette manière: `login(token="hf_xxx")`. Si vous choisissez
cette méthode, faites attention lorsque vous partagez votre code source. Une bonne pratique est
de charger le token depuis un trousseau sécurisé au lieu de l'enregistrer en clair dans votre
codebase/notebook.

Vous ne pouvez être connecté qu'à un seul compte à la fois. Si vous connectez votre machine à un autre compte,
vous serez déconnecté du premier compte. Vérifiez toujours le compte que vous utilisez avec la commande
`hf auth whoami`. Si vous voulez gérer plusieurs compte dans le même script, vous pouvez passer votre
token à chaque appel de méthode. C'est aussi utile si vous ne voulez pas sauvegarder de token sur votre machine.

> [!WARNING]
> Une fois que vous êtes connectés, toutes les requêtes vers le Hub (même les méthodes qui ne nécessite pas explicitement
> d'authentification) utiliseront votre token d'authentification par défaut. Si vous voulez supprimer l'utilisation implicite
> de votre token, vous devez définir la variable d'environnement `HF_HUB_DISABLE_IMPLICIT_TOKEN`.

## Créer un dépôt

Une fois que vous avez créé votre compte et que vous vous êtes connectés,
vous pouvez créer un dépôt avec la fonction [`create_repo`]:

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.create_repo(repo_id="super-cool-model")
```

Si vous voulez que votre dépôt soit privé, alors:

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.create_repo(repo_id="super-cool-model", private=True)
```

Les dépôts privés ne seront visible que par vous.

> [!TIP]
> Pour créer un dépôt ou push du contenu sur le Hub, vous devez fournir un token
> d'authentification qui a les permissions `write`. Vous pouvez choisir la permission
> lorsque vous générez le token dans vos [paramètres](https://huggingface.co/settings/tokens).

## Upload des fichiers

Utilisez la fonction [`upload_file`] pour ajouter un fichier à votre dépôt.
Vous devez spécifier:

1. Le chemin du fichier à upload.
2. Le chemin du fichier dans le dépôt.
3. L'id du dépôt dans lequel vous voulez ajouter le fichier.

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.upload_file(
...     path_or_fileobj="/home/lysandre/dummy-test/README.md",
...     path_in_repo="README.md",
...     repo_id="lysandre/test-model",
... )
```

Pour upload plus d'un fichier à la fois, consultez le guide [upload](./guides/upload)
qui détaille plusieurs méthodes pour upload des fichiers (avec ou sans Git).

## Prochaines étapes

La librairie `huggingface_hub` permet à ses utilisateurs d'intéragir facilementavec le Hub via
Python. Pour en apprendre plus sur comment gérer vos fichiers
et vos dépôts sur le Hub, nous vous recommandons de lire notre [guide conceptuel](./guides/overview)
pour :

- [Gérer votre dépôt](./guides/repository).
- [Télécharger](./guides/download) des fichiers depuis le Hub.
- [Upload](./guides/upload) des fichiers vers le Hub.
- [Faire des recherches dans le Hub](./guides/search) pour votre modèle ou dataset.
- [Accédder à l'API d'inférence](./guides/inference) pour faire des inférences rapides.