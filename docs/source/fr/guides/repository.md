<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Créer et gérer un dépôt

Le Hugging Face Hub est une collection de dépôts Git. [Git](https://git-scm.com/) est un outil largement utilisé dans le développement logiciel
pour versionner facilement des projets lors de travail collaboratif. Ce guide vous montrera comment interagir avec les
dépôts sur le Hub, notamment :

- Créer et supprimer un dépôt.
- Gérer les branches et les tags.
- Renommer votre dépôt.
- Mettre à jour la visibilité de votre dépôt.
- Gérer une copie locale de votre dépôt.
et plus encore !

> [!WARNING]
> Si vous avez l'habitude de travailler avec des plateformes comme GitLab/GitHub/Bitbucket, votre premier réflexe
> pourrait être d'utiliser le CLI `git` pour cloner votre dépôt (`git clone`), commiter des modifications (`git add, git commit`) et les pousser
> (`git push`). C'est exactement ce qu'il faut faire lors de l'utilisation du Hugging Face Hub. Cependant, l'ingénierie logicielle et le machine learning n'ont
> pas les mêmes exigences et workflows. Les dépôts de modèles peuvent maintenir de gros fichiers de poids de modèles pour différents
> frameworks et outils, donc cloner le dépôt peut conduire à maintenir de gros dossiers locaux avec des tailles massives. Par
> conséquent, il peut être plus efficace d'utiliser nos méthodes HTTP personnalisées. Vous pouvez lire notre page d'explication sur le [paradigme Git vs HTTP](../concepts/git_vs_http)
> pour plus de détails.

Si vous souhaitez créer et gérer un dépôt sur le Hub, votre machine doit être authentifiée. Si ce n'est pas le cas, veuillez vous référer à
[cette section](../quick-start#authentication). Dans le reste de ce guide, nous supposerons que votre machine est connectée à un compte Hugging Face.

## Création et suppression de dépôt

La première étape consiste à savoir comment créer et supprimer des dépôts. Vous ne pouvez gérer que les dépôts que vous possédez (sous
votre nom d'utilisateur) ou des organisations dans lesquelles vous avez des permissions d'écriture.

### Créer un dépôt

Créez un dépôt vide avec [`create_repo`] et donnez-lui un nom avec le paramètre `repo_id`. Le `repo_id` est votre espace de noms suivi du nom du dépôt : `username_or_org/repo_name`.

```py
>>> from huggingface_hub import create_repo
>>> create_repo("lysandre/test-model")
'https://huggingface.co/lysandre/test-model'
```

Ou via le CLI :

```bash
>>> hf repo create lysandre/test-model
Successfully created lysandre/test-model on the Hub.
Your repo is now available at https://huggingface.co/lysandre/test-model
```

Par défaut, [`create_repo`] crée un dépôt de modèle. Mais vous pouvez utiliser le paramètre `repo_type` pour spécifier un autre type de dépôt. Par exemple, si vous voulez créer un dépôt de dataset :

```py
>>> from huggingface_hub import create_repo
>>> create_repo("lysandre/test-dataset", repo_type="dataset")
'https://huggingface.co/datasets/lysandre/test-dataset'
```

Ou via le CLI :

```bash
>>> hf repo create lysandre/test-dataset --repo-type dataset
```

Lorsque vous créez un dépôt, vous pouvez définir la visibilité de votre dépôt avec le paramètre `private`.

```py
>>> from huggingface_hub import create_repo
>>> create_repo("lysandre/test-private", private=True)
```

Ou via le CLI :

```bash
>>> hf repo create lysandre/test-private --private
```

Si vous souhaitez changer la visibilité du dépôt ultérieurement, vous pouvez utiliser la fonction [`update_repo_settings`].

> [!TIP]
> Si vous faites partie d'une organisation avec un plan Enterprise, vous pouvez créer un dépôt dans un groupe de ressources spécifique en passant `resource_group_id` comme paramètre à [`create_repo`]. Les groupes de ressources sont une fonctionnalité de sécurité pour contrôler quels membres de votre organisation peuvent accéder à une ressource donnée. Vous pouvez obtenir l'ID du groupe de ressources en le copiant depuis l'URL de la page de paramètres de votre organisation sur le Hub (par exemple `"https://huggingface.co/organizations/huggingface/settings/resource-groups/66670e5163145ca562cb1988"` => `"66670e5163145ca562cb1988"`). Pour plus de détails sur les groupes de ressources, consultez ce [guide](https://huggingface.co/docs/hub/en/security-resource-groups).

### Supprimer un dépôt

Supprimez un dépôt avec [`delete_repo`]. Assurez-vous de bien vouloir supprimer un dépôt car il s'agit d'un processus irréversible ! Aucune sauvegarde n'est effectuée !

Spécifiez le `repo_id` du dépôt que vous souhaitez supprimer :

```py
>>> delete_repo(repo_id="lysandre/my-corrupted-dataset", repo_type="dataset")
```

Ou via le CLI :

```bash
>>> hf repo delete lysandre/my-corrupted-dataset --repo-type dataset
```

### Dupliquer un dépôt (uniquement pour les Spaces)

Dans certains cas, vous souhaitez copier le dépôt de quelqu'un d'autre pour l'adapter à votre cas d'usage.
C'est possible pour les Spaces en utilisant la méthode [`duplicate_space`]. Elle dupliquera le dépôt entier.
Vous devrez toujours configurer vos propres paramètres (matériel, temps de veille, stockage, variables et secrets). Consultez notre guide [Gérer votre Space](./manage-spaces) pour plus de détails.

```py
>>> from huggingface_hub import duplicate_space
>>> duplicate_space("multimodalart/dreambooth-training", private=False)
RepoUrl('https://huggingface.co/spaces/nateraw/dreambooth-training',...)
```

## Uploader et télécharger des fichiers

Maintenant que vous avez créé votre dépôt, vous devrez pousser des modifications et télécharger des fichiers depuis celui-ci.

Ces 2 sujets méritent leurs propres guides. Veuillez vous référer aux guides [upload](./upload) et [download](./download)
pour apprendre comment utiliser votre dépôt.


## Branches et tags

Les dépôts Git utilisent souvent des branches pour stocker différentes versions d'un même dépôt.
Les tags peuvent également être utilisés pour marquer un état spécifique de votre dépôt, par exemple lors de la publication d'une version.
Plus généralement, les branches et les tags sont appelés [références git](https://git-scm.com/book/en/v2/Git-Internals-Git-References).

### Créer des branches et des tags

Vous pouvez créer de nouvelles branches et tags en utilisant [`create_branch`] et [`create_tag`] :

```py
>>> from huggingface_hub import create_branch, create_tag

# Créer une branche sur un dépôt Space depuis la branche `main`
>>> create_branch("Matthijs/speecht5-tts-demo", repo_type="space", branch="handle-dog-speaker")

# Créer un tag sur un dépôt Dataset depuis la branche `v0.1-release`
>>> create_tag("bigcode/the-stack", repo_type="dataset", revision="v0.1-release", tag="v0.1.1", tag_message="Bump release version.")
```

Ou via le CLI :

```bash
>>> hf repo branch create Matthijs/speecht5-tts-demo handle-dog-speaker --repo-type space
>>> hf repo tag create bigcode/the-stack v0.1.1 --repo-type dataset --revision v0.1-release -m "Bump release version."
```

Vous pouvez utiliser les fonctions [`delete_branch`] et [`delete_tag`] de la même manière pour supprimer une branche ou un tag, ou `hf repo branch delete` et `hf repo tag delete` respectivement dans le CLI.


### Lister toutes les branches et tags

Vous pouvez également lister les références git existantes d'un dépôt en utilisant [`list_repo_refs`] :

```py
>>> from huggingface_hub import list_repo_refs
>>> list_repo_refs("bigcode/the-stack", repo_type="dataset")
GitRefs(
   branches=[
         GitRefInfo(name='main', ref='refs/heads/main', target_commit='18edc1591d9ce72aa82f56c4431b3c969b210ae3'),
         GitRefInfo(name='v1.1.a1', ref='refs/heads/v1.1.a1', target_commit='f9826b862d1567f3822d3d25649b0d6d22ace714')
   ],
   converts=[],
   tags=[
         GitRefInfo(name='v1.0', ref='refs/tags/v1.0', target_commit='c37a8cd1e382064d8aced5e05543c5f7753834da')
   ]
)
```

## Modifier les paramètres du dépôt

Les dépôts viennent avec des paramètres que vous pouvez configurer. La plupart du temps, vous devrez le faire manuellement dans la
page de paramètres du dépôt dans votre navigateur. Vous devez avoir un accès en écriture à un dépôt pour le configurer : soit le posséder, soit faire partie d'une organisation. Dans cette section, nous verrons les paramètres que vous pouvez également configurer avec du code en utilisant `huggingface_hub`.

Certains paramètres sont spécifiques aux Spaces (matériel, variables d'environnement,...). Pour les configurer, veuillez vous référer à notre guide [Gérer vos Spaces](../guides/manage-spaces).

### Mettre à jour la visibilité

Un dépôt peut être public ou privé. Un dépôt privé n'est visible que par vous ou les membres de l'organisation dans laquelle le dépôt est situé. Changez un dépôt en privé comme indiqué ci-dessous :

```py
>>> from huggingface_hub import update_repo_settings
>>> update_repo_settings(repo_id=repo_id, private=True)
```

Ou via le CLI :

```bash
>>> hf repo settings lysandre/test-private --private true
```

### Configurer l'accès restreint

Pour donner plus de contrôle sur la façon dont les dépôts sont utilisés, le Hub permet aux auteurs de dépôts d'activer **les demandes d'accès** pour leurs dépôts. Les utilisateurs doivent accepter de partager leurs informations de contact (nom d'utilisateur et adresse e-mail) avec les auteurs du dépôt pour accéder aux fichiers lorsque cette option est activée. Un dépôt avec des demandes d'accès activées est appelé un **dépôt restreint**.

Vous pouvez définir un dépôt comme restreint en utilisant [`update_repo_settings`] :

```py
>>> from huggingface_hub import HfApi

>>> api = HfApi()
>>> api.update_repo_settings(repo_id=repo_id, gated="auto")  # Définir un accès restreint automatique pour un modèle
```

Ou via le CLI :

```bash
>>> hf repo settings lysandre/test-private --gated auto
```

### Renommer votre dépôt

Vous pouvez renommer votre dépôt sur le Hub en utilisant [`move_repo`]. En utilisant cette méthode, vous pouvez également déplacer le dépôt d'un utilisateur vers
une organisation. Cependant, il y a [quelques limitations](https://hf.co/docs/hub/repositories-settings#renaming-or-transferring-a-repo)
dont vous devez être conscient. Par exemple, vous ne pouvez pas transférer votre dépôt à un autre utilisateur.

```py
>>> from huggingface_hub import move_repo
>>> move_repo(from_id="Wauplin/cool-model", to_id="huggingface/cool-model")
```

Ou via le CLI :

```bash
>>> hf repo move Wauplin/cool-model huggingface/cool-model
```
