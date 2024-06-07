<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Créer et gérer un dépôt

Le Hub Hugging Face est une collection de dépôt git. [Git](https://git-scm.com/) est un outil utilisé par tous les développeurs
de software, il permet de versionner facilement des projet lors de travaux en équipe. Ce guide vous montrera commen intéragir
avec les dépôts sur le Hub, ne particulier:

- Créer et supprimer un dépôt.
- Gérer les branches et les tags.
- Renommer votre dépôt.
- Mettre à jour la visibilité de votre dépôt.
- Gérer une copie local de votre dépôt.

<Tip warning={true}>

Si vous êtes habitués à utiliser des plateformes telles que GitLab/GitHub/Bitbucket, votre premier
instinct sera peut-être d'utiliser le CLI `git` pour cloner votre dépôt (`git clone`), commit les changements
(`git add, git commit`) et les push (`git push`). C'est faisable lors de l'utilisation d'Hugging Face Hub.
Cependant, le développement d'application et le machine learning n'ont pas les même besoins et workflow. Les dépôts de
modèles ont souvent des fichiers avec les poids du modèle très volumineux pour différents frameworks et outils, cloner
un dépôt peut donc demander de gérer des fichiers de très grande taille en local. Ainsi, il peut être plus efficace d'utiliser
nos méthodes HTTP personnalisées. Vous pouvez lire notre page de docuemntation [paradigme Git vs HTTP](../concepts/git_vs_http)
pour plus de détails.

</Tip>

Si vous voulez créer et gérer un dépôt sur le Hub, votre machine doit être authentifiée. Si vous ne l'êtes pas, consultez
[this section](../quick-start#login). Dans le reste de ce guide, nous supposerons que votre machien est connectée.

## Création et suppression d'un dépôt

La première étape est de savoir comment créer et supprimer des dépôts. Vous ne pouvez gérer que des dépôts que vous
possédez (qui sont à votre nom) ou d'une organisation dont vous avez les permissions d'écriture.

### Créer un dépôt

Le code ci dessous créé un dépôt vide avec [`create_repo`] et lui donne un nom avec le paramètre `repo_id`. Le `repo_id` est votre namespace suivi
du nom du dépôt: `nom_utilisateur_ou_organisation/nom_depot`

```py
>>> from huggingface_hub import create_repo
>>> create_repo("lysandre/test-model")
'https://huggingface.co/lysandre/test-model'
```

Par défaut, [`create_repo`] créé un dépôt de modèle, mais vous pouvez utiliser le paramètre `repo_type` pour spécifier un autre type de dépôt. Par exemple, si vous voulez créer un dépôt de dataset:

```py
>>> from huggingface_hub import create_repo
>>> create_repo("lysandre/test-dataset", repo_type="dataset")
'https://huggingface.co/datasets/lysandre/test-dataset'
```

Lorsque vous créer un dépôt, vous pouvez définir la visibilité de votre dépôt avec le paramètre `private`.

```py
>>> from huggingface_hub import create_repo
>>> create_repo("lysandre/test-private", private=True)
```

Si vous voulez changer la visibilité du dépôt plus tard, vous pouvez utiliser la fonction [`update_repo_visibility`].

### Supprimer un dépôt

Vous pouvez supprimer un dépôt avec [`delete_repo`]. Assurez vous que vous voulez supprimer ce dépôt avant car c'est un processus irréversible!

Précisez le `repo_id` du dépôt que vous voulez supprimer:

```py
>>> delete_repo(repo_id="lysandre/my-corrupted-dataset", repo_type="dataset")
```

### Dupliquer un dépôt (uniquement pour les spaces)

Dans certains cas, vous avez besoin de copier les dépôts de quelqu'un d'autre pour l'adapter à votre cas d'utilisation.
C'est possible pour les spaces en utilisant la méthode [`duplicate_space`]. Elle dupliquera le dépôt entier.
Vous aurez toujours besoin de configurer vos propres paramètres (hardware, temps de veille, stockage, variables et secrets).
Consultez notre guide [gérez vos spaces](./manage-spaces) pour plus de détails.

```py
>>> from huggingface_hub import duplicate_space
>>> duplicate_space("multimodalart/dreambooth-training", private=False)
RepoUrl('https://huggingface.co/spaces/nateraw/dreambooth-training',...)
```

## Upload et téléchargement de fichiers

Maintenant que vous avez créé votre dépôt, vous aurez besoin de push des changements dessus et de télécharger des fichiers
de votre dépôt.

Ces deux sujets méritent leur propre guides. Consultez les guides [upload](./upload) et [téléchargement](./download)
pour apprendre à utiliser vos dépôts.


## Branches et tags

Les dépôts Git utilisent souvent des branches pour enregistrer différentes version d'un même dépôt.
Les tags peuvent aussi être utilisés pour flager un état spécifique de votre dépôt, par exemple,
lors de la sortie d'une version. PLus généralement, on désigne les branches et les tags par le terme [git references](https://git-scm.com/book/en/v2/Git-Internals-Git-References).

### Créer des branches et de tags

Vous pouvez créer de nouvelles branches et de nouveaux tags en utilisant [`create_branch`] et [`create_tag`]:

```py
>>> from huggingface_hub import create_branch, create_tag

# Créé une branche sur le dépôt d'un space basée sur la branche `main`
>>> create_branch("Matthijs/speecht5-tts-demo", repo_type="space", branch="handle-dog-speaker")

# Créé un tag sur un dépôt de dataset à partir de la branche `v0.1-release`
>>> create_tag("bigcode/the-stack", repo_type="dataset", revision="v0.1-release", tag="v0.1.1", tag_message="Bump release version.")
```

Vous pouvez utiliser les fonctions [`delete_branch`] et [`deelte_tag`] de la même manière pour supprimer une branche ou un tag.

### Lister toutes les branches et les tags

Vous pouvez aussi lister toutes les références git d'un dépôt en utilisant [`list_repo_refs`]:

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

## Changer les paramètres de dépôt

Les dépôts ont certains paramètres que vous pouvez configurer. La plupart du temps, vous aurez envie de faire ceci à la main
dans la page de paramètre du dépôt dans votre navigateur. Vous devez avoir la permission en mode écriture sur un dépôt pour le
configure (soit en étant le propriétaire ou en faisant partie d'une organisation). Dans cette secction, nous verrons les
paramètres que vous pouvez aussi configurer par le code en utilisant `huggingface_hub`.

Certains paramètres sont spécifique aux spaces (hardware, variables d'environnement,...). Pour les configurern, consultez notre guide [Gérez vos spaces](../guides/manage-spaces)

### Changer la visibilité

Un dépôt peut être public ou privé. Un dépôt privé n'est visible que par vous ou les membres de l'organisation dans laquelle le dépôt est situé. Passez un dépôt en mode privé en faisant ceci:

```py
>>> from huggingface_hub import update_repo_visibility
>>> update_repo_visibility(repo_id=repo_id, private=True)
```

### Renommez votre dépôt

Vous pouvez renommer votre dépôt sur le Hub en utilisant [`move_repo`]. En utilisant cette même méthode, vous pouvez aussi faire
passer le dépôt d'un utilisateur à une organisation. Lorsque vous faites ainsi, il y a [quelques limitations](https://hf.co/docs/hub/repositories-settings#renaming-or-transferring-a-repo) qu'il vous faut connaitre. Par exemple, vous ne pouvez pas transférer le dépôt à
un autre utilisateur.

```py
>>> from huggingface_hub import move_repo
>>> move_repo(from_id="Wauplin/cool-model", to_id="huggingface/cool-model")
```

## Gérer une copie locale de votre dépôt

Toutes les actions décrites ci dessus peuvent être faites en utilisant des requêtes HTTP. Cependant, dans certains cas vous
aurez peut-être besoin d'avoir une copie local de votre dépôt et intéragir avec les commandes Git que vous connaissez déjà.

La classe [`Repository`] vous permet d'intéragir avec des fichiers et des dépôts sur le Hub avec des fonctions similaire aux commandes Git. C'est un wrapper autour des méthodes Git et Git-LFS pour utiliser les commandes git que vous conaissez déjà. Avant de commencer, assurez vous que Git-LFS est bien installé sur votre machine (voir [ici](https://git-lfs.github.com/) pour les instructions d'installation).

<Tip warning={true}>

[`Repository`] est aujourd'hui deprecated en faveur des alternatives basées sur de http implémentées dans [`HfApi`]. Au vu de son adoption assez large, la suppression complète de [`Repository`] ne sera faite que dans la version `v1.0`. Pour plus de détails, consultez [cette page](./concepts/git_vs_http).

</Tip>

### Utiliser un dépôt local

Instanciez un objet [`Repository`] avec un chemin vers un dépôt local:

```py
>>> from huggingface_hub import Repository
>>> repo = Repository(local_dir="<path>/<to>/<folder>")
```

### Cloner

Le paramètre `clone_from` clone un dépôt à partir de l'ID du dépôt Hugging Face vers un chemin local spécifié avec l'argument `local_dir`:

```py
>>> from huggingface_hub import Repository
>>> repo = Repository(local_dir="w2v2", clone_from="facebook/wav2vec2-large-960h-lv60")
```

`clone_from` peut aussi cloner un dépôt en utilisant un URL:

```py
>>> repo = Repository(local_dir="huggingface-hub", clone_from="https://huggingface.co/facebook/wav2vec2-large-960h-lv60")
```

Vous pouvez combiner le paramètre `clone_from` avec [`create_repo`] pour créer et cloner un dépôt:

```py
>>> repo_url = create_repo(repo_id="repo_name")
>>> repo = Repository(local_dir="repo_local_path", clone_from=repo_url)
```

Vous pouvez aussi configurer un nom d'utilisateur Git et un email vers un dépôt cloné en précisant les paramètres `git_user` et `git_email` lorsque vous clonez un dépôt. Lorsque les utilisateurs feront des commits sur ce dépôt, Git connaitre l'auteur du commit.

```py
>>> repo = Repository(
...   "my-dataset",
...   clone_from="<user>/<dataset_id>",
...   token=True,
...   repo_type="dataset",
...   git_user="MyName",
...   git_email="me@cool.mail"
... )
```

### Branche

Les branches sont importante pour la collaboration l'expérimentation sans impact sur vos fichiers ou votre code actuel. Changez de branche avec [`~Repository.git_checkout`]. Par exemple, si vous voulez passer de `branche1` à `branche2`:

```py
>>> from huggingface_hub import Repository
>>> repo = Repository(local_dir="huggingface-hub", clone_from="<user>/<dataset_id>", revision='branche1')
>>> repo.git_checkout("branche2")
```

### Pull

[`~Repository.git_pull`] vous permet de mettre à jour une branche local avec des changements d'un dépôt distant:

```py
>>> from huggingface_hub import Repository
>>> repo.git_pull()
```

Utilisez `rebase=True` si vous voulez que vos commits locaux soient opérationnels après que votre branche ai été mise à jour avec les nouveaux commits:

```py
>>> repo.git_pull(rebase=True)
```
