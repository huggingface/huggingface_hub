<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Upload des fichiers vers le Hub

Partager vos fichiers et votre travail est un aspect important du Hub. La librairie `huggingface_hub` offre plusieurs options pour upload vos fichiers vers le Hub. Vous pouvez utiliser ces fonction indépendemment ou les intégrer à votre librairie, pour rendre l'intéraction avec le Hub plus pratique pour vos utilisateurs. Ce guide vous montrera comment push des fichiers:

- Sans utiliser Git.
- Qui sont très volumineux avec [Git LFS](https://git-lfs.github.com/).
- Avec le gestionnaire de contexte des `commit`.
- Avec la fonction [`~Repository.push_to_hub`].

Lorsque vous voulez upload des fichiers sur le HUb, vous devez vous connecter à votre compte Hugging Face:

- Connectez vous à votre compte Hugging Face avec la commande suivante:

  ```bash
  huggingface-cli login
  # Ou en utilisant une variable d\'environnement
  huggingface-cli login --token $HUGGINGFACE_TOKEN
  ```

- Sinon, vous pouvez vous connecter par le code en utilisant [`login`] dans un notebook ou un script:

  ```python
  >>> from huggingface_hub import login
  >>> login()
  ```

  Si lancé dans un notebook Jupyter ou Colaboratory, [`login`] démarera un widget
  depuis lequel vous pouvez rentrer vos token d'authentification Hugging Face. Sinon,
  un message sera affiché dans le terminal.

  Il est aussi possible de se connecter par le code sans widget en passant directement
  votre token à la méthode [`login`]. Si vous faites ainsi, faites attention lors du
  partage de votre notenook. Une bonne pratique est de charger le token d'un trousseau
  sécurisé aulieu de le sauvegarder en clair dans votre notebook.

## Upload un fichier

Une fois que vous avez créé un dépôt avec [`create_repo`], vous puovez upload un fichier sur votre dépôt en utilisant [`upload_file`].

Précisez le chemin du fichier à upload, le nom du dépôt dans lequel vous voulez ajouter le fichier et l'endroit du dépôt dans lequel vous voulez qu'il soit. Selon votre type de dépôt, vous pouvez, facultativement définir le type de dépôt à `dataset`, `model` ou `space`.

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.upload_file(
...     path_or_fileobj="/path/to/local/folder/README.md",
...     path_in_repo="README.md",
...     repo_id="username/test-dataset",
...     repo_type="dataset",
... )
```

## Upload un dossier

Utilisez la fonction [`upload_folder`] pour upload un dossier local vers un dépôt. Précisez le chemin du dossier local,
où vous voulez que le dossier soit upload dans le dépôt et le nom du dépôt dans lequel vous voulez ajouter le dossier. Selon
votre type de dépôt, vous pouvez facultativement définir le type de dépôt à `dataset`, `model` ou `space`.

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()

# Upload tout le contenu du fichier local vers votre space distant
# Par défaut, les fichiers sont upload à la racine du dépôt
>>> api.upload_folder(
...     folder_path="/path/to/local/space",
...     repo_id="username/my-cool-space",
...     repo_type="space",
... )
```

Par défaut, le fichier `.gitignore` sera pris en compte pour savoir quels fichiers doivent être commit ou pas. Par défaut, nous vérifions si un fichier `.gitignore` est présent dans un commit, s'il n'y en a pas, nous vérifions s'il existe sur le Hub. Notez que seul un fichier `.gitignore` présent à la racine du chemin sera utilisé. Nous ne cherchons pas de fichiers `.gitignore` dans les sous-dossiers.

Si vous ne voulez pas utiliser un fichier `.gitignore` codé en dur, vous pouvez utiliser les arguments `allow_patterns` et `ignore_patterns` pour filtrer les fichiers à upload. Ces paramètres prennent en entrée soit un pattern, soit une liste de patterns. Plus d'informations sur ce que sont les patterns [ici](https://tldp.org/LDP/GNU-Linux-Tools-Summary/html/x11655.htm). Si `allow_patterns` et `ignore_patterns` sont donnés, les deux contraintes s'appliquent.

En plus du fichier `.gitignore` et des patterns allow/ignore, n'importe quel dossier `.git/` présent dans n'import quel sous chemin sera ignoré.

```py
>>> api.upload_folder(
...     folder_path="/path/to/local/folder",
...     path_in_repo="my-dataset/train", # Upload vers un dossier spécifique
...     repo_id="username/test-dataset",
...     repo_type="dataset",
...     ignore_patterns="**/logs/*.txt", # Ignore tous les logs en .txt
... )
```

Vous pouvez aussi utiliser l'argument `delete_patterns` pour préciser les fichiers que vous voulez supprimer du dépôt
dans le même commit. Cet argument peut-être utile si voulez nettoyer un fichier distant avant de push vos fichiers dedans
et que vous ne savez pas quels fichiers existent déjà.

L'exemple ci-dessous upload le fichier local `./logs` vers le fichier distant `/experiment/logs/`. Seul les fichiers textuels
sont upload. Avant ça, tous les logs précédents sur le dépôt sont supprimés, le tout dans un seul commit.
```py
>>> api.upload_folder(
...     folder_path="/path/to/local/folder/logs",
...     repo_id="username/trained-model",
...     path_in_repo="experiment/logs/",
...     allow_patterns="*.txt", # Upload tous les fichiers textes locaux
...     delete_patterns="*.txt", # Supprime tous les fichiers textes distants avant d'upload
... )
```

## Upload depuis le CLI

Vous pouvez utiliser la commande `huggingface-cli upload` depuis le terminal pour upload directement des fichiers vers le Hub. En interneelle utilise aussi les helpers [`upload_file`] et [`upload_folder`] décrits ci dessus.

Vous pouvez upload un unique fichier ou un dossier entier:

```bash
# Cas d'usage:  huggingface-cli upload [repo_id] [local_path] [path_in_repo]
>>> huggingface-cli upload Wauplin/my-cool-model ./models/model.safetensors model.safetensors
https://huggingface.co/Wauplin/my-cool-model/blob/main/model.safetensors

>>> huggingface-cli upload Wauplin/my-cool-model ./models .
https://huggingface.co/Wauplin/my-cool-model/tree/main
```

`local_path` et `path_in_repo` sont optionnels et peuvent être déterminés implicitement. Si `local_path` n'est pas défini,
l'outil vérifiera si un dossier local ou un fichier a le même nom que le `repo_id`. Si ce n'est pas le cas, son contenu
sera upload. Sinon, une exception est levée demandant à l'utilisateur de définir exxplicitement `local_path`. Dans tous
les cas, si `path_in_repo` n'est pas défini, les fichiers sont upload à la racine du dépôt.

Pour plus de détails sur la commande uplaod du CLI, consultez le[guide CLI](./cli#huggingface-cli-upload).

## Fonctionnalités avancées

Dans la plupart des cas, vous n'aurez besoin de rien de plus que [`upload_file`] et [`upload_folder`] pour upload
vos fichiers sur le Hub. Cependant, `huggingface_hub` a des fonctionnalités plus avancées pour rendre le processus
plus simple. Regardons les dans la suite de ce guide.


### Uploads non bloquants

Dans certains cas, vous aura envie de push des données sans blocker votre thread principale. C'est particulièrement
utile pour upload des logs des artefacts tout en continuant à entrainer un modèle. Pour ce faire, vous pouvez utiliser
l'argument `run_as_future` dans [`upload_file`] et [`upload_folder`]. La méthode renvera un objet
[`concurrent.futures.Future`](https://docs.python.org/3/library/concurrent.futures.html#future-objects) que vous pouvez utiliser
pour vérifier le statu de l'upload.

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> future = api.upload_folder( # Upload en arrière plan (action non bloquante)
...     repo_id="username/my-model",
...     folder_path="checkpoints-001",
...     run_as_future=True,
... )
>>> future
Future(...)
>>> future.done()
False
>>> future.result() # Attend que l'upload soit finie (action bloquante)
...
```

<Tip>

Le tâche d'arrière plan sont mise dans une queue en utilisat `run_as_future=True`. Ceci signfie que vous êtes sur que
la tâche sera exécutée dans le bon ordre.

</Tip>

Même si les tâches en arrière plan sont la plupart du temps utiles pour upload des données ou créer des commits, vous
pouvez mettre dans la queue la méthode que vous voulez en utilisant [`run_as_future`]. Par exemple, vous pouvez l'utiliser
pour créer un dépôt puis upload des données dessus en arrière plan. L'argument `run_as_future` dans les méthodes d'upload
est juste un alias autour de cette méthode.

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.run_as_future(api.create_repo, "username/my-model", exists_ok=True)
Future(...)
>>> api.upload_file(
...     repo_id="username/my-model",
...     path_in_repo="file.txt",
...     path_or_fileobj=b"file content",
...     run_as_future=True,
... )
Future(...)
```

### Upload un dossier par morceaux

[`upload_folder`] rend l'upload d'un dossier entier sur le Hub facile. Cependant, pour des fichiers volumineux (des
milliers de fichiers ou des centaines de GB), cette tâche peut rester ardue. Si vous avez un dossier avec beaucoup de fichiers,
vous aurez peut-être envie de l'upload en plusieurs commits. Si vous avez une erreur ou des problèmes de connection pendant
l'upload, vous n'aurez surement pas envie de tout recommencer à zéro.

Pour upload un dossier en plusieurs commits, passez simplement l'argument `multi_commits=True`. En arrière plan,
`huggingface_hub` listera tous les fichiers pour upload/supprimer et découper le tout en plusieurs commits. La
"stratégie" (i.e. comment les commits sont séparés) est basée sur le nombre et la taille des fichiers à upload. Une
pull request sera ouverte sur le Hub pour push tous les commits. Une fois la pull request prête, les commits sont
regroupés en un seul commit. Si le processu est interrompu avant sa fin, vous pouvez relancer votre script pour continuer
l'upload. La pull request créé sera  automatiquement détectée et l'upload continuera là où il a été arrêté. Il est
recommandé d'utiliser l'argument `multi_commits_verbose=True` pour avoir une meilleure compréhension de l'upload et
de sont avancement.

L'exemple ci dessous uploadera plusieurs dossiers vers un dataset en plusieurs commits. Une pull request sera créé sur le
Hub, elle sera merge automatiquement une fois que l'upload est finie. Si vous préférez que la pull request reste ouverte
pour pouvoir faire une review manuelle, utiliser `create_pr=True`.

```py
>>> upload_folder(
...     folder_path="local/checkpoints",
...     repo_id="username/my-dataset",
...     repo_type="dataset",
...     multi_commits=True,
...     multi_commits_verbose=True,
... )
```

Si vous voulez un meilleur controle de la stratégie d'upload (i.e. les commits créés), vous pouvez consulter les
méthodes bas niveau [`plan_multi_commits`] et [`create_commits_on_pr`].

<Tip warning={true}>

`multi_commits` est toujours une fonctionnalité expérimentale. Son API et son comportement pourraient changer dans le futur
sans avertissement préalable.

</Tip>

### Uploads planifiées

Le Hub Hugging Face rend facile l'enregistrement et le versionning de données. Cependant, il y a des limitations lorsqu'on met à jour un même fichier des milliers de fois. Par exemple, vous aurez peut-être envie d'enregistrer les logs d'un processus d'entrainement ou le feedback des utilisateur sur un space déployé. Dans ces deux cas, upload la donnée en tant que dataset sur le Hub semble logique, mais il peut-être difficile de le faire correctement. La raison principale est que vous ne voulez pas versionner toutes les mises à jour de vos donnée, car cela rendrait le dépôt git inutilisable. La classe [`CommitScheduler`] offre une solution à ce problème.

L'idée est de faire tourner une tâche en arrière plan qui va push à intervalles réguliers un dossier local vers le Hub.
Supposons que vous avez un space Gradio qui prend en entré du texte et qui génére deux traductions. Dans ce cas, l'utilisateur peut sélectionner sa traduction préférée. Pour chaque traduction, vous voulez enregistrer l'input, output et les préférences de l'uitlisateur pour analyser les résultats.
C'est un cas d'usage parfait pour [`CommitScheduler`]; vous voulez enregistrer des données sur le Hub (potentiellement des millions
de retour utilisateurs) mais vous n'avez pas besoin d'enregistrer en temps réel chaque input de l'utilisateur. Aulieu de ça,
vous pouvez enregistrer les données en local dans un fichier JSON et l'upload toutes les 10 minutes. Par exemple:

```py
>>> import json
>>> import uuid
>>> from pathlib import Path
>>> import gradio as gr
>>> from huggingface_hub import CommitScheduler

# Définit le fichier dans lequel il faut enregistrer les données. On utilise le UUID pour s'assurer de ne pas overwrite des données existantes d'un feedback préalable
>>> feedback_file = Path("user_feedback/") / f"data_{uuid.uuid4()}.json"
>>> feedback_folder = feedback_file.parent

# Planifie des uploads réguliers. Le dépôt distant et le dossier local sont créés s'il n'existent pas déjà
>>> scheduler = CommitScheduler(
...     repo_id="report-translation-feedback",
...     repo_type="dataset",
...     folder_path=feedback_folder,
...     path_in_repo="data",
...     every=10,
... )

# Définit la fonction qui sera appelée lorsque l'utilisateur enverra son feedback
>>> def save_feedback(input_text:str, output_1: str, output_2:str, user_choice: int) -> None:
...     """
...     Append input/outputs and user feedback to a JSON Lines file using a thread lock to avoid concurrent writes from different users.
...     """
...     with scheduler.lock:
...         with feedback_file.open("a") as f:
...             f.write(json.dumps({"input": input_text, "output_1": output_1, "output_2": output_2, "user_choice": user_choice}))
...             f.write("\n")

# Lancement de Gradio
>>> with gr.Blocks() as demo:
>>>     ... # Définition de la démo Gradio, ne pas oublier d'utiliser `save_feedback`
>>> demo.launch()
```

Et c'est tout! Lesinputs/outputs et feedback des utilisateur seront disponible en tant que dataset sur le Hub. En utilisant un unique nom de fichier JSON, vous êtes sur que vous n'overwriterez pas de données qui se pushent en même sur le même dépôt.

Pour plus de détails sur le [`CommitScheduler`], voici ce que vous devez savoir:
- **append-only:**
    Il est supposé que vous ne faites qu'ajouter du contenu au dossier. Vous devez uniquement ajouter des données à 
    des fichiers existants ou créer de nouveaux fichier. Supprimer ou overwrite des fichiers pourrait corrompre votre
    dépôt.
- **historique git:**
    Le planificateur commitera le dossier toutes les `every` minutes. Pour éviter de polluer le dépôt git, il est reccomadé
    de mettre une valeur minimal d'aumoins 5 minutes. Par ailleurs, les planificateur est créé pour éviter les commits
    vides. Si aucun nouveau contenu n'est détecté dans le dossier, le commit planifié sera abandonné.
- **erreurs:**
    Le planificateur tourne en tant que thread en arrière plan. Il commance quand vous instantiez la classe et ne s'arrête
    jamais. En particulier, si une erreur arrive pendant l'upload (par exemple un problème de connexion), le planificateur
    ignorera cette erreur et réessaiera au prochain commit planifié
- **sécurité des threads:**
    Dans la plupart des cas, il est plus sécuriser de supposer que vous pouvez écrire dans un fichier sans se soucier des
    fichiers bloqués. Le planificateur de crashera pas et ne sera pas corrumpu si vous écrivez du contenue sur le dossier
    pendant l'upload. En pratique, il est possible que de telles problèmes arrivent pour des applications lourdes. Dans
    ce cas, nous conseillons d'utiliser le lock `scheduler.lock` pour s'assurer que le thread soient sécurisés. Le lock
    est bloquée uniquement lorsque le planificateur scan le dossier à la recherche de changements, pas lors de l'upload
    de données. Vous pouvez sans problème supposer que ça n'affectera pas l'expérience utilisateur sur votre space.

#### Space persistence demo

Faire persister des données d'un space vers un dataset sur le Hub est le cas d'usage le plus courant pour [`CommitScheduler`].
Selon les cas d'usages, vous aurez peut-être envie de structurer vos données différemment. La structure doit être assez robuste
pour gérer simultanément la connexion d'un utilisateur et le redémarrage ce qui implique souvent la génération d'UUIDs.
En plus de la robustesse, vous devez upload des données dans un format lisible pour les librairies de datasets 🤗, afin
de pouvoir les réuitiliser plus tard. Nous avons créé un [space](https://huggingface.co/spaces/Wauplin/space_to_dataset_saver)
qui montre comment enregistrer plusieurs formats de données ddifférents (vous aurez peut-être besoin de l'adapter à vos
propres besoins).

#### Uploads personnalisées

[`CommitScheduler`] suppose que votre donnée est append-only. Cependant, vous aurez peut-être besoin de
customiser la manière dont la donnée est uploadée. Vous pouvez faire ça en créant une classe qui hérite
de [`CommitScheduler`] et qui overvrite la méthode `push_to_hub`. Vous êtes sur que cette méthode
sera appelée toutes les `every` minutes dans un thread en arrière plan. Vous n'avez pas besoin de vous
occuper des erreurs et des utilisations simultanées, mais vous devez faire attention à d'autres aspects,
tels que les commits vides ou les données dupliquées.

Dans l'exemple simplifié ci dessous, nous faisons un overwrite de `push_to_hub` pour ziper tous les fichiers PNG
dans une unique archive pour éviter de surcharger le dépôt sur le Hub:

```py
class ZipScheduler(CommitScheduler):
    def push_to_hub(self):
        # 1. Liste les fichiers PNG
          png_files = list(self.folder_path.glob("*.png"))
          if len(png_files) == 0:
              return None  # return directement si rien à commit

        # 2. Zip les fichiers PNG dans une unique archive
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "train.zip"
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zip:
                for png_file in png_files:
                    zip.write(filename=png_file, arcname=png_file.name)

            # 3. Upload l'archive
            self.api.upload_file(..., path_or_fileobj=archive_path)

        # 4. Supprime les fichiers PNG locaux pour éviter de les ré-upload plus tard
        for png_file in png_files:
            png_file.unlink()
```

Lorsque vous modifier `push_to_hub` en faisant un overwrite, vous avez accès aux attributs de [`CommitScheduler`], plus précisément:
- Le client [`HfApi`]: `api`
- Les paramètres du dossier: `folder_path` et `path_in_repo`
- Les paramètres du dépôt: `repo_id`, `repo_type` et `revision`
- Le lock du thread `lock`

<Tip>

Pour plus d'exemples de planififcateurs personnalisés, consultez notre
[space de démo](https://huggingface.co/spaces/Wauplin/space_to_dataset_saver) contenant différentes implementations
dépendant de votre cas d'usage.

</Tip>

### create_commit

Les fonctions [`upload_file`] et [`upload_folder`] sonr des APIs de haut niveau qui sont généralement assez pratiques à
utiliser. Il est recommandé d'essayer ces fonction en premier si vous ne voulez pas travailler à un plus bas niveau.
Cependant, si vous voulez travailler au niveau du commit, vous pouvez utiliser directement la fonction [`create_commit`].

Il y a trois types d'opérations supportées par [`create_commit`]:

- [`CommitOperationAdd`] upload un fichier vers le Hub. Si le fichier existe déjà, le contenu du fichier seront overwrite. Cette opération accepte deux arguments:
  - `path_in_repo`: le chemin vers le dépôt sur lequel vous voulez upload un fichier.
  - `path_or_fileobj`: soit le chemin vers un fichier sur votre machine ou un fichier lui même. C'est le contenu du fichier à upload vers le Hub.

- [`CommitOperationDelete`] supprime un fichier ou un dossier d'un dépôt. Cette opération accepte `path_in_repo` en argument.

- [`CommitOperationCopy`] copie un fichier d'un dépôt. Cette opération prend en entré trois arguments:

  - `src_path_in_repo`: le chemin vers le dépôt du fichier à copier.
  - `path_in_repo`: le chemin vers le dépôt sur lequel le fichier doit être copié.
  - `src_revision`: argument optionnel - la révision du fichier à copier si vous voulez copier un fichiers d'une branche/version différente de main.

Par exmeple, si vous voulez upload deux fichiers et supprimer dossier:

1. Utilisez la bonne `CommitOperation` pour ajouter ou supprimer un fichier en supprimer un dossier:

```py
>>> from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
>>> api = HfApi()
>>> operations = [
...     CommitOperationAdd(path_in_repo="LICENSE.md", path_or_fileobj="~/repo/LICENSE.md"),
...     CommitOperationAdd(path_in_repo="weights.h5", path_or_fileobj="~/repo/weights-final.h5"),
...     CommitOperationDelete(path_in_repo="old-weights.h5"),
...     CommitOperationDelete(path_in_repo="logs/"),
...     CommitOperationCopy(src_path_in_repo="image.png", path_in_repo="duplicate_image.png"),
... ]
```

2. Passez vos opérations à [`create_commit`]:

```py
>>> api.create_commit(
...     repo_id="lysandre/test-model",
...     operations=operations,
...     commit_message="Upload my model weights and license",
... )
```

En plus d'[`upload_file`] et [`upload_folder`], les fonctions suivante utilisent aussi [`create_commit`] en arrière plan:

- [`delete_file`] supprime un fichier d'un dépôt sur le Hub.
- [`delete_folder`] supprime un dossier d'un dépôt sur le Hub.
- [`metadata_update`] Met à jour les métadonnées d'un dépôt.

Pour plus d'informations, consultez la référence [`HfApi`].

### Preupload des fichier LFS avant le commit

Dans certains cas, vous aurez peut-être envie d'upload d'immense fichiers vers S3 **avant** de faire le commit. Par
exemple, si vous commitez un dataset dans plusieurs shards qui sont générées in-memory, vous aurez besoin d'upload
les shards une par une pour éviter de manquer de mémoire. Une solution est d'upload chaque shard comme commit séparé
sur le dépôt. Tout en étant parfaitement valide, cette solution a le désavantage de brouiller l'historique git en
générant de dizaines de commits. Pour éviter ce problème, vous pouvez upload vos fichier un par un vers S3 puis créer
un seul commit à la fin. C'est possible en utilisatn [`preupload_lfs_files`] en combinaison avec [`create_commit`].

<Tip warning={true}>

Cette méthode est complexe. Utiliser directement [`upload_file`], [`upload_folder`] ou [`create_commit`] aulieu de
gérer la logique bas niveau des fichiers qui s'uploadent en avance est la meilleur manière ed faire dans la plupart
des cas. Le problème principal de [`preupload_lfs_files`] est que tant que le commit est fait, les fichiers ne sont pas
accessibles sur le dépôt du Hub. Si vous avez une question, n'hésitez pas à nous ping sur Discord ou en créant
une issue GitHub. 

</Tip>

Voici un exemple simple illustrant comme pre-upload des fichiers:

```py
>>> from huggingface_hub import CommitOperationAdd, preupload_lfs_files, create_commit, create_repo

>>> repo_id = create_repo("test_preupload").repo_id

>>> operations = [] # Liste de toutes les objets `CommitOperationsAdd` qui seront générés
>>> for i in range(5):
...     content = ... # génère un contenu binaire
...     addition = CommitOperationAdd(path_in_repo=f"shard_{i}_of_5.bin", path_or_fileobj=content)
...     preupload_lfs_files(repo_id, additions=[addition])
...     operations.append(addition)

>>> # Créé un commit
>>> create_commit(repo_id, operations=operations, commit_message="Commit all shards")
```

Premièrement, nous créons les objets [`CommitOperationAdd`] un par un. Dans un vrai exemple, ils contiendraient
les shard générées. Chaque fichier est uploadé avant de générer le suivant. Pendant l'étape [`preupload_lfs_files`],
**L'objet `CommitoperationAdd` est muté**. Vous devez uniquement l'utiliser pour le passer directement à [`create_commit`].
Le changement principal sur l'objet est **la suppression du contenu binaire**, ce qui signifie que le ramasse miette
le supprimera si vous ne stockez pas une autre référence à ce contenu. C'est un mécanisime prévu car nous ne voulons pas
garder en mémoire le contenu qui est déjà upload. Enfin, nous créons un commit en passant toutes les opérations à
[`create_commit`]. Vous pouvez passer des opérations supplémentaires (add, delete ou copy) qui n'ont pas encore été
gérées et elles le seront correctement.

## Quelques astuces pour les uploads volumineux

Il y a des limitations à connaitre lors de l'utilisation de grandes quantités de données sur votre dépôt. Étant donné le délai pour  transférer
la donnée, faire un upload pour avoir une erreur à la fin du processus, que ce soit sur hf.co ou en local, peut être très frustrant.

Consultez notre guide sur les [limitations des dépôts et recommendations](https://huggingface.co/docs/hub/repositories-recommendations) afin de connaitre les bonnes pratiques sur la structuration de dépôts sur le Hub. Maintenant, continuons avec des conseils pratiques pour faire en sorte que vos upload fonctionnent de la manière la plus fluide possible.

- **Commencez petit**: Nous vous recommandons de commencer avec une petite quantité de données pour tester votre script
d'upload. Ce sera plus facile d'itérer sur une script lorsque les erreur ne prennent que très peu de temps à arriver.
- **Attendez vous à avoir des erreurs**: Déplacer de grandes quantités de donées est un vrai challenge. Vous ne savez
pas ce qui peut se passer, mais il est troujours mieux de considérer que quelque chose va malfonctionner aumoins une fois,
que ce soit votre machine, votre connexion, ou nos serveurs. Par exemple, si vous comptez upload un grand nombre de fichiers,
il vaut mieux garder une trace locale des fichiers que vous avez déjà upload avant d'upload le dossier suivant. Normalement
un fichier LFS déjà commit ne sera jamais re-uploadé deux fois mais le vérifier côté client peut quand même vous faire
gagner du temps.
- **Utilisez `hf_transfer`**: c'est une [librarie basée sur Rust](https://github.com/huggingface/hf_transfer) qui a pour
but d'accélérer les upload sur les machines avec une grande bande passante. Pour l'utiliser, vous devez l'installer
(`pip install hf_transfer`) et l'activer en définissant la variable d'environnement `HF_HUB_ENABLE_HF_TRANSFER=1`. Vous
pouvez enfuiste utiliser `huggingface_hub` normalement. Disclaimer: c'est un outil complexe. Il est testé et prêt à la
mise en production mais n'a pas toutes les fonctionnalités user-friendly telles que la gestion d'erreurs avancée ou les
proxys. Pour plus de détails, consultez cette [section](https://huggingface.co/docs/huggingface_hub/hf_transfer).

<Tip>

Les barres de progression sont supportées par `hf_transfer` à partir de la version `0.1.4`. Mettez à jour (`pip install -U hf_transfer`) si vous comptez activer les uploads rapides.

</Tip>

## (approche historique) Uploadez des fichiers avec Git LFS

Toutes les méthodes décrites ci-dessus utilisent l'API du Hub pour upload des fichiers. C'est la méthode recommandée pour
upload des fichiers sur le Hub. Toutesfois, nous fournissons aussi [`Repository`], un wrapper autour de git pour gérer
un dépôt local.

<Tip warning={true}>

Bien que [`Repository`] ne soit pas réellement deprecated, nous recommandons l'utilisation des méthodes basées sur
l'HTTP décrite ci dessus. Pour plus de détails sur cette recommandation, consultez [ce guide](../concepts/git_vs_http)
qui explique les différences fondamentales entre les deux approches.

</Tip>

Git LFS gère automatiquement des fichiers d'une taille supérieure à 10MB. Mais pour les fichiers très larges (>5GB), vous devez installer un agent
de transfert personnalisé pour Git LFS:

```bash
huggingface-cli lfs-enable-largefiles
```

Vous devez faire cette installation pour chaque dépôt qui a un fichier de taille supérieure à 5GB. Une fois installé, vous pourrez push
des fichiers volumineux.

### Gestionnaire de contexte de commit

Le gestionnaire de contexte de `commit` gère quatre des commandes les plus utilisées sur Git: pull, add, commit et push. `git-lfs` traque automatiquement n'importe quel fichier d'une taille supérieure à 10MB. Dans les exemples suivant, le Le gestionnaire de contexte de `commit`:

1. Pull depuis le dépôt `text-files`.
2. Ajoute un changment fait à `file.txt`
3. Commit le changement.
4. Push le changement vers le dépôt `text-files`.

```python
>>> from huggingface_hub import Repository
>>> with Repository(local_dir="text-files", clone_from="<user>/text-files").commit(commit_message="My first file :)"):
...     with open("file.txt", "w+") as f:
...         f.write(json.dumps({"hey": 8}))
```

Voici un autre exemple expliquant comment utiliser le gestionnaire de contexte de `commit` pour sauvegarder et
upload un fichier vers un dépôt:

```python
>>> import torch
>>> model = torch.nn.Transformer()
>>> with Repository("torch-model", clone_from="<user>/torch-model", token=True).commit(commit_message="My cool model :)"):
...     torch.save(model.state_dict(), "model.pt")
```

Définissez `blocking=False` si vous voulez push vous commits de manière asynchrone. Les comportements non bloquants sont utiles quand vous voulez continuer à faire tourner un script lorsque vous pushez vos commits.

```python
>>> with repo.commit(commit_message="My cool model :)", blocking=False)
```

Vous pouvez vérifier le statut de votre push avec la méthode `command_queue`:

```python
>>> last_command = repo.command_queue[-1]
>>> last_command.status
```

Référez vous à la table ci dessous pour la liste de statuts possibles:

| Statut   | Description                          |
| -------- | ------------------------------------ |
| -1       |  Le push est en cours                |
| 0        |  Le push s'est fini sans erreurs.    |
| Non-zero |  Il y a eu une erreur.               |

Lorsque vous utilisez `blocking=False`, les commandes sont suivies, et votre script se finira uniquement lorsque toues les push sont finis, même si d'autres erreurs arrivent dans votre script. Voici des commandes utiles pour vérifier le statut d'un push:

```python
# Inspecte une erreur
>>> last_command.stderr

# Vérifie si un push est fini ou en cours
>>> last_command.is_done

# Vérifie si une commande push a donné une erreur
>>> last_command.failed
```

### push_to_hub

la classe [`Repository`] a une fonction [`~Repository.push_to_hub`] pour ajouter des fichiers, faire un commit et les push vers un dépôt. A la différence du gestionnaire de contexte de `commit`, vous aurez besoin de pull depuis un dépôt d'abord avant d'appeler [`~Repository.push_to_hub`].

Par exemple, si vous avez déjà cloné un dépôt du Hub, vous pouvez initialiser le `repo` depuis le chemin local:

```python
>>> from huggingface_hub import Repository
>>> repo = Repository(local_dir="path/to/local/repo")
```

Mettez à jour votre clone local avec [`~Repository.git_pull`] et pushez ensuite votre fichier vers le Hub:

```py
>>> repo.git_pull()
>>> repo.push_to_hub(commit_message="Commit my-awesome-file to the Hub")
```

Cependant si vous n'êtes pas prêt à push un fichier, vous pouvez utiliser [`~Repository.git_add`] et [`~Repository.git_commit`] pour simplement add et commit votre fichier:

```py
>>> repo.git_add("path/to/file")
>>> repo.git_commit(commit_message="add my first model config file :)")
```

Une fois que vous êtes prêt, push le fichier vers votre dépôt avec [`~Repository.git_push`]:

```py
>>> repo.git_push()
```
