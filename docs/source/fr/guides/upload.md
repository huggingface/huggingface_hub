<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Uploader des fichiers sur le Hub

La bibliothèque `huggingface_hub` offre plusieurs options pour uploader vos fichiers sur le Hub. Vous pouvez utiliser ces fonctions indépendamment ou les intégrer dans votre propre bibliothèque.

Chaque fois que vous souhaitez uploader des fichiers sur le Hub, vous devez vous connecter à votre compte Hugging Face. Pour plus de détails sur l'authentification, consultez [cette section](../quick-start#authentication).

## Uploader un fichier

Une fois que vous avez créé un dépôt avec [`create_repo`], vous pouvez uploader un fichier vers votre dépôt en utilisant [`upload_file`].

Spécifiez le chemin du fichier à uploader, où vous souhaitez uploader le fichier dans le dépôt, et le nom du dépôt auquel vous souhaitez ajouter le fichier. Vous pouvez optionnellement définir le type de dépôt comme `dataset`, `model` ou `space` en fonction de votre besoin.

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.upload_file(
...     path_or_fileobj="/path/to/local/folder/README.md",
...     path_in_repo="README.md",
...     repo_id="username/test-dataset",
...     repo_type="dataset", # Uploader vers un dépôt de dataset
... )
```

## Uploader un dossier

Utilisez la fonction [`upload_folder`] pour uploader un dossier local vers un dépôt existant. Spécifiez le chemin du dossier local
à uploader, où vous souhaitez uploader le dossier dans le dépôt, et le nom du dépôt auquel vous souhaitez ajouter le
dossier. Selon votre type de dépôt, vous devez définir `dataset`, `model` ou `space`. (model par défaut)

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()

# Uploader tout le contenu du dossier local vers votre Space distant.
# Par défaut, les fichiers sont uploadés à la racine du dépôt
>>> api.upload_folder(
...     folder_path="/path/to/local/space",
...     repo_id="username/my-cool-space",
...     repo_type="space",
... )
```

Par défaut, le fichier `.gitignore` sera pris en compte pour savoir quels fichiers doivent être commités ou non : Nous vérifions si un fichier `.gitignore` est présent dans un commit, si il n'y en a pas, nous vérifions s'il existe sur le Hub. Veuillez noter que seul un fichier `.gitignore` présent à la racine du répertoire sera utilisé. Nous ne vérifions pas les fichiers `.gitignore` dans les sous-répertoires.

Si vous ne souhaitez pas utiliser un fichier `.gitignore` codé en dur, vous pouvez utiliser les arguments `allow_patterns` et `ignore_patterns` pour filtrer les fichiers à uploader. Ces paramètres acceptent soit un seul motif, soit une liste de motifs. Les motifs sont des wildcards standard (motifs de globbing) comme documenté [ici](https://tldp.org/LDP/GNU-Linux-Tools-Summary/html/x11655.htm). Si `allow_patterns` et `ignore_patterns` sont tous deux fournis, les deux contraintes s'appliquent.

En plus du fichier `.gitignore` et des motifs allow/ignore, tout dossier `.git/` présent dans n'importe quel sous-répertoire sera ignoré.

```py
>>> api.upload_folder(
...     folder_path="/path/to/local/folder",
...     path_in_repo="my-dataset/train", # Uploader vers un dossier spécifique
...     repo_id="username/test-dataset",
...     repo_type="dataset",
...     ignore_patterns="**/logs/*.txt", # Ignorer tous les logs texte (fichiers .txt dans le dossier logs)
... )
```

Vous pouvez également utiliser l'argument `delete_patterns` pour spécifier les fichiers que vous souhaitez supprimer du dépôt dans le même commit.
Cela peut s'avérer utile si vous souhaitez nettoyer un dossier distant avant de pousser des fichiers dedans et que vous ne savez pas quels fichiers
existent déjà.

L'exemple ci-dessous uploade le dossier local `./logs` vers le dossier distant `/experiment/logs/`. Seuls les fichiers txt sont uploadés
mais avant, il y aura une purge de tous les logs précédents sur le dépôt.
```py
>>> api.upload_folder(
...     folder_path="/path/to/local/folder/logs",
...     repo_id="username/trained-model",
...     path_in_repo="experiment/logs/",
...     allow_patterns="*.txt", # Uploader tous les fichiers texte locaux
...     delete_patterns="*.txt", # Supprimer tous les fichiers texte distants avant
... )
```

## Uploader depuis le CLI

Vous pouvez utiliser la commande `hf upload` depuis le terminal pour uploader directement des fichiers sur le Hub. En interne, elle utilise les mêmes helpers [`upload_file`] et [`upload_folder`] décrits ci-dessus.

Vous pouvez uploader soit un seul fichier, soit un dossier entier :

```bash
# Usage:  hf upload [repo_id] [local_path] [path_in_repo]
>>> hf upload Wauplin/my-cool-model ./models/model.safetensors model.safetensors
https://huggingface.co/Wauplin/my-cool-model/blob/main/model.safetensors

>>> hf upload Wauplin/my-cool-model ./models .
https://huggingface.co/Wauplin/my-cool-model/tree/main
```

`local_path` et `path_in_repo` sont optionnels et peuvent être implicitement déduits. Si `local_path` n'est pas défini, l'outil vérifiera
si un dossier ou fichier local a le même nom que le `repo_id`. Si c'est le cas, son contenu sera uploadé.
Sinon, une exception est levée demandant à l'utilisateur de définir explicitement `local_path`. Dans tous les cas, si `path_in_repo` n'est pas
défini, les fichiers seront uploadés à la racine du dépôt.

Pour plus de détails sur la commande upload du CLI, veuillez consulter le [guide CLI](./cli#hf-upload).

## Uploader un grand dossier

Dans la plupart des cas, la méthode [`upload_folder`] et la commande `hf upload` devraient être les solutions de référence pour uploader des fichiers sur le Hub. Elles garantissent qu'un seul commit sera effectué. Elles gèrent de nombreux cas d'usage et échouent explicitement lorsque quelque chose ne va pas. Cependant, lorsqu'il s'agit d'une grande quantité de données, il faut utiliser la méthode [`upload_large_folder`] :
- le processus d'upload est divisé en plusieurs petites tâches (hachage de fichiers, pré-upload de ceux-ci et commit). Chaque fois qu'une tâche est terminée, le résultat est mis en cache localement dans un dossier `./cache/huggingface` à l'intérieur du dossier que vous essayez d'uploader. En faisant cela, il y a la possibilité de redémarrer le processus après une interruption et de reprendre toutes les tâches.
- le hachage de gros fichiers et leur pré-upload bénéficient  du multi-threading si votre machine le permet.
- Un mécanisme de nouvelle tentative a été ajouté pour réessayer chaque tâche indépendante indéfiniment jusqu'à ce qu'elle réussisse (peu importe s'il s'agit d'une OSError, ConnectionError, PermissionError, etc.). Ce mécanisme est à double tranchant. Si des erreurs transitoires se produisent, le processus continuera et réessayera. Si des erreurs permanentes se produisent (par exemple permission refusée), il réessayera indéfiniment sans résoudre la cause première. (Retry)

Si vous souhaitez plus de détails techniques sur la façon dont `upload_large_folder` est implémentée, veuillez consulter la référence du package [`upload_large_folder`].

Voici comment utiliser [`upload_large_folder`] dans un script. La signature de la méthode est très similaire à [`upload_folder`] :

```py
>>> api.upload_large_folder(
...     repo_id="HuggingFaceM4/Docmatix",
...     repo_type="dataset",
...     folder_path="/path/to/local/docmatix",
... )
```

Vous verrez la sortie suivante dans votre terminal :
```
Repo created: https://huggingface.co/datasets/HuggingFaceM4/Docmatix
Found 5 candidate files to upload
Recovering from metadata files: 100%|█████████████████████████████████████| 5/5 [00:00<00:00, 542.66it/s]

---------- 2024-07-22 17:23:17 (0:00:00) ----------
Files:   hashed 5/5 (5.0G/5.0G) | pre-uploaded: 0/5 (0.0/5.0G) | committed: 0/5 (0.0/5.0G) | ignored: 0
Workers: hashing: 0 | get upload mode: 0 | pre-uploading: 5 | committing: 0 | waiting: 11
---------------------------------------------------
```

D'abord, le dépôt est créé s'il n'existait pas auparavant. Ensuite, le dossier local est scanné pour les fichiers à uploader. Pour chaque fichier, nous essayons de récupérer les métadonnées (depuis un upload précédemment interrompu). À partir de là, il est capable de lancer des workers et d'afficher un statut de mise à jour toutes les 1 minute. Ici, nous pouvons voir que 5 fichiers ont déjà été hachés mais pas pré-uploadés. 5 workers sont en train de pré-uploader des fichiers tandis que les 11 autres attendent une tâche.

Une ligne de commande est également disponible. Vous pouvez définir le nombre de workers dans la commande en utilisant l'argument `--num-workers` :

```sh
hf upload-large-folder HuggingFaceM4/Docmatix --repo-type=dataset /path/to/local/docmatix --num-workers=16
```

> [!TIP]
> Pour les grands uploads, vous devez définir `repo_type="model"` ou `--repo-type=model` explicitement. Ceci permet d'éviter d'avoir des données uploadées vers un dépôt avec un mauvais type. Si c'est le cas, vous devrez malheureusement tout re-uploader.

> [!WARNING]
> Bien qu'étant beaucoup plus robuste pour uploader de grands dossiers, `upload_large_folder` est plus limitée que [`upload_folder`] au niveau des fonctionnalités. En pratique :
> - vous ne pouvez pas définir un `path_in_repo` personnalisé. Si vous voulez uploader vers un sous-dossier, vous devez définir la structure appropriée localement.
> - vous ne pouvez pas définir un `commit_message` et `commit_description` personnalisés car plusieurs commits sont créés.
> - vous ne pouvez pas supprimer du dépôt lors de l'upload. (`delete_patterns` n'est pas supporté)
> - vous ne pouvez pas créer une PR directement. Veuillez d'abord créer une PR (depuis l'interface ou en utilisant [`create_pull_request`]) puis commiter dessus en passant `revision`.

### Conseils et astuces pour les grands uploads

Il existe certaines limitations à connaître lorsque vous traitez une grande quantité de données dans votre dépôt.

Consultez notre guide [Limitations et recommandations des dépôts](https://huggingface.co/docs/hub/repositories-recommendations) pour appliquer les meilleures pratiques sur la façon de structurer vos dépôts sur le Hub. Passons maintenant à quelques conseils pratiques pour rendre votre processus d'upload aussi fluide que possible.

- **Commencez petit** : Nous recommandons de commencer avec une petite quantité de données pour tester votre script d'upload. Il est plus facile d'itérer sur un script lorsque l'échec ne prend que peu de temps.
- **Attendez-vous à des échecs** : Streamer de grandes quantités de données est difficile. Vous ne savez pas ce qui peut arriver, mais il est toujours préférable de considérer que quelque chose échouera au moins une fois - peu importe si c'est dû à votre machine, votre connexion ou nos serveurs. Par exemple, si vous prévoyez d'uploader un grand nombre de fichiers, il est préférable de garder une trace localement des fichiers que vous avez déjà uploadés avant d'uploader le prochain lot. Vous êtes assuré qu'un fichier LFS qui est déjà commité ne sera jamais re-uploadé deux fois, mais le vérifier côté client peut quand même économiser du temps. C'est ce que [`upload_large_folder`] est disponible.
- **Utilisez `hf_xet`** : cela exploite le nouveau backend de stockage pour le Hub, est écrit en Rust et est maintenant disponible pour tout le monde. En réalité, `hf_xet` est déjà activé par défaut lors de l'utilisation de `huggingface_hub` ! Pour des performances maximales, définissez [`HF_XET_HIGH_PERFORMANCE=1`](../package_reference/environment_variables.md#hf_xet_high_performance) comme variable d'environnement. Sachez que lorsque le mode haute performance est activé, l'outil essaiera d'utiliser toute la bande passante et tous les cœurs CPU disponibles.

## Fonctionnalités avancées

Dans la plupart des cas, vous n'aurez pas besoin de plus que [`upload_file`] et [`upload_folder`] pour uploader vos fichiers sur le Hub.
Cependant, `huggingface_hub` possède des fonctionnalités plus avancées pour faciliter l'upload. Jetons-y un coup d'œil !

### Uploads plus rapides

Profitez d'uploads plus rapides grâce à `hf_xet`, la liaison Python vers la bibliothèque [`xet-core`](https://github.com/huggingface/xet-core) qui permet la déduplication basée sur les chunks pour des uploads et téléchargements plus rapides. `hf_xet` s'intègre parfaitement avec `huggingface_hub`, mais utilise la bibliothèque Rust `xet-core` et le stockage Xet au lieu de LFS.

`hf_xet` utilise le système de stockage Xet, qui décompose les fichiers en chunks immuables, stockant des collections de ces chunks (appelés blocks ou xorbs) à distance et les récupérant pour ré-assembler le fichier lorsque demandé. Lors de l'upload, après avoir confirmé que l'utilisateur est autorisé à écrire dans ce dépôt, `hf_xet` scannera les fichiers, les décomposant en leurs chunks et collectant ces chunks dans des xorbs (et dédupliquant les chunks connus), puis uploadera ces xorbs vers le service d'adressage de contenu Xet (CAS), qui vérifiera l'intégrité des xorbs, enregistrera les métadonnées des xorbs ainsi que le hash SHA256 LFS (pour supporter la recherche/téléchargement), et écrira les xorbs dans le stockage distant.

Pour l'activer, installez simplement la dernière version de `huggingface_hub` :

```bash
pip install -U "huggingface_hub"
```

À partir de `huggingface_hub` 0.32.0, `hf_xet` est activé par défaut.

Toutes les autres APIs `huggingface_hub` continueront à fonctionner sans aucune modification. Pour en savoir plus sur les avantages du stockage Xet et `hf_xet`, consultez cette [section](https://huggingface.co/docs/hub/xet/index).

**Considérations pour l'upload depuis un Cluster / Système de fichiers distribué**

Lors de l'upload depuis un cluster, les fichiers uploadés résident souvent sur un système de fichiers distribué ou en réseau (NFS, EBS, Lustre, Fsx, etc.). Le stockage Xet va découper ces fichiers en chunks et les écrire dans des blocs (également appelés xorbs) localement, et une fois le bloc terminé, les uploadera. Pour de meilleures performances lors de l'upload depuis un système de fichiers distribué, assurez-vous de définir [`HF_XET_CACHE`](../package_reference/environment_variables#hfxetcache) vers un répertoire qui est sur un disque local (ex. un disque NVMe ou SSD local). L'emplacement par défaut du cache Xet est sous `HF_HOME` à (`~/.cache/huggingface/xet`) et celui-ci se trouvant dans le répertoire personnel de l'utilisateur est souvent également situé sur le système de fichiers distribué.

### Uploads non-bloquants

Dans certains cas, vous souhaitez pousser des données sans bloquer votre thread principal. Ceci est particulièrement utile pour uploader des logs et
des artefacts tout en continuant un entraînement par exemple. Pour ce faire, vous pouvez utiliser l'argument `run_as_future` dans [`upload_file`] et
[`upload_folder`]. Cela retournera un objet [`concurrent.futures.Future`](https://docs.python.org/3/library/concurrent.futures.html#future-objects)
que vous pouvez utiliser pour vérifier le statut de l'upload.

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> future = api.upload_folder( # Uploader en arrière-plan (action non-bloquante)
...     repo_id="username/my-model",
...     folder_path="checkpoints-001",
...     run_as_future=True,
... )
>>> future
Future(...)
>>> future.done()
False
>>> future.result() # Attendre que l'upload soit terminé (action bloquante)
...
```

> [!TIP]
> Les tâches en arrière-plan sont mises en file d'attente lors de l'utilisation de `run_as_future=True`. Cela signifie que vous êtes assuré que les tâches seront
> exécutées dans le bon ordre.

Même si les tâches en arrière-plan sont principalement utiles pour uploader des données/créer des commits, vous pouvez mettre en file d'attente n'importe quelle méthode en utilisant
[`run_as_future`]. Par exemple, vous pouvez l'utiliser pour créer un dépôt puis uploader des données dessus en arrière-plan. L'
argument intégré `run_as_future` dans les méthodes d'upload est juste un alias autour de lui.

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

### Uploader un dossier par chunks

[`upload_folder`] facilite l'upload d'un dossier entier sur le Hub. Cependant, pour les grands dossiers (milliers de fichiers ou
centaines de Go), nous recommandons d'utiliser [`upload_large_folder`], qui divise l'upload en plusieurs commits. Consultez la section [Uploader un grand dossier](#uploader-un-grand-dossier) pour plus de détails.

### Uploads programmés

Le Hugging Face Hub facilite la sauvegarde et la version des données. Cependant, il existe certaines limitations lors de la mise à jour du même fichier des milliers de fois. Par exemple, vous pourriez vouloir sauvegarder les logs d'un processus d'entraînement ou les retours d'utilisateurs sur un Space déployé. Dans ces cas, uploader les données comme un dataset sur le Hub a du sens, mais cela peut être difficile à faire correctement. La raison principale est que vous ne voulez pas versionner chaque mise à jour de vos données car cela rendrait le dépôt git inutilisable. La classe [`CommitScheduler`] offre une solution à ce problème.

L'idée est d'exécuter une tâche en arrière-plan qui pousse régulièrement un dossier local vers le Hub. Supposons que vous ayez un
Space Gradio qui prend en entrée du texte et génère deux traductions de celui-ci. Ensuite, l'utilisateur peut sélectionner sa traduction préférée. Pour chaque exécution, vous voulez sauvegarder l'entrée, la sortie et la préférence de l'utilisateur pour analyser les résultats. C'est un
cas d'usage parfait pour [`CommitScheduler`] ; vous voulez sauvegarder des données sur le Hub (potentiellement des millions de retours d'utilisateurs), mais
vous n'avez pas _besoin_ de sauvegarder en temps réel chaque entrée d'utilisateur. Au lieu de cela, vous pouvez sauvegarder les données localement dans un fichier JSON et
les uploader toutes les 10 minutes. Par exemple :

```py
>>> import json
>>> import uuid
>>> from pathlib import Path
>>> import gradio as gr
>>> from huggingface_hub import CommitScheduler

# Définir le fichier où sauvegarder les données. Utiliser UUID pour s'assurer de ne pas écraser les données existantes d'une exécution précédente.
>>> feedback_file = Path("user_feedback/") / f"data_{uuid.uuid4()}.json"
>>> feedback_folder = feedback_file.parent

# Planifier des uploads réguliers. Le dépôt distant et le dossier local sont créés s'ils n'existent pas déjà.
>>> scheduler = CommitScheduler(
...     repo_id="report-translation-feedback",
...     repo_type="dataset",
...     folder_path=feedback_folder,
...     path_in_repo="data",
...     every=10,
... )

# Définir la fonction qui sera appelée lorsque l'utilisateur soumettra son feedback (à appeler dans Gradio)
>>> def save_feedback(input_text:str, output_1: str, output_2:str, user_choice: int) -> None:
...     """
...     Ajouter les entrées/sorties et le feedback utilisateur à un fichier JSON Lines en utilisant un verrou de thread pour éviter les écritures concurrentes de différents utilisateurs.
...     """
...     with scheduler.lock:
...         with feedback_file.open("a") as f:
...             f.write(json.dumps({"input": input_text, "output_1": output_1, "output_2": output_2, "user_choice": user_choice}))
...             f.write("\n")

# Démarrer Gradio
>>> with gr.Blocks() as demo:
>>>     ... # définir la démo Gradio + utiliser `save_feedback`
>>> demo.launch()
```

C'est tout ! Les entrées/sorties utilisateur et le feedback seront disponibles comme un dataset sur le Hub. En utilisant un nom de fichier JSON unique, vous êtes assuré de ne pas écraser les données d'une exécution précédente ou les données d'autres
Spaces/répliques poussant simultanément vers le même dépôt.

Pour plus de détails sur le [`CommitScheduler`], voici ce que vous devez savoir :
- **ajout uniquement :**
    Il est supposé que vous ne ferez qu'ajouter du contenu au dossier. Supprimer ou écraser un fichier pourrait corrompre votre dépôt.
- **historique git** :
    Le scheduler commitera le dossier toutes les `every` minutes. Pour éviter de polluer trop le dépôt git, il est
    recommandé de définir une valeur minimale de 5 minutes. De plus, le scheduler est conçu pour éviter les commits vides. Si aucun
    nouveau contenu n'est détecté dans le dossier, le commit programmé est abandonné.
- **erreurs :**
    Le scheduler fonctionne comme un thread en arrière-plan. Il est démarré lorsque vous instanciez la classe et ne s'arrête jamais. En particulier,
    si une erreur se produit pendant l'upload (exemple : problème de connexion), le scheduler l'ignorera silencieusement et réessayera
    au prochain commit programmé.

#### Démo de persistance de Space

Persister les données d'un Space vers un Dataset sur le Hub est le principal cas d'usage pour [`CommitScheduler`]. Selon le cas
d'usage, vous pourriez vouloir structurer vos données différemment. La structure doit être robuste aux utilisateurs concurrents et
aux redémarrages, ce qui implique souvent de générer des UUIDs. En plus de la robustesse, vous devriez uploader des données dans un format lisible par la bibliothèque 🤗. Nous avons créé un [Space](https://huggingface.co/spaces/Wauplin/space_to_dataset_saver)
qui montre comment sauvegarder plusieurs formats de données différents (vous pourriez avoir besoin de l'adapter pour vos propres besoins spécifiques).

#### Uploads personnalisés

[`CommitScheduler`] suppose que vos données sont en ajout uniquement et doivent être uploadées "telles quelles". Cependant, vous
pourriez vouloir personnaliser la façon dont les données sont uploadées. Vous pouvez le faire en créant une classe héritant de [`CommitScheduler`]
et en écrasant la méthode `push_to_hub` (n'hésitez pas à l'écraser comme vous le souhaitez). Vous êtes assuré qu'elle sera
appelée toutes les `every` minutes dans un thread en arrière-plan. Vous n'avez pas à vous soucier de la concurrence et des erreurs, mais vous
devez faire attention à d'autres aspects, comme pousser des commits vides ou des données dupliquées.

Dans l'exemple (simplifié) ci-dessous, nous écrasons `push_to_hub` pour zipper tous les fichiers PNG dans une seule archive afin d'éviter
de surcharger le dépôt sur le Hub :

```py
class ZipScheduler(CommitScheduler):
    def push_to_hub(self):
        # 1. Lister les fichiers PNG
          png_files = list(self.folder_path.glob("*.png"))
          if len(png_files) == 0:
              return None  # retourner tôt s'il n'y a rien à commiter

        # 2. Zipper les fichiers png dans une seule archive
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "train.zip"
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zip:
                for png_file in png_files:
                    zip.write(filename=png_file, arcname=png_file.name)

            # 3. Uploader l'archive
            self.api.upload_file(..., path_or_fileobj=archive_path)

        # 4. Supprimer les fichiers png locaux pour éviter de les re-uploader plus tard
        for png_file in png_files:
            png_file.unlink()
```

Lorsque vous écrasez `push_to_hub`, vous avez accès aux attributs de [`CommitScheduler`] et en particulier :
- Client [`HfApi`] : `api`
- Paramètres du dossier : `folder_path` et `path_in_repo`
- Paramètres du dépôt : `repo_id`, `repo_type`, `revision`
- Le verrou de thread : `lock`

> [!TIP]
> Pour plus d'exemples de schedulers personnalisés, consultez notre [Space de démo](https://huggingface.co/spaces/Wauplin/space_to_dataset_saver)
> contenant différentes implémentations selon vos cas d'usage.

### create_commit

Les fonctions [`upload_file`] et [`upload_folder`] sont des APIs qui sont généralement pratiques à utiliser. Nous recommandons
d'essayer ces fonctions en premier. Cependant, si vous voulez travailler au niveau du commit,
vous pouvez utiliser directement la fonction [`create_commit`].

Il existe trois types d'opérations supportés par [`create_commit`] :

- [`CommitOperationAdd`] uploade un fichier sur le Hub. Si le fichier existe déjà, le contenu du fichier est écrasé. Cette opération accepte deux arguments :

  - `path_in_repo` : le chemin du dépôt vers lequel uploader un fichier.
  - `path_or_fileobj` : soit un chemin vers un fichier sur votre système de fichiers, soit un objet file-like. C'est le contenu du fichier à uploader sur le Hub.

- [`CommitOperationDelete`] supprime un fichier ou un dossier d'un dépôt. Cette opération accepte `path_in_repo` comme argument.

- [`CommitOperationCopy`] copie un fichier dans un dépôt. Cette opération accepte trois arguments :

  - `src_path_in_repo` : le chemin du dépôt du fichier à copier.
  - `path_in_repo` : le chemin du dépôt où le fichier doit être copié.
  - `src_revision` : optionnel - la révision du fichier à copier si vous voulez copier un fichier depuis une branche/révision différente.

Par exemple, si vous voulez uploader deux fichiers et supprimer un fichier dans un dépôt Hub :

1. Utilisez le `CommitOperation` approprié pour ajouter ou supprimer un fichier et pour supprimer un dossier :

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

2. Passez vos opérations à [`create_commit`] :

```py
>>> api.create_commit(
...     repo_id="lysandre/test-model",
...     operations=operations,
...     commit_message="Upload my model weights and license",
... )
```

En plus de [`upload_file`] et [`upload_folder`], les fonctions suivantes utilisent également [`create_commit`] en interne:

- [`delete_file`] supprime un seul fichier d'un dépôt sur le Hub.
- [`delete_folder`] supprime un dossier entier d'un dépôt sur le Hub.
- [`metadata_update`] met à jour les métadonnées d'un dépôt.

Pour des informations plus détaillées, consultez la référence [`HfApi`].

### Pré-uploader les fichiers LFS avant le commit

Dans certains cas, vous pourriez vouloir uploader d'énormes fichiers vers S3 **avant** de faire l'appel commit. Par exemple, si vous
commitez un dataset en plusieurs shards qui sont générés en mémoire, vous auriez besoin d'uploader les shards un par un
pour éviter un problème de mémoire insuffisante. Une solution est d'uploader chaque shard comme un commit séparé sur le dépôt. Bien qu'étant
parfaitement valide, cette solution a l'inconvénient de potentiellement salir l'historique git en générant des dizaines de commits.
Pour surmonter ce problème, vous pouvez uploader vos fichiers un par un vers S3 puis créer un seul commit à la fin. Ceci
est possible en utilisant [`preupload_lfs_files`] en combinaison avec [`create_commit`].

> [!WARNING]
> Ceci est une méthode pour utilisateur expérimenté. Utiliser directement [`upload_file`], [`upload_folder`] ou [`create_commit`] au lieu de gérer
> la logique de bas niveau de pré-upload. Le principal inconvénient de
> [`preupload_lfs_files`] est que jusqu'à ce que le commit soit réellement fait, les fichiers uploadés ne sont pas accessibles sur le dépôt sur
> le Hub. Si vous avez une question, n'hésitez pas à nous contacter sur notre Discord ou dans une issue GitHub.

Voici un exemple simple illustrant comment pré-uploader des fichiers :

```py
>>> from huggingface_hub import CommitOperationAdd, preupload_lfs_files, create_commit, create_repo

>>> repo_id = create_repo("test_preupload").repo_id

>>> operations = [] # Liste de tous les objets `CommitOperationAdd` qui seront générés
>>> for i in range(5):
...     content = ... # générer du contenu binaire
...     addition = CommitOperationAdd(path_in_repo=f"shard_{i}_of_5.bin", path_or_fileobj=content)
...     preupload_lfs_files(repo_id, additions=[addition])
...     operations.append(addition)

>>> # Créer le commit
>>> create_commit(repo_id, operations=operations, commit_message="Commit all shards")
```

D'abord, nous créons les objets [`CommitOperationAdd`] un par un. Dans un exemple réel, ceux-ci contiendraient les
shards générés. Chaque fichier est uploadé avant de générer le suivant. Pendant l'étape [`preupload_lfs_files`], **l'
objet `CommitOperationAdd` est muté**. Vous devriez uniquement l'utiliser pour le passer directement à [`create_commit`]. La principale
mise à jour de l'objet est que **le contenu binaire en est retiré**, ce qui signifie qu'il sera récupéré par le garbage collector si
vous ne conservez pas une autre référence à celui-ci. Ceci est totalement normal car nous ne voulons pas garder en mémoire le contenu qui est
déjà uploadé. Enfin, nous créons le commit en passant toutes les opérations à [`create_commit`]. Vous pouvez passer
des opérations supplémentaires (ajouter, supprimer ou copier).
