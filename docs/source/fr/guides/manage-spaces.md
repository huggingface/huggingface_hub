<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Gérer votre Space

Dans ce guide, nous allons voir comment gérer le runtime de votre Space ([secrets](https://huggingface.co/docs/hub/spaces-overview#managing-secrets), [hardware](https://huggingface.co/docs/hub/spaces-gpus), et [stockage](https://huggingface.co/docs/hub/spaces-storage#persistent-storage)) en utilisant `huggingface_hub`.

## Un exemple simple : configurer les secrets et le hardware

Voici un exemple de bout en bout pour créer et configurer un Space sur le Hub.

**1. Créer un Space sur le Hub.**

```py
>>> from huggingface_hub import HfApi
>>> repo_id = "Wauplin/my-cool-training-space"
>>> api = HfApi()

# Par exemple avec un SDK Gradio
>>> api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio")
```

**1. (bis) Dupliquer un Space.**

Cela peut s'avérer utile si vous voulez vous baser sur un Space existant au lieu de partir de zéro. C'est également utile si vous voulez avoir le contrôle sur la configuration/paramètres d'un Space public. Voir [`duplicate_space`] pour plus de détails.

```py
>>> api.duplicate_space("multimodalart/dreambooth-training")
```

**2. Uploader votre code en utilisant votre solution préférée.**

Voici un exemple pour uploader le dossier local `src/` depuis votre machine vers votre Space :

```py
>>> api.upload_folder(repo_id=repo_id, repo_type="space", folder_path="src/")
```

À cette étape, votre app devrait déjà être en cours d'exécution sur le Hub gratuitement ! Cependant, vous pourriez vouloir la configurer davantage avec des secrets et du hardware amélioré.

**3. Configurer les secrets et variables**

Votre Space pourrait nécessiter quelques clés secrètes, tokens ou variables pour fonctionner. Voir [docs](https://huggingface.co/docs/hub/spaces-overview#managing-secrets) pour plus de détails. Par exemple, un token HF pour uploader un dataset d'images vers le Hub une fois généré depuis votre Space.

```py
>>> api.add_space_secret(repo_id=repo_id, key="HF_TOKEN", value="hf_api_***")
>>> api.add_space_variable(repo_id=repo_id, key="MODEL_REPO_ID", value="user/repo")
```

Les secrets et variables peuvent également être supprimés :

```py
>>> api.delete_space_secret(repo_id=repo_id, key="HF_TOKEN")
>>> api.delete_space_variable(repo_id=repo_id, key="MODEL_REPO_ID")
```

> [!TIP]
> Depuis l'intérieur de votre Space, les secrets sont disponibles comme variables d'environnement (ou Streamlit Secrets Management si vous utilisez Streamlit). Pas besoin de les récupérer via l'API !

> [!WARNING]
> Tout changement dans la configuration de votre Space (secrets ou hardware) déclenchera un redémarrage de votre app.

**Bonus : définir les secrets et variables lors de la création ou duplication du Space !**

Les secrets et variables peuvent être définis lors de la création ou duplication d'un space :

```py
>>> api.create_repo(
...     repo_id=repo_id,
...     repo_type="space",
...     space_sdk="gradio",
...     space_secrets=[{"key"="HF_TOKEN", "value"="hf_api_***"}, ...],
...     space_variables=[{"key"="MODEL_REPO_ID", "value"="user/repo"}, ...],
... )
```

```py
>>> api.duplicate_space(
...     from_id=repo_id,
...     secrets=[{"key"="HF_TOKEN", "value"="hf_api_***"}, ...],
...     variables=[{"key"="MODEL_REPO_ID", "value"="user/repo"}, ...],
... )
```

**4. Configurer le hardware**

Par défaut, votre Space fonctionnera sur un environnement CPU gratuitement. Vous pouvez améliorer le hardware pour l'exécuter sur des GPUs. Une carte bancaire ou une subvention est requise pour accéder à l'amélioration de votre Space. Voir [docs](https://huggingface.co/docs/hub/spaces-gpus) pour plus de détails.

```py
# Utiliser l'enum `SpaceHardware`
>>> from huggingface_hub import SpaceHardware
>>> api.request_space_hardware(repo_id=repo_id, hardware=SpaceHardware.T4_MEDIUM)

# Ou simplement passer une valeur string
>>> api.request_space_hardware(repo_id=repo_id, hardware="t4-medium")
```

Les mises à jour hardware ne sont pas effectuées immédiatement car votre Space doit être rechargé sur nos serveurs. À tout moment, vous pouvez vérifier sur quel hardware votre Space est en cours d'exécution.

```py
>>> runtime = api.get_space_runtime(repo_id=repo_id)
>>> runtime.stage
"RUNNING_BUILDING"
>>> runtime.hardware
"cpu-basic"
>>> runtime.requested_hardware
"t4-medium"
```

Vous avez maintenant un Space entièrement configuré. Assurez-vous de rétrograder votre Space vers "cpu-classic" lorsque vous avez fini de l'utiliser afin d'éviter des frais inutiles.

**Bonus : demander du hardware lors de la création ou duplication du Space !**

Le hardware amélioré sera automatiquement attribué à votre Space une fois qu'il sera construit.

```py
>>> api.create_repo(
...     repo_id=repo_id,
...     repo_type="space",
...     space_sdk="gradio"
...     space_hardware="cpu-upgrade",
...     space_storage="small",
...     space_sleep_time="7200", # 2 heures en secondes
... )
```

```py
>>> api.duplicate_space(
...     from_id=repo_id,
...     hardware="cpu-upgrade",
...     storage="small",
...     sleep_time="7200", # 2 heures en secondes
... )
```

**5. Mettre en pause et redémarrer votre Space**

Par défaut, si votre Space fonctionne sur un hardware amélioré, il ne sera jamais arrêté. Cependant, pour éviter d'être facturé, vous pourriez vouloir le mettre en pause lorsque vous ne l'utilisez pas. Cela est possible en utilisant [`pause_space`]. Un Space en pause sera inactif jusqu'à ce que le propriétaire du Space le redémarre, soit avec l'UI soit via l'API en utilisant [`restart_space`]. Pour plus de détails sur le mode pause, veuillez vous référer à [cette section](https://huggingface.co/docs/hub/spaces-gpus#pause).

```py
# Mettre en pause votre Space pour éviter d'être facturé
>>> api.pause_space(repo_id=repo_id)
# (...)
# Le redémarrer lorsque vous en avez besoin
>>> api.restart_space(repo_id=repo_id)
```

Une autre possibilité est de définir un timeout pour votre Space. Si votre Space est inactif pendant plus que la durée du timeout, il se mettra en veille. Tout visiteur arrivant sur votre Space le redémarrera. Vous pouvez définir un timeout en utilisant [`set_space_sleep_time`]. Pour plus de détails sur le mode veille, veuillez vous référer à [cette section](https://huggingface.co/docs/hub/spaces-gpus#sleep-time).

```py
# Mettre votre Space en veille après 1h d'inactivité
>>> api.set_space_sleep_time(repo_id=repo_id, sleep_time=3600)
```

Note : si vous utilisez un hardware 'cpu-basic', vous ne pouvez pas configurer un temps de veille personnalisé. Votre Space sera automatiquement mis en pause après 48h d'inactivité.

**Bonus : définir un temps de veille lors de la demande de hardware**

Le hardware amélioré sera automatiquement attribué à votre Space une fois qu'il sera construit.

```py
>>> api.request_space_hardware(repo_id=repo_id, hardware=SpaceHardware.T4_MEDIUM, sleep_time=3600)
```

**Bonus : définir un temps de veille lors de la création ou duplication du Space !**

```py
>>> api.create_repo(
...     repo_id=repo_id,
...     repo_type="space",
...     space_sdk="gradio"
...     space_hardware="t4-medium",
...     space_sleep_time="3600",
... )
```

```py
>>> api.duplicate_space(
...     from_id=repo_id,
...     hardware="t4-medium",
...     sleep_time="3600",
... )
```

**6. Ajouter un stockage persistant à votre Space**

Vous pouvez choisir le niveau de stockage pour accéder à un espace disque qui persiste à travers les redémarrages de votre Space. Cela signifie que vous pouvez lire et écrire depuis le disque comme vous le feriez avec un disque dur traditionnel. Voir [docs](https://huggingface.co/docs/hub/spaces-storage#persistent-storage) pour plus de détails.

```py
>>> from huggingface_hub import SpaceStorage
>>> api.request_space_storage(repo_id=repo_id, storage=SpaceStorage.LARGE)
```

Vous pouvez également supprimer votre stockage, vous perdrez toutes les données de façon permanente.

```py
>>> api.delete_space_storage(repo_id=repo_id)
```

Note : Vous ne pouvez pas diminuer le niveau de stockage de votre space une fois qu'il a été accordé. Pour ce faire, vous devez d'abord supprimer le stockage puis demander la nouvelle taille souhaitée.

**Bonus : demander du stockage lors de la création ou duplication du Space !**

```py
>>> api.create_repo(
...     repo_id=repo_id,
...     repo_type="space",
...     space_sdk="gradio"
...     space_storage="large",
... )
```

```py
>>> api.duplicate_space(
...     from_id=repo_id,
...     storage="large",
... )
```

## Plus avancé : améliorer temporairement votre Space !

Les Spaces permettent beaucoup de cas d'utilisation différents. Parfois, vous pourriez vouloir exécuter temporairement un Space sur un hardware spécifique, faire quelque chose puis l'arrêter. Dans cette section, nous allons explorer comment bénéficier des Spaces pour finetuner un modèle à la demande. Ceci n'est qu'une façon de résoudre ce problème. Il doit être pris comme une suggestion et adapté à votre cas d'utilisation.

Supposons que nous ayons un Space pour finetuner un modèle. C'est une app Gradio qui prend en entrée un model id et un dataset id. Le workflow est le suivant :

0. (Demander à l'utilisateur un modèle et un dataset)
1. Charger le modèle depuis le Hub.
2. Charger le dataset depuis le Hub.
3. Finetuner le modèle sur le dataset.
4. Uploader le nouveau modèle vers le Hub.

L'étape 3 nécessite un hardware personnalisé mais vous ne voulez pas que votre Space soit en cours d'exécution tout le temps sur un GPU payant. Une solution est de demander dynamiquement du hardware pour l'entraînement et l'arrêter ensuite. Comme demander du hardware redémarre votre Space, votre app doit d'une manière ou d'une autre "se souvenir" de la tâche actuelle qu'elle effectue. Il y a plusieurs façons de faire cela. Dans ce guide, nous allons voir une des solutions utilisant un Dataset comme "planificateur de tâches".

### Squelette de l'app

Voici à quoi ressemblerait votre app. Au démarrage, vérifiez si une tâche est planifiée et si oui, l'exécuter sur le hardware correct. Une fois terminé, redéfinir le hardware vers le CPU du plan gratuit et demander à l'utilisateur une nouvelle tâche.

> [!WARNING]
> Un tel workflow ne supporte pas l'accès concurrent comme les démos normales. En particulier, l'interface sera désactivée lors de l'entraînement. Il est préférable de définir votre dépôt comme privé pour vous assurer que vous êtes le seul utilisateur.

```py
# Le Space aura besoin de votre token pour demander du hardware : définissez-le comme un Secret !
HF_TOKEN = os.environ.get("HF_TOKEN")

# repo_id propre au Space
TRAINING_SPACE_ID = "Wauplin/dreambooth-training"

from huggingface_hub import HfApi, SpaceHardware
api = HfApi(token=HF_TOKEN)

# Au démarrage du Space, vérifier si une tâche est planifiée. Si oui, finetuner le modèle. Sinon,
# afficher une interface pour demander une nouvelle tâche.
task = get_task()
if task is None:
    # Démarrer l'app Gradio
    def gradio_fn(task):
        # Sur la demande de l'utilisateur, ajouter la tâche et demander du hardware
        add_task(task)
        api.request_space_hardware(repo_id=TRAINING_SPACE_ID, hardware=SpaceHardware.T4_MEDIUM)

    gr.Interface(fn=gradio_fn, ...).launch()
else:
    runtime = api.get_space_runtime(repo_id=TRAINING_SPACE_ID)
    # Vérifier si le Space est chargé avec un GPU.
    if runtime.hardware == SpaceHardware.T4_MEDIUM:
        # Si oui, finetuner le modèle de base sur le dataset !
        train_and_upload(task)

        # Ensuite, marquer la tâche comme "DONE"
        mark_as_done(task)

        # NE PAS OUBLIER : redéfinir le hardware CPU
        api.request_space_hardware(repo_id=TRAINING_SPACE_ID, hardware=SpaceHardware.CPU_BASIC)
    else:
        api.request_space_hardware(repo_id=TRAINING_SPACE_ID, hardware=SpaceHardware.T4_MEDIUM)
```

### Planificateur de tâches

La planification des tâches peut être effectuée de nombreuses façons. Voici un exemple de comment cela pourrait être fait en utilisant un simple CSV stocké comme Dataset.

```py
# ID du Dataset dans lequel un fichier `tasks.csv` contient les tâches à effectuer.
# Voici un exemple basique pour `tasks.csv` contenant les entrées (modèle de base et dataset)
# et le statut (PENDING ou DONE).
#     multimodalart/sd-fine-tunable,Wauplin/concept-1,DONE
#     multimodalart/sd-fine-tunable,Wauplin/concept-2,PENDING
TASK_DATASET_ID = "Wauplin/dreambooth-task-scheduler"

def _get_csv_file():
    return hf_hub_download(repo_id=TASK_DATASET_ID, filename="tasks.csv", repo_type="dataset", token=HF_TOKEN)

def get_task():
    with open(_get_csv_file()) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[2] == "PENDING":
                return row[0], row[1] # model_id, dataset_id

def add_task(task):
    model_id, dataset_id = task
    with open(_get_csv_file()) as csv_file:
        with open(csv_file, "r") as f:
            tasks = f.read()

    api.upload_file(
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo="tasks.csv",
        # Méthode rapide et simple pour ajouter une tâche
        path_or_fileobj=(tasks + f"\n{model_id},{dataset_id},PENDING").encode()
    )

def mark_as_done(task):
    model_id, dataset_id = task
    with open(_get_csv_file()) as csv_file:
        with open(csv_file, "r") as f:
            tasks = f.read()

    api.upload_file(
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo="tasks.csv",
        # Méthode rapide et simple pour définir la tâche comme DONE
        path_or_fileobj=tasks.replace(
            f"{model_id},{dataset_id},PENDING",
            f"{model_id},{dataset_id},DONE"
        ).encode()
    )
```
