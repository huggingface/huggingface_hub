<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Gérez votre space

Dans ce guide, nous allons voir comment gérer le temps de calcul sur votre space,
([les secrets](https://huggingface.co/docs/hub/spaces-overview#managing-secrets),
[le hardware](https://huggingface.co/docs/hub/spaces-gpus), et [le stockage](https://huggingface.co/docs/hub/spaces-storage#persistent-storage))
en utilisant `huggingface_hub`.

## Un exemple simple: configurez les secrets et le hardware.

Voici un exemple de A à Z pour créer et mettre en place un space sur le Hub.

**1. Créez un space sur le Hub.**

```py
>>> from huggingface_hub import HfApi
>>> repo_id = "Wauplin/my-cool-training-space"
>>> api = HfApi()

# Par exemple, avec un SDK Gradio
>>> api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio")
```

**1. (bis) Dupliquez un space.**

Ceci peut-être utile si vous voulez construire depuis un space existant aulieu de commencer à zéro.
C'est aussi utile si vous voulez controler la configuration et les paramètres d'un space publique.
Consultez [`duplicate_space`] pour plus de détails.

```py
>>> api.duplicate_space("multimodalart/dreambooth-training")
```

**2. Uploadez votre code en utilisant votre solution préférée.**

Voici un exemple pour upload le dossier local `src/` depuis votre machine vers votre space:

```py
>>> api.upload_folder(repo_id=repo_id, repo_type="space", folder_path="src/")
```

A ce niveau là, votre application devrait déjà être en train de tourner sur le Hub gratuitement !
Toutefois, vous voudez peut-être la configurer plus en détail avec des secrets et un
meilleur hardware.

**3. Configurez des secrets et des variables**

Votre space aura peut-être besoin d'une clef secrète, un token ou de variables
pour fonctionner. Consultez [la doc](https://huggingface.co/docs/hub/spaces-overview#managing-secrets)
pour plus de détails. Par exemple, un token HF pour upload un dataset d'image vers le Hub
une fois généré depuis votre space.

```py
>>> api.add_space_secret(repo_id=repo_id, key="HF_TOKEN", value="hf_api_***")
>>> api.add_space_variable(repo_id=repo_id, key="MODEL_REPO_ID", value="user/repo")
```

Les secrets et les variables peuvent supprimés aussi:
```py
>>> api.delete_space_secret(repo_id=repo_id, key="HF_TOKEN")
>>> api.delete_space_variable(repo_id=repo_id, key="MODEL_REPO_ID")
```

<Tip>
Depuis votre space, les secrets sont définissables en tant que variables
(ou en tant que management de secrets Streamlit si vous utilisez Streamlit).
Pas besoin de les ajouter via l'API!
</Tip>

<Tip warning={true}>
Tout changement dans la configuration de votre space (secrets ou hardware) relancera votre
application.
</Tip>

**Bonus: définissez les secrets et les variables lors de la création ou la duplication du space!**

Les secrets et les variables peuvent être défini lors de la création ou la duplication d'un space:

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

**4. Configurez le hardware**

Par défaut, votre space tournera sur un CPU gratuitement. Vous pouvez améliorer le
hardware pour le faire tourner sur des GPUs. Une carte bleue ou un community grant sera
nécessaire pour accéder à l'amélioration de votre space. Consultez [la doc](https://huggingface.co/docs/hub/spaces-gpus)
pour plus de détails.

```py
# Utilisez l'enum `SpaceHardware`
>>> from huggingface_hub import SpaceHardware
>>> api.request_space_hardware(repo_id=repo_id, hardware=SpaceHardware.T4_MEDIUM)

# Ou simplement passez un string
>>> api.request_space_hardware(repo_id=repo_id, hardware="t4-medium")
```

Les mises à jour d'hardware ne sont pas faites immédiatement vu que votre space doit
être rechargé sur nos serveurs. A n'importe quel moment, vous pouvez vérifier sur quel
hardware votre space tourne pour vérifier que votre demande a été réalisée.

```py
>>> runtime = api.get_space_runtime(repo_id=repo_id)
>>> runtime.stage
"RUNNING_BUILDING"
>>> runtime.hardware
"cpu-basic"
>>> runtime.requested_hardware
"t4-medium"
```

Vous avez maintenant un space totalement configuré. Une fois que vous avez fini avec les
GPUs, assurez vous de revenir à "cpu-classic".
You now have a Space fully configured. Make sure to downgrade your Space back to "cpu-classic"
when you are done using it.

**Bonus: demandez du hardware lors de la création ou la duplication d'un space!**

Les nouvel hardware sera automatiquement assigné à votre space une fois qu'il
a été construit.

```py
>>> api.create_repo(
...     repo_id=repo_id,
...     repo_type="space",
...     space_sdk="gradio"
...     space_hardware="cpu-upgrade",
...     space_storage="small",
...     space_sleep_time="7200", # 2 heure en secondes
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

**5. Mettez en pause et relancez votre space**

Par défaut, si votre space tourne sur un hardware amélioré, il ne sera jamais arrêté. Toutefois pour éviter de vous
faire facturer, vous aurez surement besoin de le mettre en pause lorsque vous ne l'utilisez pas. C'est possible en
utilisant [`pause_space`]. Un space en pause sera inactif tant que le propriétaire du space ne l'a pas relancé,
soit avec l'interface utliisateur ou via l'API en utilisant [`restart_space`]. Pour plus de détails sur le mode "en pause",
consultez [cette section](https://huggingface.co/docs/hub/spaces-gpus#pause) du guide.

```py
# Met en pause le space pour éviter de payer
>>> api.pause_space(repo_id=repo_id)
# (...)
# Relance le space quand vous en avez besoin
>>> api.restart_space(repo_id=repo_id)
```

Une auter possibilité est de définir un timeout pour votre space. Si votre space est inactif pour une durée
plus grande que la durée de timeout, alors il se mettra en pause automatiquement. N'importe quel visiteur qui
arrive sur votre space le relancera. Vous pouvez définir un timeout en utilisant [`set_space_sleep_time`].
Pour plus de détails sur le mode "en pause", consultez [cette section](https://huggingface.co/docs/hub/spaces-gpus#sleep-time).

```py
# Met le space en pause après une heure d'inactivité
>>> api.set_space_sleep_time(repo_id=repo_id, sleep_time=3600)
```

Note: si vous utlisez du du hardware 'cpu-basic', vous ne pouvez pas configurer un timeout personnalisé. Votre space
se mettra en pause automatiquement aprèss 48h d'inactivité.

**Bonus: définissez le temps de timeout lorsque vous demandez le hardware**

Le hardware amélioré sera automatiquement assigné à votre space une fois construit.

```py
>>> api.request_space_hardware(repo_id=repo_id, hardware=SpaceHardware.T4_MEDIUM, sleep_time=3600)
```

**Bonus: définissez un temps de timeout lors de la création ou de la duplication d'un space!**

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

**6. Ajoutez du stockage persistant à votre space**

Vous pouvez choisir le stockage de votre choix pour accéder au disque dont la mémoire ne s'écrase pas lors du redémarrage du space. Ceci signifie que
vous pouvez lire et écrire sur ce disque comme vous l'auriez fait avec un disque dur traditionnel.
Consultez See [la doc](https://huggingface.co/docs/hub/spaces-storage#persistent-storage) pour plus de détails.

```py
>>> from huggingface_hub import SpaceStorage
>>> api.request_space_storage(repo_id=repo_id, storage=SpaceStorage.LARGE)
```

Vous pouvez aussi supprimer votre stockage, mais vous perdrez toute la donnée de manière irréversible.
```py
>>> api.delete_space_storage(repo_id=repo_id)
```

Note: Vous ne pouvez pas diminuer le niveau de stockage de votre space une fois qu'il a été
donné. Pour ce faire, vous devez d'abord supprimer le stockage
(attention, les données sont supprimés définitivement) puis demander le niveau de stockage désiré.

**Bonus: demandez du stockage lors de la création ou la duplication du space!**

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

## Avancé: améliorez votre space pour un durée déterminée!

Les spaces ont un grand nombre de cas d'application. Parfois, vous aurez
peut-être envie de faire un tourner un space pendant une durée déterminée sur un hardware
spécifique, faire quelque chose puis éteindre le space. dans cette section, nous explorerons
les avantgaes des spaces pour finetune un modèle sur demande. C'est la seule manière de
résoudre ce problème. Toutefois, ces tutoriaux ne sont que des suggestions et doivent être
adaptés à votre cas d'usage.

Supposons que nous avons un space pour finetune un modèle. C'est une aplication Gradio qui
prend en entrée l'id d'un modèle et l'id d'un dataset. Le workflow est le suivant:

0.Demander à l'utilisateur un modèle et un dataset.
1.Charger le modèle depuis le Hub.
2.Charger le dataset depuis le Hub.
3.Finetune le modèle sur le dataset.
4.Upload le nouveau modèle vers le Hub.

La 3ème étape demande un hardware personnalisé mais vous n'aurez surement pas envie que votre space
tourne tout le temps sur un GPU que vous payez. Une solution est de demander du hardware de manière
dynmaique pour l'entrainement et l'éteindre après. Vu que demander du hardware redémarre voter space,
votre application doit d'une manière ou d'une autre "se rappeler" la tache qu'il est entrain de réaliser.
Il y a plusieurs manières de faire ceci. Dans ce guide, nous verrons une solution utilisant un dataset
qui fera office de "programmateur de tâche".

### Le squelette de l'application

Voici ce à quoi votre application ressemblerait. Au lancement, elle vérifie si une tâche est
programmée et si oui, cette tâche sera lancée sur le bon hardware. Une fois fini, le
hardware est remis au CPU du plan gratuit et demande à l'utilisateur une nouvelle tâche.

<Tip warning={true}>
Un tel workflow ne permet pas un accès simultané en tant que démo
normales. En particulier, l'interface sera supprimée lors de
l'entrainement. il est préférable de mettre votre dépôt en privé
pour vous assurer que vous êtes le seul utilisateur.
</Tip>

```py
# Un space aura besoin de votre token pour demander du hardware: définissez le en temps que secret!
HF_TOKEN = os.environ.get("HF_TOKEN")

# Le repo_id du space
TRAINING_SPACE_ID = "Wauplin/dreambooth-training"

from huggingface_hub import HfApi, SpaceHardware
api = HfApi(token=HF_TOKEN)

# Lors du lancement du space, vérifiez si une tâche est programmée. Si oui, finetunez le modèle. Si non,
# affichez une interface pour demander une nouvelle tâche.
task = get_task()
if task is None:
    # Lancez l'application Gradio
    def gradio_fn(task):
        # Lorsque l'utilisateur le demande, ajoutez une tâche et demandez du hardware.
        add_task(task)
        api.request_space_hardware(repo_id=TRAINING_SPACE_ID, hardware=SpaceHardware.T4_MEDIUM)

    gr.Interface(fn=gradio_fn, ...).launch()
else:
    runtime = api.get_space_runtime(repo_id=TRAINING_SPACE_ID)
    # Vérifiez si le space est chargé avec un GPU.
    if runtime.hardware == SpaceHardware.T4_MEDIUM:
        # Si oui, fintunez le modèle de base sur le dataset!
        train_and_upload(task)

        # Ensuite, signalez la tâche comme finie
        mark_as_done(task)

        # N'OUBLIEZ PAS: remettez le hardware en mode CPU
        api.request_space_hardware(repo_id=TRAINING_SPACE_ID, hardware=SpaceHardware.CPU_BASIC)
    else:
        api.request_space_hardware(repo_id=TRAINING_SPACE_ID, hardware=SpaceHardware.T4_MEDIUM)
```

### Le task scheduler

Programmer une tâche peut-être fait de plusieurs manières. Voici un exemple de comment
on pourrait le faire en utilisant un simple CSV enregistré en tant que dataset.

```py
# ID du dataset dans lequel un fichier `task.csv` cotient la tâche à accomplir.
# Voici un exemple basique pour les inputs de `tasks.csv` et le status PEDING (en cours) ou DONE (fait).
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
        # Manière simple et inélégante d'ajouter une tâche
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
        # Manière simple et inélégante de marquer une tâche comme DONE
        path_or_fileobj=tasks.replace(
            f"{model_id},{dataset_id},PENDING",
            f"{model_id},{dataset_id},DONE"
        ).encode()
    )
```