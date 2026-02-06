<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Exécuter et gérer des Jobs

Le Hugging Face Hub fournit du calcul pour les workflows AI et data via les Jobs. Un job s'exécute sur l'infrastructure Hugging Face et est défini avec une commande à exécuter (par exemple une commande python), une Image Docker depuis Hugging Face Spaces ou Docker Hub, et une couche de hardware (CPU, GPU, TPU). Ce guide vous montrera comment interagir avec les Jobs sur le Hub:

- Exécuter un job.
- Vérifier le statut d'un job.
- Sélectionner le hardware.
- Configurer les variables d'environnement et secrets.
- Exécuter des scripts UV.

Si vous voulez exécuter et gérer un job sur le Hub, votre machine doit être authentifiée. Si ce n'est pas le cas, veuillez vous référer à [cette section](../quick-start#authentication). Dans le reste de ce guide, nous supposerons que votre machine est connectée à votre compte Hugging Face.

> [!TIP]
> **Hugging Face Jobs** est disponible uniquement pour les [utilisateurs Pro](https://huggingface.co/pro) et les [organisations Team ou Enterprise](https://huggingface.co/enterprise). Mettez à niveau votre abonnement pour commencer !

## Interface en ligne de commande pour les Jobs

Utilisez le CLI [`hf jobs`](./cli#hf-jobs) pour exécuter des Jobs depuis la ligne de commande, et passez `--flavor` pour spécifier votre hardware.

`hf jobs run` exécute des Jobs avec une image Docker et une commande avec une interface familière similaire à Docker. Pensez `docker run`, mais pour exécuter du code sur n'importe quel hardware :

```bash
>>> hf jobs run python:3.12 python -c "print('Hello world!')"
>>> hf jobs run --flavor a10g-small pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel python -c "import torch; print(torch.cuda.get_device_name())"
```

Utilisez `hf jobs uv run` pour exécuter des scripts UV locaux ou distants :

```bash
>>> hf jobs uv run my_script.py
>>> hf jobs uv run --flavor a10g-small "https://raw.githubusercontent.com/huggingface/trl/main/trl/scripts/sft.py" 
```

Les scripts UV sont des scripts Python qui incluent leurs dépendances directement dans le fichier en utilisant une syntaxe de commentaire spéciale définie dans la [documentation UV](https://docs.astral.sh/uv/guides/scripts/).

Maintenant, le reste de ce guide vous montrera l'API python. Si vous souhaitez plutôt voir toutes les commandes et options `hf jobs` disponibles, consultez le [guide sur l'interface en ligne de commande `hf jobs`](./cli#hf-jobs).

## Exécuter un Job

Exécutez des Jobs de calcul définis avec une commande et une Image Docker sur l'infrastructure Hugging Face (y compris les GPUs et TPUs).

Vous ne pouvez gérer que les Jobs que vous possédez (sous votre namespace de nom d'utilisateur) ou des organisations dans lesquelles vous avez des permissions d'écriture. Cette fonctionnalité est au paiement à l'usage : vous ne payez que pour les secondes que vous utilisez.

[`run_job`] vous permet d'exécuter n'importe quelle commande sur l'infrastructure de Hugging Face :

```python
# Exécuter directement du code Python
>>> from huggingface_hub import run_job
>>> run_job(
...     image="python:3.12",
...     command=["python", "-c", "print('Hello from the cloud!')"],
... )

# Utiliser des GPUs sans aucune configuration
>>> run_job(
...     image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
...     command=["python", "-c", "import torch; print(torch.cuda.get_device_name())"],
...     flavor="a10g-small",
... )

# Exécuter dans un compte d'organisation
>>> run_job(
...     image="python:3.12",
...     command=["python", "-c", "print('Running in an org account')"],
...     namespace="my-org-name",
... )

# Exécuter depuis Hugging Face Spaces
>>> run_job(
...     image="hf.co/spaces/lhoestq/duckdb",
...     command=["duckdb", "-c", "select 'hello world'"],
... )

# Exécuter un script Python avec `uv` (expérimental)
>>> from huggingface_hub import run_uv_job
>>> run_uv_job("my_script.py")
```

> [!WARNING]
> **Important** : Les Jobs ont un timeout par défaut (30 minutes), après quoi ils s'arrêteront automatiquement. Pour les tâches de longue durée comme l'entraînement de modèles, assurez-vous de définir un timeout personnalisé en utilisant le paramètre `timeout`. Voir [Configurer le Timeout du Job](#configurer-le-timeout-du-job) pour plus de détails.

[`run_job`] retourne le [`JobInfo`] qui a l'URL du Job sur Hugging Face, où vous pouvez voir le statut du Job et les logs. Sauvegardez l'ID du Job depuis [`JobInfo`] pour gérer le job :

```python
>>> from huggingface_hub import run_job
>>> job = run_job(
...     image="python:3.12",
...     command=["python", "-c", "print('Hello from the cloud!')"]
... )
>>> job.url
https://huggingface.co/jobs/lhoestq/687f911eaea852de79c4a50a
>>> job.id
687f911eaea852de79c4a50a
```

Les Jobs s'exécutent en arrière-plan. La section suivante vous guide à travers [`inspect_job`] pour connaître le statut d'un job et [`fetch_job_logs`] pour voir les logs.

## Vérifier le statut du Job

```python
# Lister vos jobs
>>> from huggingface_hub import list_jobs
>>> jobs = list_jobs()
>>> jobs[0]
JobInfo(id='687f911eaea852de79c4a50a', created_at=datetime.datetime(2025, 7, 22, 13, 24, 46, 909000, tzinfo=datetime.timezone.utc), docker_image='python:3.12', space_id=None, command=['python', '-c', "print('Hello from the cloud!')"], arguments=[], environment={}, secrets={}, flavor='cpu-basic', status=JobStatus(stage='COMPLETED', message=None), owner=JobOwner(id='5e9ecfc04957053f60648a3e', name='lhoestq'), endpoint='https://huggingface.co', url='https://huggingface.co/jobs/lhoestq/687f911eaea852de79c4a50a')

# Lister vos jobs en cours d'exécution
>>> running_jobs = [job for job in list_jobs() if job.status.stage == "RUNNING"]

# Inspecter le statut d'un job
>>> from huggingface_hub import inspect_job
>>> inspect_job(job_id=job_id)
JobInfo(id='687f911eaea852de79c4a50a', created_at=datetime.datetime(2025, 7, 22, 13, 24, 46, 909000, tzinfo=datetime.timezone.utc), docker_image='python:3.12', space_id=None, command=['python', '-c', "print('Hello from the cloud!')"], arguments=[], environment={}, secrets={}, flavor='cpu-basic', status=JobStatus(stage='COMPLETED', message=None), owner=JobOwner(id='5e9ecfc04957053f60648a3e', name='lhoestq'), endpoint='https://huggingface.co', url='https://huggingface.co/jobs/lhoestq/687f911eaea852de79c4a50a')

# Voir les logs d'un job
>>> from huggingface_hub import fetch_job_logs
>>> for log in fetch_job_logs(job_id=job_id):
...     print(log)
Hello from the cloud!

# Annuler un job
>>> from huggingface_hub import cancel_job
>>> cancel_job(job_id=job_id)
```

Vérifiez le statut de plusieurs jobs pour savoir quand ils sont tous terminés en utilisant une boucle et [`inspect_job`] :

```python
# Exécuter plusieurs jobs en parallèle et attendre leurs achèvements
>>> import time
>>> from huggingface_hub import inspect_job, run_job
>>> jobs = [run_job(image=image, command=command) for command in commands]
>>> for job in jobs:
...     while inspect_job(job_id=job.id).status.stage not in ("COMPLETED", "ERROR"):
...         time.sleep(10)
```

## Sélectionner le hardware

Il existe de nombreux cas où l'exécution de Jobs sur GPUs est utile :

- **Entraînement de modèles** : Finetuner ou entraîner des modèles sur GPUs (T4, A10G, A100) sans gérer l'infrastructure
- **Génération de données synthétiques** : Générer des datasets à grande échelle en utilisant des LLMs sur du hardware puissant
- **Traitement de données** : Traiter des datasets massifs avec des configurations haute-CPU pour des charges de travail parallèles
- **Inférence par batch** : Exécuter l'inférence hors ligne sur des milliers d'échantillons en utilisant des configurations GPU optimisées
- **Expériences & Benchmarks** : Exécuter des expériences ML sur du hardware cohérent pour des résultats reproductibles
- **Développement & Débogage** : Tester du code GPU sans configuration CUDA locale

Exécutez des jobs sur GPUs ou TPUs avec l'argument `flavor`. Par exemple, pour exécuter un job PyTorch sur un GPU A10G :

```python
# Utiliser un GPU A10G pour vérifier PyTorch CUDA
>>> from huggingface_hub import run_job
>>> run_job(
...     image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
...     command=["python", "-c", "import torch; print(f'This code ran with the following GPU: {torch.cuda.get_device_name()}')"],
...     flavor="a10g-small",
... )
```

L'exécution de ceci affichera la sortie suivante !

```bash
This code ran with the following GPU: NVIDIA A10G
```

Utilisez ceci pour exécuter un script de finetuning comme [trl/scripts/sft.py](https://github.com/huggingface/trl/blob/main/trl/scripts/sft.py) avec UV :

```python
>>> from huggingface_hub import run_uv_job
>>> run_uv_job(
...     "sft.py",
...     script_args=["--model_name_or_path", "Qwen/Qwen2-0.5B", ...],
...     dependencies=["trl"],
...     env={"HF_TOKEN": ...},
...     flavor="a10g-small",
... )
```

> [!TIP]
> Pour des conseils complets sur l'exécution de jobs d'entraînement de modèles avec TRL sur l'infrastructure Hugging Face, consultez la [documentation TRL Jobs Training](https://huggingface.co/docs/trl/main/en/jobs_training). Elle couvre les recettes de finetuning, la sélection de hardware et les meilleures pratiques pour entraîner des modèles efficacement.

Options `flavor` disponibles :

- CPU: `cpu-basic`, `cpu-upgrade`
- GPU: `t4-small`, `t4-medium`, `l4x1`, `l4x4`, `a10g-small`, `a10g-large`, `a10g-largex2`, `a10g-largex4`,`a100-large`
- TPU: `v5e-1x1`, `v5e-2x2`, `v5e-2x4`

(mis à jour en 07/2025 depuis la documentation Hugging Face [suggested_hardware docs](https://huggingface.co/docs/hub/en/spaces-config-reference))

C'est tout ! Vous exécutez maintenant du code sur l'infrastructure de Hugging Face.

## Configurer le Timeout du Job

Les Jobs ont un timeout par défaut (30 minutes), après quoi ils s'arrêteront automatiquement. C'est important à savoir lors de l'exécution de tâches de longue durée comme l'entraînement de modèles.

### Définir un timeout personnalisé

Vous pouvez spécifier une valeur de timeout personnalisée en utilisant le paramètre `timeout` lors de l'exécution d'un job. Le timeout peut être spécifié de deux façons :

1. **Comme un nombre** (interprété comme des secondes) :

```python
>>> from huggingface_hub import run_job
>>> job = run_job(
...     image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
...     command=["python", "train_model.py"],
...     flavor="a10g-large",
...     timeout=7200,  # 2 heures en secondes
... )
```

2. **Comme une chaîne avec des unités de temps** :

```python
>>> # Utilisation de différentes unités de temps
>>> job = run_job(
...     image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
...     command=["python", "train_model.py"],
...     flavor="a10g-large",
...     timeout="2h",  # 2 heures
... )

>>> # Autres exemples :
>>> # timeout="30m"    # 30 minutes
>>> # timeout="1.5h"   # 1.5 heures
>>> # timeout="1d"     # 1 jour
>>> # timeout="3600s"  # 3600 secondes
```

Unités de temps supportées :
- `s` - secondes
- `m` - minutes  
- `h` - heures
- `d` - jours

### Utiliser le timeout avec les jobs UV

Pour les jobs UV, vous pouvez également spécifier le timeout :

```python
>>> from huggingface_hub import run_uv_job
>>> job = run_uv_job(
...     "training_script.py",
...     flavor="a10g-large",
...     timeout="90m",  # 90 minutes
... )
```

> [!WARNING]
> Si vous ne spécifiez pas de timeout, un timeout par défaut sera appliqué à votre job. Pour les tâches de longue durée comme l'entraînement de modèles qui peuvent prendre des heures, assurez-vous de définir un timeout approprié pour éviter des arrêts inattendus de job. 

### Surveiller la durée du job

Lors de l'exécution de tâches longues, c'est une bonne pratique de :
- Estimer la durée attendue de votre job et définir un timeout avec une certaine marge
- Surveiller la progression de votre job via les logs
- Vérifier le statut du job pour s'assurer qu'il n'a pas expiré

```python
>>> from huggingface_hub import inspect_job, fetch_job_logs
>>> # Vérifier le statut du job
>>> job_info = inspect_job(job_id=job.id)
>>> if job_info.status.stage == "ERROR":
...     print(f"Job failed: {job_info.status.message}")
...     # Vérifier les logs pour plus de détails
...     for log in fetch_job_logs(job_id=job.id):
...         print(log)
```

Pour plus de détails sur le paramètre timeout, voir la [référence API `run_job`](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.run_job.timeout).

## Passer des variables d'environnement et Secrets

Vous pouvez passer des variables d'environnement à votre job en utilisant `env` et `secrets` :

```python
# Passer des variables d'environnement
>>> from huggingface_hub import run_job
>>> run_job(
...     image="python:3.12",
...     command=["python", "-c", "import os; print(os.environ['FOO'], os.environ['BAR'])"],
...     env={"FOO": "foo", "BAR": "bar"},
... )
```

```python
# Passer des secrets - ils seront chiffrés côté serveur
>>> from huggingface_hub import run_job
>>> run_job(
...     image="python:3.12",
...     command=["python", "-c", "import os; print(os.environ['MY_SECRET'])"],
...     secrets={"MY_SECRET": "psswrd"},
... )
```

### Scripts UV (Expérimental)

> [!TIP]
> Vous recherchez des scripts UV prêts à l'emploi ? Consultez l'[organisation uv-scripts](https://huggingface.co/uv-scripts) sur le Hugging Face Hub, qui offre une collection communautaire de scripts UV pour des tâches comme l'entraînement de modèles, la génération de données synthétiques, le traitement de données, et plus encore.

Exécutez des scripts UV (scripts Python avec dépendances inline) sur l'infrastructure HF :

```python
# Exécuter un script UV (crée un dépôt temporaire)
>>> from huggingface_hub import run_uv_job
>>> run_uv_job("my_script.py")

# Exécuter avec GPU
>>> run_uv_job("ml_training.py", flavor="gpu-t4-small")

# Exécuter avec des dépendances
>>> run_uv_job("inference.py", dependencies=["transformers", "torch"])

# Exécuter un script directement depuis une URL
>>> run_uv_job("https://huggingface.co/datasets/username/scripts/resolve/main/example.py")

# Exécuter une commande
>>> run_uv_job("python", script_args=["-c", "import lighteval"], dependencies=["lighteval"])
```

Les scripts UV sont des scripts Python qui incluent leurs dépendances directement dans le fichier en utilisant une syntaxe de commentaire spéciale. Cela les rend parfaits pour les tâches autonomes qui ne nécessitent pas de configurations de projet complexes. En savoir plus sur les scripts UV dans la [documentation UV](https://docs.astral.sh/uv/guides/scripts/).

#### Images Docker pour les Scripts UV

Bien que les scripts UV puissent spécifier leurs dépendances inline, les charges de travail ML ont souvent des dépendances complexes. L'utilisation d'images Docker pré-construites avec ces bibliothèques déjà installées peut considérablement accélérer le démarrage du job et éviter les problèmes de dépendances.

Par défaut, lorsque vous exécutez `hf jobs uv run`, l'image `astral-sh/uv:python3.12-bookworm` est utilisée. Cette image est basée sur la distribution Python 3.12 Bookworm avec uv préinstallé.

Vous pouvez spécifier une image différente en utilisant le flag `--image` :

```bash
hf jobs uv run \
 --flavor a10g-large \
 --image vllm/vllm-openai:latest \
...
```

La commande ci-dessus s'exécutera en utilisant l'image `vllm/vllm-openai:latest`. Cette approche pourrait être utile si vous utilisez vLLM pour la génération de données synthétiques.

> [!TIP]
> De nombreux frameworks d'inférence fournissent des images docker optimisées. Comme uv est de plus en plus adopté dans l'écosystème Python, davantage de ces images auront également uv préinstallé, ce qui signifie qu'elles fonctionneront lors de l'utilisation de hf jobs uv run.

### Jobs planifiés

Planifiez et gérez des jobs qui s'exécuteront sur l'infrastructure HF.

Utilisez [`create_scheduled_job`] ou [`create_scheduled_uv_job`] avec un planning de `@annually`, `@yearly`, `@monthly`, `@weekly`, `@daily`, `@hourly`, ou une expression de planning CRON (par exemple, `"0 9 * * 1"` pour 9h chaque lundi) :

```python
# Planifier un job qui s'exécute toutes les heures
>>> from huggingface_hub import create_scheduled_job
>>> create_scheduled_job(
...     image="python:3.12",
...     command=["python",  "-c", "print('This runs every hour!')"],
...     schedule="@hourly"
... )

# Utiliser la syntaxe CRON
>>> create_scheduled_job(
...     image="python:3.12",
...     command=["python",  "-c", "print('This runs every 5 minutes!')"],
...     schedule="*/5 * * * *"
... )

# Planifier avec GPU
>>> create_scheduled_job(
...     image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
...     command=["python",  "-c", 'import torch; print(f"This code ran with the following GPU: {torch.cuda.get_device_name()}")'],
...     schedule="@hourly",
...     flavor="a10g-small",
... )

# Planifier un script UV
>>> from huggingface_hub import create_scheduled_uv_job
>>> create_scheduled_uv_job("my_script.py", schedule="@hourly")
```

Utilisez les mêmes paramètres que [`run_job`] et [`run_uv_job`] pour passer des variables d'environnement, secrets, timeout, etc.

Gérez les jobs planifiés en utilisant [`list_scheduled_jobs`], [`inspect_scheduled_job`], [`suspend_scheduled_job`], [`resume_scheduled_job`], et [`delete_scheduled_job`] :

```python
# Lister vos jobs planifiés actifs
>>> from huggingface_hub import list_scheduled_jobs
>>> list_scheduled_jobs()

# Inspecter le statut d'un job
>>> from huggingface_hub import inspect_scheduled_job
>>> inspect_scheduled_job(scheduled_job_id)

# Suspendre (mettre en pause) un job planifié
>>> from huggingface_hub import suspend_scheduled_job
>>> suspend_scheduled_job(scheduled_job_id)

# Reprendre un job planifié
>>> from huggingface_hub import resume_scheduled_job
>>> resume_scheduled_job(scheduled_job_id)

# Supprimer un job planifié
>>> from huggingface_hub import delete_scheduled_job
>>> delete_scheduled_job(scheduled_job_id)
```

### Déclencher des Jobs avec des webhooks

Les webhooks vous permettent d'écouter les nouveaux changements sur des dépôts spécifiques ou sur tous les dépôts appartenant à un ensemble particulier d'utilisateurs/organisations (pas seulement vos dépôts, mais n'importe quel dépôt).

Utilisez [`create_webhook`] pour créer un webhook qui déclenche un Job lorsqu'un changement se produit dans un dépôt Hugging Face :

```python
from huggingface_hub import create_webhook

# Exemple : Créer un webhook qui déclenche un Job
webhook = create_webhook(
    job_id=job_id,
    watched=[{"type": "user", "name": "your-username"}, {"type": "org", "name": "your-org-name"}],
    domains=["repo", "discussion"],
    secret="your-secret"
)
```

Le webhook déclenche le Job avec le payload du webhook dans la variable d'environnement `WEBHOOK_PAYLOAD`. Vous pouvez trouver plus d'informations sur les webhooks dans la [documentation Webhooks](./webhooks).
