<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Manage your Space

In this guide, we will see how to manage your Space runtime
([secrets](https://huggingface.co/docs/hub/spaces-overview#managing-secrets),
[hardware](https://huggingface.co/docs/hub/spaces-gpus), and [storage](https://huggingface.co/docs/hub/spaces-storage#persistent-storage)) using `huggingface_hub`.

## A simple example: configure secrets and hardware.

Here is an end-to-end example to create and setup a Space on the Hub.

**1. Create a Space on the Hub.**

```py
>>> from huggingface_hub import HfApi
>>> repo_id = "Wauplin/my-cool-training-space"
>>> api = HfApi()

# For example with a Gradio SDK
>>> api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio")
```

**1. (bis) Duplicate a Space.**

This can prove useful if you want to build up from an existing Space instead of starting from scratch.
It is also useful is you want control over the configuration/settings of a public Space. See [`duplicate_space`] for more details.

```py
>>> api.duplicate_space("multimodalart/dreambooth-training")
```

**2. Upload your code using your preferred solution.**

Here is an example to upload the local folder `src/` from your machine to your Space:

```py
>>> api.upload_folder(repo_id=repo_id, repo_type="space", folder_path="src/")
```

At this step, your app should already be running on the Hub for free !
However, you might want to configure it further with secrets and upgraded hardware.

**3. Configure secrets and variables**

Your Space might require some secret keys, token or variables to work.
See [docs](https://huggingface.co/docs/hub/spaces-overview#managing-secrets) for more details.
For example, an HF token to upload an image dataset to the Hub once generated from your Space.

```py
>>> api.add_space_secret(repo_id=repo_id, key="HF_TOKEN", value="hf_api_***")
>>> api.add_space_variable(repo_id=repo_id, key="MODEL_REPO_ID", value="user/repo")
```

Secrets and variables can be deleted as well:
```py
>>> api.delete_space_secret(repo_id=repo_id, key="HF_TOKEN")
>>> api.delete_space_variable(repo_id=repo_id, key="MODEL_REPO_ID")
```

<Tip>
From within your Space, secrets are available as environment variables (or
Streamlit Secrets Management if using Streamlit). No need to fetch them via the API!
</Tip>

<Tip warning={true}>
Any change in your Space configuration (secrets or hardware) will trigger a restart of your app.
</Tip>

**Bonus: set secrets and variables when creating or duplicating the Space!**

Secrets and variables can be set when creating or duplicating a space:

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

**4. Configure the hardware**

By default, your Space will run on a CPU environment for free. You can upgrade the hardware
to run it on GPUs. A payment card or a community grant is required to access upgrade your
Space. See [docs](https://huggingface.co/docs/hub/spaces-gpus) for more details.

```py
# Use `SpaceHardware` enum
>>> from huggingface_hub import SpaceHardware
>>> api.request_space_hardware(repo_id=repo_id, hardware=SpaceHardware.T4_MEDIUM)

# Or simply pass a string value
>>> api.request_space_hardware(repo_id=repo_id, hardware="t4-medium")
```

Hardware updates are not done immediately as your Space has to be reloaded on our servers.
At any time, you can check on which hardware your Space is running to see if your request
has been met.

```py
>>> runtime = api.get_space_runtime(repo_id=repo_id)
>>> runtime.stage
"RUNNING_BUILDING"
>>> runtime.hardware
"cpu-basic"
>>> runtime.requested_hardware
"t4-medium"
```

You now have a Space fully configured. Make sure to downgrade your Space back to "cpu-classic"
when you are done using it.

**Bonus: request hardware when creating or duplicating the Space!**

Upgraded hardware will be automatically assigned to your Space once it's built.

```py
>>> api.create_repo(
...     repo_id=repo_id,
...     repo_type="space",
...     space_sdk="gradio"
...     space_hardware="cpu-upgrade",
...     space_storage="small",
...     space_sleep_time="7200", # 2 hours in secs
... )
```
```py
>>> api.duplicate_space(
...     from_id=repo_id,
...     hardware="cpu-upgrade",
...     storage="small",
...     sleep_time="7200", # 2 hours in secs
... )
```

**5. Pause and restart your Space**

By default if your Space is running on an upgraded hardware, it will never be stopped. However to avoid getting billed,
you might want to pause it when you are not using it. This is possible using [`pause_space`]. A paused Space will be
inactive until the owner of the Space restarts it, either with the UI or via API using [`restart_space`].
For more details about paused mode, please refer to [this section](https://huggingface.co/docs/hub/spaces-gpus#pause)

```py
# Pause your Space to avoid getting billed
>>> api.pause_space(repo_id=repo_id)
# (...)
# Restart it when you need it
>>> api.restart_space(repo_id=repo_id)
```

Another possibility is to set a timeout for your Space. If your Space is inactive for more than the timeout duration,
it will go to sleep. Any visitor landing on your Space will start it back up. You can set a timeout using
[`set_space_sleep_time`]. For more details about sleeping mode, please refer to [this section](https://huggingface.co/docs/hub/spaces-gpus#sleep-time).

```py
# Put your Space to sleep after 1h of inactivity
>>> api.set_space_sleep_time(repo_id=repo_id, sleep_time=3600)
```

Note: if you are using a 'cpu-basic' hardware, you cannot configure a custom sleep time. Your Space will automatically
be paused after 48h of inactivity.

**Bonus: set a sleep time while requesting hardware**

Upgraded hardware will be automatically assigned to your Space once it's built.

```py
>>> api.request_space_hardware(repo_id=repo_id, hardware=SpaceHardware.T4_MEDIUM, sleep_time=3600)
```

**Bonus: set a sleep time when creating or duplicating the Space!**

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

**6. Add persistent storage to your Space**

You can choose the storage tier of your choice to access disk space that persists across restarts of your Space. This means you can read and write from disk like you would with a traditional hard drive. See [docs](https://huggingface.co/docs/hub/spaces-storage#persistent-storage) for more details.

```py
>>> from huggingface_hub import SpaceStorage
>>> api.request_space_storage(repo_id=repo_id, storage=SpaceStorage.LARGE)
```

You can also delete your storage, losing all the data permanently.
```py
>>> api.delete_space_storage(repo_id=repo_id)
```

Note: You cannot decrease the storage tier of your space once it's been granted. To do so,
you must delete the storage first then request the new desired tier.

**Bonus: request storage when creating or duplicating the Space!**

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

## More advanced: temporarily upgrade your Space !

Spaces allow for a lot of different use cases. Sometimes, you might want
to temporarily run a Space on a specific hardware, do something and then shut it down. In
this section, we will explore how to benefit from Spaces to finetune a model on demand.
This is only one way of solving this particular problem. It has to be taken as a suggestion
and adapted to your use case.

Let's assume we have a Space to finetune a model. It is a Gradio app that takes as input
a model id and a dataset id. The workflow is as follows:

0. (Prompt the user for a model and a dataset)
1. Load the model from the Hub.
2. Load the dataset from the Hub.
3. Finetune the model on the dataset.
4. Upload the new model to the Hub.

Step 3. requires a custom hardware but you don't want your Space to be running all the time on a paid
GPU. A solution is to dynamically request hardware for the training and shut it
down afterwards. Since requesting hardware restarts your Space, your app must somehow "remember"
the current task it is performing. There are multiple ways of doing this. In this guide
we will see one solution using a Dataset as "task scheduler".

### App skeleton

Here is what your app would look like. On startup, check if a task is scheduled and if yes,
run it on the correct hardware. Once done, set back hardware to the free-plan CPU and
prompt the user for a new task.

<Tip warning={true}>
Such a workflow does not support concurrent access as normal demos.
In particular, the interface will be disabled when training occurs.
It is preferable to set your repo as private to ensure you are the only user.
</Tip>

```py
# Space will need your token to request hardware: set it as a Secret !
HF_TOKEN = os.environ.get("HF_TOKEN")

# Space own repo_id
TRAINING_SPACE_ID = "Wauplin/dreambooth-training"

from huggingface_hub import HfApi, SpaceHardware
api = HfApi(token=HF_TOKEN)

# On Space startup, check if a task is scheduled. If yes, finetune the model. If not,
# display an interface to request a new task.
task = get_task()
if task is None:
    # Start Gradio app
    def gradio_fn(task):
        # On user request, add task and request hardware
        add_task(task)
        api.request_space_hardware(repo_id=TRAINING_SPACE_ID, hardware=SpaceHardware.T4_MEDIUM)

    gr.Interface(fn=gradio_fn, ...).launch()
else:
    runtime = api.get_space_runtime(repo_id=TRAINING_SPACE_ID)
    # Check if Space is loaded with a GPU.
    if runtime.hardware == SpaceHardware.T4_MEDIUM:
        # If yes, finetune base model on dataset !
        train_and_upload(task)

        # Then, mark the task as "DONE"
        mark_as_done(task)

        # DO NOT FORGET: set back CPU hardware
        api.request_space_hardware(repo_id=TRAINING_SPACE_ID, hardware=SpaceHardware.CPU_BASIC)
    else:
        api.request_space_hardware(repo_id=TRAINING_SPACE_ID, hardware=SpaceHardware.T4_MEDIUM)
```

### Task scheduler

Scheduling tasks can be done in many ways. Here is an example how it could be done using
a simple CSV stored as a Dataset.

```py
# Dataset ID in which a `tasks.csv` file contains the tasks to perform.
# Here is a basic example for `tasks.csv` containing inputs (base model and dataset)
# and status (PENDING or DONE).
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
        # Quick and dirty way to add a task
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
        # Quick and dirty way to set the task as DONE
        path_or_fileobj=tasks.replace(
            f"{model_id},{dataset_id},PENDING",
            f"{model_id},{dataset_id},DONE"
        ).encode()
    )
```
