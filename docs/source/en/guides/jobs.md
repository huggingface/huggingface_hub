<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->
# Run and manage Jobs

The Hugging Face Hub provides compute for AI and data workflows via Jobs.
A job runs on Hugging Face infrastructure and are defined with a command to run (e.g. a python command), a Docker Image from Hugging Face Spaces or Docker Hub, and a hardware flavor (CPU, GPU, TPU). This guide will show you how to interact with Jobs on the Hub, especially:

- Run a job.
- Check job status.
- Select the hardware.
- Configure environment variables and secrets.
- Run UV scripts.

If you want to run and manage a job on the Hub, your machine must be logged in. If you are not, please refer to
[this section](../quick-start#authentication). In the rest of this guide, we will assume that your machine is logged in.

> [!TIP]
> **Hugging Face Jobs** are available only to [Pro users](https://huggingface.co/pro) and [Team or Enterprise organizations](https://huggingface.co/enterprise). Upgrade your plan to get started!

## Jobs Command Line Interface

Use the [`hf jobs` CLI](./cli#hf-jobs) to run Jobs from the command line, and pass `--flavor` to specify your hardware.

`hf jobs run` runs Jobs with a Docker image and a command with a familiar Docker-like interface. Think `docker run`, but for running code on any hardware:

```bash
>>> hf jobs run python:3.12 python -c "print('Hello world')"
>>> hf jobs run --flavor a10g-small pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel python -c "import torch; print(torch.cuda.get_device_name())"
```

Use `hf jobs uv run` to run local or remote UV scripts:

```bash
>>> hf jobs uv run my_script.py
>>> hf jobs uv run --flavor a10g-small "https://raw.githubusercontent.com/huggingface/trl/main/trl/scripts/sft.py" 
```

UV scripts are Python scripts that include their dependencies directly in the file using a special comment syntax defined in the [UV documentation](https://docs.astral.sh/uv/guides/scripts/).

Now the rest of this guide will show you the python API.
If you would like to view all the available `hf jobs` commands and options instead, check out the [guide on the `hf jobs` command line interface](./cli#hf-jobs).

## Run a Job

Run compute Jobs defined with a command and a Docker Image on Hugging Face infrastructure (including GPUs and TPUs).

You can only manage Jobs that you own (under your username namespace) or from organizations in which you have write permissions.
This feature is pay-as-you-go: you only pay for the seconds you use.

[`run_job`] lets you run any command on Hugging Face's infrastructure:

```python
# Directly run Python code
>>> from huggingface_hub import run_job
>>> run_job(
...     image="python:3.12",
...     command=["python", "-c", "print('Hello from the cloud!')"],
... )

# Use GPUs without any setup
>>> run_job(
...     image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
...     command=["python", "-c", "import torch; print(torch.cuda.get_device_name())"],
...     flavor="a10g-small",
... )

# Run in an organization account
>>> run_job(
...     image="python:3.12",
...     command=["python", "-c", "print('Running in an org account')"],
...     namespace="my-org-name",
... )

# Run from Hugging Face Spaces
>>> run_job(
...     image="hf.co/spaces/lhoestq/duckdb",
...     command=["duckdb", "-c", "select 'hello world'"],
... )

# Run a Python script with `uv` (experimental)
>>> from huggingface_hub import run_uv_job
>>> run_uv_job("my_script.py")
```

> [!WARNING]
> **Important**: Jobs have a default timeout (30 minutes), after which they will automatically stop. For long-running tasks like model training, make sure to set a custom timeout using the `timeout` parameter. See [Configure Job Timeout](#configure-job-timeout) for details.

[`run_job`] returns the [`JobInfo`] which has the URL of the Job on Hugging Face, where you can see the Job status and the logs.
Save the Job ID from [`JobInfo`] to manage the job:

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

Jobs run in the background. The next section guides you through [`inspect_job`] to know a jobs' status, [`fetch_job_logs`] to view the logs and [`fetch_job_metrics`] to monitor resources usage.

## Check Job status

```python
# List your jobs
>>> from huggingface_hub import list_jobs
>>> jobs = list_jobs()
>>> jobs[0]
JobInfo(id='687f911eaea852de79c4a50a', created_at=datetime.datetime(2025, 7, 22, 13, 24, 46, 909000, tzinfo=datetime.timezone.utc), docker_image='python:3.12', space_id=None, command=['python', '-c', "print('Hello from the cloud!')"], arguments=[], environment={}, secrets={}, flavor='cpu-basic', status=JobStatus(stage='COMPLETED', message=None), owner=JobOwner(id='5e9ecfc04957053f60648a3e', name='lhoestq'), endpoint='https://huggingface.co', url='https://huggingface.co/jobs/lhoestq/687f911eaea852de79c4a50a')

# List your running jobs
>>> running_jobs = [job for job in list_jobs() if job.status.stage == "RUNNING"]

# Inspect the status of a job
>>> from huggingface_hub import inspect_job
>>> inspect_job(job_id=job_id)
JobInfo(id='687f911eaea852de79c4a50a', created_at=datetime.datetime(2025, 7, 22, 13, 24, 46, 909000, tzinfo=datetime.timezone.utc), docker_image='python:3.12', space_id=None, command=['python', '-c', "print('Hello from the cloud!')"], arguments=[], environment={}, secrets={}, flavor='cpu-basic', status=JobStatus(stage='COMPLETED', message=None), owner=JobOwner(id='5e9ecfc04957053f60648a3e', name='lhoestq'), endpoint='https://huggingface.co', url='https://huggingface.co/jobs/lhoestq/687f911eaea852de79c4a50a')

# View logs from a job
>>> from huggingface_hub import fetch_job_logs
>>> for log in fetch_job_logs(job_id=job_id):
...     print(log)
Hello from the cloud!

# View resources usage metrics from a job
>>> from huggingface_hub import fetch_job_metrics
>>> for metrics in fetch_job_metrics(job_id=job_id):
...     print(metrics)
{
    "cpu_usage_pct": 0,
    "cpu_millicores": 2000,
    "memory_used_bytes": 929792,
    "memory_total_bytes": 17179869184,
    "rx_bps": 0,
    "tx_bps": 0,
    "gpus": {},
    "replica": "4dzsh"
}

# Cancel a job
>>> from huggingface_hub import cancel_job
>>> cancel_job(job_id=job_id)
```

Check the status of multiple jobs to know when they're all finished using a loop and [`inspect_job`]:

```python
# Run multiple jobs in parallel and wait for their completions
>>> import time
>>> from huggingface_hub import inspect_job, run_job
>>> jobs = [run_job(image=image, command=command) for command in commands]
>>> for job in jobs:
...     while inspect_job(job_id=job.id).status.stage not in ("COMPLETED", "ERROR"):
...         time.sleep(10)
```

## Select the hardware

There are numerous cases where running Jobs on GPUs are useful:

- **Model Training**: Fine-tune or train models on GPUs (T4, A10G, A100) without managing infrastructure
- **Synthetic Data Generation**: Generate large-scale datasets using LLMs on powerful hardware
- **Data Processing**: Process massive datasets with high-CPU configurations for parallel workloads
- **Batch Inference**: Run offline inference on thousands of samples using optimized GPU setups
- **Experiments & Benchmarks**: Run ML experiments on consistent hardware for reproducible results
- **Development & Debugging**: Test GPU code without local CUDA setup

Run jobs on GPUs or TPUs with the `flavor` argument. For example, to run a PyTorch job on an A10G GPU:

```python
# Use an A10G GPU to check PyTorch CUDA
>>> from huggingface_hub import run_job
>>> run_job(
...     image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
...     command=["python", "-c", "import torch; print(f'This code ran with the following GPU: {torch.cuda.get_device_name()}')"],
...     flavor="a10g-small",
... )
```

Running this will show the following output!

```bash
This code ran with the following GPU: NVIDIA A10G
```

Use this to run a fine-tuning script like [trl/scripts/sft.py](https://github.com/huggingface/trl/blob/main/trl/scripts/sft.py) with UV:

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
> For comprehensive guidance on running model training jobs with TRL on Hugging Face infrastructure, check out the [TRL Jobs Training documentation](https://huggingface.co/docs/trl/main/en/jobs_training). It covers fine-tuning recipes, hardware selection, and best practices for training models efficiently.

Available `flavor` options:

- CPU: `cpu-basic`, `cpu-upgrade`
- GPU: `t4-small`, `t4-medium`, `l4x1`, `l4x4`, `a10g-small`, `a10g-large`, `a10g-largex2`, `a10g-largex4`,`a100-large`
- TPU: `v5e-1x1`, `v5e-2x2`, `v5e-2x4`

(updated in 07/2025 from Hugging Face [suggested_hardware docs](https://huggingface.co/docs/hub/en/spaces-config-reference))

That's it! You're now running code on Hugging Face's infrastructure.

## Configure Job Timeout

Jobs have a default timeout (30 minutes), after which they will automatically stop. This is important to know when running long-running tasks like model training.

### Setting a custom timeout

You can specify a custom timeout value using the `timeout` parameter when running a job. The timeout can be specified in two ways:

1. **As a number** (interpreted as seconds):
```python
>>> from huggingface_hub import run_job
>>> job = run_job(
...     image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
...     command=["python", "train_model.py"],
...     flavor="a10g-large",
...     timeout=7200,  # 2 hours in seconds
... )
```

2. **As a string with time units**:
```python
>>> # Using different time units
>>> job = run_job(
...     image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
...     command=["python", "train_model.py"],
...     flavor="a10g-large",
...     timeout="2h",  # 2 hours
... )

>>> # Other examples:
>>> # timeout="30m"    # 30 minutes
>>> # timeout="1.5h"   # 1.5 hours
>>> # timeout="1d"     # 1 day
>>> # timeout="3600s"  # 3600 seconds
```

Supported time units:
- `s` - seconds
- `m` - minutes  
- `h` - hours
- `d` - days

### Using timeout with UV jobs

For UV jobs, you can also specify the timeout:

```python
>>> from huggingface_hub import run_uv_job
>>> job = run_uv_job(
...     "training_script.py",
...     flavor="a10g-large",
...     timeout="90m",  # 90 minutes
... )
```

> [!WARNING]
> If you don't specify a timeout, a default timeout will be applied to your job. For long-running tasks like model training that may take hours, make sure to set an appropriate timeout to avoid unexpected job terminations.

### Monitoring job duration

When running long tasks, it's good practice to:
- Estimate your job's expected duration and set a timeout with some buffer
- Monitor your job's progress through the logs
- Check the job status to ensure it hasn't timed out

```python
>>> from huggingface_hub import inspect_job, fetch_job_logs
>>> # Check job status
>>> job_info = inspect_job(job_id=job.id)
>>> if job_info.status.stage == "ERROR":
...     print(f"Job failed: {job_info.status.message}")
...     # Check logs for more details
...     for log in fetch_job_logs(job_id=job.id):
...         print(log)
```

For more details about the timeout parameter, see the [`run_job` API reference](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.run_job.timeout).

## Pass Environment variables and Secrets

You can pass environment variables to your job using `env` and `secrets`:

```python
# Pass environment variables
>>> from huggingface_hub import run_job
>>> run_job(
...     image="python:3.12",
...     command=["python", "-c", "import os; print(os.environ['FOO'], os.environ['BAR'])"],
...     env={"FOO": "foo", "BAR": "bar"},
... )
```


```python
# Pass secrets - they will be encrypted server side
>>> from huggingface_hub import run_job
>>> run_job(
...     image="python:3.12",
...     command=["python", "-c", "import os; print(os.environ['MY_SECRET'])"],
...     secrets={"MY_SECRET": "psswrd"},
... )
```


### UV Scripts (Experimental)

> [!TIP]
> Looking for ready-to-use UV scripts? Check out the [uv-scripts organization](https://huggingface.co/uv-scripts) on the Hugging Face Hub, which offers a community collection of UV scripts for tasks like model training, synthetic data generation, data processing, and more.

Run UV scripts (Python scripts with inline dependencies) on HF infrastructure:

```python
# Run a UV script (creates temporary repo)
>>> from huggingface_hub import run_uv_job
>>> run_uv_job("my_script.py")

# Run with GPU
>>> run_uv_job("ml_training.py", flavor="gpu-t4-small")

# Run with dependencies
>>> run_uv_job("inference.py", dependencies=["transformers", "torch"])

# Run a script directly from a URL
>>> run_uv_job("https://huggingface.co/datasets/username/scripts/resolve/main/example.py")

# Run a command
>>> run_uv_job("python", script_args=["-c", "import lighteval"], dependencies=["lighteval"])
```

UV scripts are Python scripts that include their dependencies directly in the file using a special comment syntax. This makes them perfect for self-contained tasks that don't require complex project setups. Learn more about UV scripts in the [UV documentation](https://docs.astral.sh/uv/guides/scripts/).


#### Docker Images for UV Scripts

While UV scripts can specify their dependencies inline, ML workloads often have complex dependencies. Using pre-built Docker images with these libraries already installed can significantly speed up job startup and avoid dependency issues.

By default, when you run `hf jobs uv run` the `astral-sh/uv:python3.12-bookworm` image is used. This image is based on the Python 3.12 Bookworm distribution with uv pre-installed.

You can specify a different image using the `--image` flag:

```bash
hf jobs uv run \
 --flavor a10g-large \
 --image vllm/vllm-openai:latest \
...
```

The above command will run using the `vllm/vllm-openai:latest` image. This approach could be useful if you are using vLLM for synthetic data generation.

> [!TIP]
> Many inference frameworks provide optimized docker images. As uv is increasingly adopted in the Python ecosystem more of these will also have uv pre-installed meaning they will work when using hf jobs uv run.

### Scheduled Jobs

Schedule and manage jobs that will run on HF infrastructure.

Use [`create_scheduled_job`] or [`create_scheduled_uv_job`] with a schedule of `@annually`, `@yearly`, `@monthly`, `@weekly`, `@daily`, `@hourly`, or a CRON schedule expression (e.g., `"0 9 * * 1"` for 9 AM every Monday):

```python
# Schedule a job that runs every hour
>>> from huggingface_hub import create_scheduled_job
>>> create_scheduled_job(
...     image="python:3.12",
...     command=["python",  "-c", "print('This runs every hour!')"],
...     schedule="@hourly"
... )

# Use the CRON syntax
>>> create_scheduled_job(
...     image="python:3.12",
...     command=["python",  "-c", "print('This runs every 5 minutes!')"],
...     schedule="*/5 * * * *"
... )

# Schedule with GPU
>>> create_scheduled_job(
...     image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
...     command=["python",  "-c", 'import torch; print(f"This code ran with the following GPU: {torch.cuda.get_device_name()}")'],
...     schedule="@hourly",
...     flavor="a10g-small",
... )

# Schedule a UV script
>>> from huggingface_hub import create_scheduled_uv_job
>>> create_scheduled_uv_job("my_script.py", schedule="@hourly")
```

Use the same parameters as [`run_job`] and [`run_uv_job`] to pass environment variables, secrets, timeout, etc.

Manage scheduled jobs using [`list_scheduled_jobs`], [`inspect_scheduled_job`], [`suspend_scheduled_job`], [`resume_scheduled_job`], and [`delete_scheduled_job`]:

```python
# List your active scheduled jobs
>>> from huggingface_hub import list_scheduled_jobs
>>> list_scheduled_jobs()

# Inspect the status of a job
>>> from huggingface_hub import inspect_scheduled_job
>>> inspect_scheduled_job(scheduled_job_id)

# Suspend (pause) a scheduled job
>>> from huggingface_hub import suspend_scheduled_job
>>> suspend_scheduled_job(scheduled_job_id)

# Resume a scheduled job
>>> from huggingface_hub import resume_scheduled_job
>>> resume_scheduled_job(scheduled_job_id)

# Delete a scheduled job
>>> from huggingface_hub import delete_scheduled_job
>>> delete_scheduled_job(scheduled_job_id)
```

### Trigger Jobs with webhooks

Webhooks allow you to listen for new changes on specific repos or to all repos belonging to particular set of users/organizations (not just your repos, but any repo).

Use [`create_webhook`] to create a webhook that triggers a Job when a change happens in a Hugging Face repository:

```python
from huggingface_hub import create_webhook

# Example: Creating a webhook that triggers a Job
webhook = create_webhook(
    job_id=job_id,
    watched=[{"type": "user", "name": "your-username"}, {"type": "org", "name": "your-org-name"}],
    domains=["repo", "discussion"],
    secret="your-secret"
)
```

The webhook triggers the Job with the webhook payload in the environment variable `WEBHOOK_PAYLOAD`.
You can find more information on webhooks in the [Webhooks documentation](./webhooks).
