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

<Tip>

Use [huggingface-cli jobs](./cli#huggingface-cli-jobs) to run jobs in the command line.

</Tip>

[`run_job`] returns the `JobInfo` which has the URL of the Job on Hugging Face, where you can see the Job status and the logs.
Save the Job ID from `JobInfo` to manage the job:

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

Jobs run in the background. The next section guides you through [`inspect_job`] to know a jobs' status and [`fetch_job_logs`] to the view the logs.

## Check Job status

```python
# List your jobs
>>> from huggingface_hub import list_jobs
>>> jobs = list_jobs()
>>> jobs[0]
JobInfo(id='687f911eaea852de79c4a50a', created_at=datetime.datetime(2025, 7, 22, 13, 24, 46, 909000, tzinfo=datetime.timezone.utc), docker_image='python:3.12', space_id=None, command=['python', '-c', "print('Hello from the cloud!')"], arguments=[], environment={}, secrets={}, flavor='cpu-basic', status=JobStatus(stage='COMPLETED', message=None), owner=JobOwner(id='5e9ecfc04957053f60648a3e', name='lhoestq'), endpoint='https://huggingface.co', url='https://huggingface.co/jobs/lhoestq/687f911eaea852de79c4a50a')

# List your running jobs
>>> running_jobs = [job for job in jobs if job.status.stage == "RUNNING"]

# Inspect the status of a job
>>> from huggingface_hub import inspect_job
>>> inspect_job(job_id=job_id)
JobInfo(id='687f911eaea852de79c4a50a', created_at=datetime.datetime(2025, 7, 22, 13, 24, 46, 909000, tzinfo=datetime.timezone.utc), docker_image='python:3.12', space_id=None, command=['python', '-c', "print('Hello from the cloud!')"], arguments=[], environment={}, secrets={}, flavor='cpu-basic', status=JobStatus(stage='COMPLETED', message=None), owner=JobOwner(id='5e9ecfc04957053f60648a3e', name='lhoestq'), endpoint='https://huggingface.co', url='https://huggingface.co/jobs/lhoestq/687f911eaea852de79c4a50a')

# View logs from a job
>>> from huggingface_hub import fetch_job_logs
>>> for log in fetch_job_logs(job_id=job_id):
...     print(log)
Hello from the cloud!

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
...     flavor="a10g-small"
... )
```

Running this will show the following output!

```bash
This code ran with the following GPU: NVIDIA A10G
```

Use this to run a fine tuning script like [trl/scripts/sft.py](https://github.com/huggingface/trl/blob/main/trl/scripts/sft.py) with UV:

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

Available `flavor` options:

- CPU: `cpu-basic`, `cpu-upgrade`
- GPU: `t4-small`, `t4-medium`, `l4x1`, `l4x4`, `a10g-small`, `a10g-large`, `a10g-largex2`, `a10g-largex4`,`a100-large`
- TPU: `v5e-1x1`, `v5e-2x2`, `v5e-2x4`

(updated in 07/2025 from Hugging Face [suggested_hardware docs](https://huggingface.co/docs/hub/en/spaces-config-reference))

That's it! You're now running code on Hugging Face's infrastructure. For more detailed information checkout the [Quickstart Guide](docs/quickstart.md).

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

Run UV scripts (Python scripts with inline dependencies) on HF infrastructure:

```python
# Run a UV script (creates temporary repo)
>>> from huggingface_hub import run_uv_job
>>> run_uv_job("my_script.py")

# Run with GPU
>>> run_uv_job("ml_training.py", flavor="gpu-t4-small")

# Run a script directly from a URL
>>> run_uv_job("https://huggingface.co/datasets/username/scripts/resolve/main/example.py")
```

UV scripts are Python scripts that include their dependencies directly in the file using a special comment syntax. This makes them perfect for self-contained tasks that don't require complex project setups. Learn more about UV scripts in the [UV documentation](https://docs.astral.sh/uv/guides/scripts/).
