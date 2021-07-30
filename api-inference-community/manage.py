#!/usr/bin/env python
import argparse
import ast
import sys
import os
import hashlib
import subprocess
import uuid

from huggingface_hub import HfApi
from api_inference_community.batch import batch


class cd:
    """Context manager for changing the current working directory"""

    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


class DockerPopen(subprocess.Popen):
    def __exit__(self, exc_type, exc_val, traceback):
        self.terminate()
        self.wait(5)
        return super().__exit__(exc_type, exc_val, traceback)


def create_docker(name: str) -> str:
    rand = str(uuid.uuid4())[:5]
    tag = f"{name}:{rand}"
    with cd(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "docker_images", name)
    ):
        subprocess.run(["docker", "build", ".", "-t", tag])
    return tag


def resolve_dataset(args, task: str):
    import datasets

    builder = datasets.load_dataset_builder(
        args.dataset_name, use_auth_token=args.token
    )

    if args.dataset_config is None:
        args.dataset_config = builder.config_id
        print(f"Inferred dataset_config {args.dataset_config}")

    splits = builder.info.splits
    if splits is not None:
        if args.dataset_split not in splits:
            raise ValueError(
                f"The split `{args.dataset_split}` is not a valid split, please choose from {','.join(splits.keys())}"
            )

    task_templates = builder.info.task_templates
    if task_templates is not None:
        for task_template in task_templates:
            if task_template.task == task:
                args.dataset_column = task_template.audio_file_path_column
                print(f"Inferred dataset_column {args.dataset_column}")
    return (
        args.dataset_name,
        args.dataset_config,
        args.dataset_split,
        args.dataset_column,
    )


def get_repo_name(model_id: str, dataset_name: str) -> str:
    # Hash needs to have the fully qualified name to disambiguate.
    hash_ = hashlib.md5((model_id + dataset_name).encode("utf-8")).hexdigest()

    model_name = model_id.split("/")[-1]
    dataset_name = dataset_name.split("/")[-1]
    return f"bulk-{model_name[:10]}-{dataset_name[:10]}-{hash_[:5]}"


def do_batch(args):
    model_id, task, framework = resolve_task_framework(args)
    dataset_name, dataset_config, dataset_split, dataset_column = resolve_dataset(
        args, task
    )
    repo_name = get_repo_name(model_id, dataset_name)

    local_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "docker_images", framework
    )
    sys.path.append(local_path)
    os.environ["MODEL_ID"] = model_id
    os.environ["TASK"] = task

    from app.main import get_pipeline

    pipeline = get_pipeline()

    batch(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
        dataset_column=dataset_column,
        token=args.token,
        repo_id=repo_name,
        use_gpu=False,
        pipeline=pipeline,
    )


def show(args):
    directory = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "docker_images"
    )
    for framework in sorted(os.listdir(directory)):
        print(f"{framework}")
        local_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "docker_images",
            framework,
            "app",
            "main.py",
        )
        # Using ast to prevent import issues with missing dependencies.
        # and slow loads.
        with open(local_path, "r") as source:
            tree = ast.parse(source.read())
            for item in tree.body:
                if (
                    isinstance(item, ast.AnnAssign)
                    and item.target.id == "ALLOWED_TASKS"
                ):
                    for key in item.value.keys:
                        print(" " * 4, key.value)


def resolve(model_id: str) -> [str, str]:
    info = HfApi().model_info(model_id)
    task = info.pipeline_tag
    framework = info.library_name
    return task, framework.replace("-", "_")


def resolve_task_framework(args):
    model_id = args.model_id
    task = args.task
    framework = args.framework
    if task is None or framework is None:
        rtask, rframework = resolve(model_id)
        if task is None:
            task = rtask
            print(f"Inferred task : {task}")
        if framework is None:
            framework = rframework
            print(f"Inferred framework : {framework}")
    return model_id, task, framework


def start(args):
    import uvicorn

    model_id, task, framework = resolve_task_framework(args)

    local_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "docker_images", framework
    )
    sys.path.append(local_path)
    os.environ["MODEL_ID"] = model_id
    os.environ["TASK"] = task
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, log_level="info")


def docker(args):
    model_id, task, framework = resolve_task_framework(args)

    tag = create_docker(framework)
    run_docker_command = [
        "docker",
        "run",
        "-p",
        "8000:80",
        "-e",
        f"TASK={task}",
        "-e",
        f"MODEL_ID={model_id}",
        "-v",
        "/tmp:/data",
        "-t",
        tag,
    ]

    with DockerPopen(run_docker_command) as proc:
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    parser_start = subparsers.add_parser(
        "start", help="Start a local version of a model inference"
    )
    parser_start.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Which model_id to start.",
    )
    parser_start.add_argument(
        "--task",
        type=str,
        help="Which task to load",
    )
    parser_start.add_argument(
        "--framework",
        type=str,
        help="Which framework to load",
    )
    parser_start.set_defaults(func=start)
    parser_docker = subparsers.add_parser(
        "docker", help="Start a docker version of a model inference"
    )
    parser_docker.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Which model_id to docker.",
    )
    parser_docker.add_argument(
        "--task",
        type=str,
        help="Which task to load",
    )
    parser_docker.add_argument(
        "--framework",
        type=str,
        help="Which framework to load",
    )
    parser_docker.set_defaults(func=docker)
    parser_show = subparsers.add_parser(
        "show", help="Show dockers and the various pipelines they implement"
    )
    parser_show.set_defaults(func=show)
    parser_batch = subparsers.add_parser("batch", help="Run a batch job")
    parser_batch.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Which model_id to batch.",
    )
    parser_batch.add_argument(
        "--task",
        type=str,
        help="Which task to load",
    )
    parser_batch.add_argument(
        "--framework",
        type=str,
        help="Which framework to load",
    )
    parser_batch.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Which dataset_name to batch.",
    )
    parser_batch.add_argument(
        "--dataset-config",
        type=str,
        help="Which dataset_config to batch.",
    )
    parser_batch.add_argument(
        "--dataset-split",
        type=str,
        required=True,
        help="Which dataset_split to batch.",
    )
    parser_batch.add_argument(
        "--dataset-column",
        type=str,
        help="Which column to batch.",
    )
    parser_batch.add_argument(
        "--token",
        type=str,
        help="Your API token",
    )
    parser_batch.set_defaults(func=do_batch)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
