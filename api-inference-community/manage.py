#!/usr/bin/env python
import argparse
import ast
import os
import subprocess
import uuid

from huggingface_hub import HfApi


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
    framework = None
    frameworks = {
        "allennlp",
        "asteroid",
        "espnet",
        "flair",
        "sentence-transformers",
        "spacy",
        "speechbrain",
        "timm",
    }
    for tag in info.tags:
        if tag in frameworks:
            if framework is None:
                framework = tag
            else:
                raise Exception(
                    "This model seems to implement 2 frameworks, we cannot infer"
                )
    return task, framework.replace("-", "_")


def start(args):
    import sys

    import uvicorn

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

    local_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "docker_images", framework
    )
    sys.path.append(local_path)
    os.environ["MODEL_ID"] = model_id
    os.environ["TASK"] = task
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, log_level="info")


def docker(args):
    model_id = args.model_id
    task = args.task
    framework = args.framework
    if task is None or framework is None:
        rtask, rframework = resolve(model_id)
        if task is None:
            task = rtask
        if framework is None:
            framework = rframework

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
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
