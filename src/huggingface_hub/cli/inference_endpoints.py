"""CLI commands for Hugging Face Inference Endpoints."""

from __future__ import annotations

import json
from typing import Annotated, Optional

import typer

from huggingface_hub._inference_endpoints import InferenceEndpoint
from huggingface_hub.errors import HfHubHTTPError

from ._cli_utils import TokenOpt, get_hf_api, typer_factory


app = typer_factory(help="Manage Hugging Face Inference Endpoints.")

NameArg = Annotated[
    str,
    typer.Argument(help="Endpoint name."),
]

RepoArg = Annotated[
    Optional[str],
    typer.Option(
        "--repo", help="The name of the model repository associated with the Inference Endpoint (e.g. 'gpt2')."
    ),
]

NamespaceOpt = Annotated[
    Optional[str],
    typer.Option(
        "--namespace",
        help="The namespace where the Inference Endpoint will be created. Defaults to the current user's namespace.",
    ),
]


FrameworkOpt = Annotated[
    Optional[str],
    typer.Option(
        "--framework",
        help="The machine learning framework used for the model (e.g. 'custom').",
    ),
]

AcceleratorOpt = Annotated[
    Optional[str],
    typer.Option(
        "--accelerator",
        help="The hardware accelerator to be used for inference (e.g. 'cpu').",
    ),
]

InstanceSizeOpt = Annotated[
    Optional[str],
    typer.Option(
        "--instance-size",
        help="The size or type of the instance to be used for hosting the model (e.g. 'x4').",
    ),
]

InstanceTypeOpt = Annotated[
    Optional[str],
    typer.Option(
        "--instance-type",
        help="The cloud instance type where the Inference Endpoint will be deployed (e.g. 'intel-icl').",
    ),
]

RegionOpt = Annotated[
    Optional[str],
    typer.Option(
        "--region",
        help="The cloud region in which the Inference Endpoint will be created (e.g. 'us-east-1').",
    ),
]

TaskOpt = Annotated[
    Optional[str],
    typer.Option(
        "--task",
        help="The task on which to deploy the model (e.g. 'text-classification').",
    ),
]
VendorOpt = Annotated[
    Optional[str],
    typer.Option(
        "--vendor",
        help="The cloud provider or vendor where the Inference Endpoint will be hosted (e.g. 'aws').",
    ),
]


def _print_endpoint(endpoint: InferenceEndpoint) -> None:
    typer.echo(json.dumps(endpoint.raw, indent=2, sort_keys=True))


@app.command(help="Lists all inference endpoints for the given namespace.")
def list(
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    api = get_hf_api(token=token)
    try:
        endpoints = api.list_inference_endpoints(namespace=namespace, token=token)
    except HfHubHTTPError as error:
        typer.echo(f"Listing failed: {error}")
        raise typer.Exit(code=error.response.status_code) from error

    typer.echo(
        json.dumps(
            {"items": [endpoint.raw for endpoint in endpoints]},
            indent=2,
            sort_keys=True,
        )
    )


deploy_app = typer_factory(help="Deploy Inference Endpoints from the Hub or the Catalog.")


@deploy_app.command(name="hub", help="Deploy an Inference Endpoint from a Hub repository.")
def deploy_from_hub(
    name: NameArg,
    repo: Annotated[
        str,
        typer.Option(
            "--repo",
            help="The name of the model repository associated with the Inference Endpoint (e.g. 'gpt2').",
        ),
    ],
    framework: Annotated[
        str,
        typer.Option(
            "--framework",
            help="The machine learning framework used for the model (e.g. 'custom').",
        ),
    ],
    accelerator: Annotated[
        str,
        typer.Option(
            "--accelerator",
            help="The hardware accelerator to be used for inference (e.g. 'cpu').",
        ),
    ],
    instance_size: Annotated[
        str,
        typer.Option(
            "--instance-size",
            help="The size or type of the instance to be used for hosting the model (e.g. 'x4').",
        ),
    ],
    instance_type: Annotated[
        str,
        typer.Option(
            "--instance-type",
            help="The cloud instance type where the Inference Endpoint will be deployed (e.g. 'intel-icl').",
        ),
    ],
    region: Annotated[
        str,
        typer.Option(
            "--region",
            help="The cloud region in which the Inference Endpoint will be created (e.g. 'us-east-1').",
        ),
    ],
    vendor: Annotated[
        str,
        typer.Option(
            "--vendor",
            help="The cloud provider or vendor where the Inference Endpoint will be hosted (e.g. 'aws').",
        ),
    ],
    *,
    namespace: NamespaceOpt = None,
    task: TaskOpt = None,
    token: TokenOpt = None,
) -> None:
    api = get_hf_api(token=token)
    try:
        endpoint = api.create_inference_endpoint(
            name=name,
            repository=repo,
            framework=framework,
            accelerator=accelerator,
            instance_size=instance_size,
            instance_type=instance_type,
            region=region,
            vendor=vendor,
            namespace=namespace,
            task=task,
            token=token,
        )
    except HfHubHTTPError as error:
        typer.echo(f"Deployment failed: {error}")
        raise typer.Exit(code=error.response.status_code) from error

    _print_endpoint(endpoint)


@deploy_app.command(name="catalog", help="Deploy an Inference Endpoint from the Model Catalog.")
def deploy_from_catalog(
    name: NameArg,
    repo: Annotated[
        str,
        typer.Option(
            "--repo",
            help="The name of the model repository associated with the Inference Endpoint (e.g. 'gpt2').",
        ),
    ],
    *,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    api = get_hf_api(token=token)
    try:
        endpoint = api.create_inference_endpoint_from_catalog(
            repo_id=repo,
            name=name,
            namespace=namespace,
            token=token,
        )
    except HfHubHTTPError as error:
        typer.echo(f"Deployment failed: {error}")
        raise typer.Exit(code=error.response.status_code) from error

    _print_endpoint(endpoint)


app.add_typer(deploy_app, name="deploy")


@app.command(help="Get information about an Inference Endpoint.")
def describe(
    name: NameArg,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    api = get_hf_api(token=token)
    try:
        endpoint = api.get_inference_endpoint(name=name, namespace=namespace, token=token)
    except HfHubHTTPError as error:
        typer.echo(f"Fetch failed: {error}")
        raise typer.Exit(code=error.response.status_code) from error

    _print_endpoint(endpoint)


@app.command(help="Update an existing endpoint.")
def update(
    endpoint_name: NameArg,
    repo: RepoArg = None,
    accelerator: AcceleratorOpt = None,
    instance_size: InstanceSizeOpt = None,
    instance_type: InstanceTypeOpt = None,
    framework: FrameworkOpt = None,
    revision: Annotated[
        Optional[str],
        typer.Option(
            help="The specific model revision to deploy on the Inference Endpoint (e.g. '6c0e6080953db56375760c0471a8c5f2929baf11').",
        ),
    ] = None,
    task: Annotated[
        Optional[str],
        typer.Option(
            help="The task on which to deploy the model (e.g. 'text-classification').",
        ),
    ] = None,
    min_replica: Annotated[
        Optional[int],
        typer.Option(
            help="The minimum number of replicas (instances) to keep running for the Inference Endpoint.",
        ),
    ] = None,
    max_replica: Annotated[
        Optional[int],
        typer.Option(
            help="The maximum number of replicas (instances) to scale to for the Inference Endpoint.",
        ),
    ] = None,
    scale_to_zero_timeout: Annotated[
        Optional[int],
        typer.Option(
            help="The duration in minutes before an inactive endpoint is scaled to zero.",
        ),
    ] = None,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    api = get_hf_api(token=token)
    try:
        endpoint = api.update_inference_endpoint(
            name=endpoint_name,
            namespace=namespace,
            repository=repo,
            framework=framework,
            revision=revision,
            task=task,
            accelerator=accelerator,
            instance_size=instance_size,
            instance_type=instance_type,
            min_replica=min_replica,
            max_replica=max_replica,
            scale_to_zero_timeout=scale_to_zero_timeout,
            token=token,
        )
    except HfHubHTTPError as error:
        typer.echo(f"Update failed: {error}")
        raise typer.Exit(code=error.response.status_code) from error
    _print_endpoint(endpoint)


@app.command(help="Delete an Inference Endpoint permanently.")
def delete(
    name: NameArg,
    namespace: NamespaceOpt = None,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            help="Skip confirmation prompts.",
        ),
    ] = False,
    token: TokenOpt = None,
) -> None:
    if not yes:
        confirmation = typer.prompt(f"Delete endpoint '{name}'? Type the name to confirm.")
        if confirmation != name:
            typer.echo("Aborted.")
            raise typer.Exit(code=2)

    api = get_hf_api(token=token)
    try:
        api.delete_inference_endpoint(name=name, namespace=namespace, token=token)
    except HfHubHTTPError as error:
        typer.echo(f"Delete failed: {error}")
        raise typer.Exit(code=error.response.status_code) from error

    typer.echo(f"Deleted '{name}'.")


@app.command(help="Pause an Inference Endpoint.")
def pause(
    name: NameArg,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    api = get_hf_api(token=token)
    try:
        endpoint = api.pause_inference_endpoint(name=name, namespace=namespace, token=token)
    except HfHubHTTPError as error:
        typer.echo(f"Pause failed: {error}")
        raise typer.Exit(code=error.response.status_code) from error

    _print_endpoint(endpoint)


@app.command(help="Resume an Inference Endpoint.")
def resume(
    name: NameArg,
    namespace: NamespaceOpt = None,
    running_ok: Annotated[
        bool,
        typer.Option(
            help="If `True`, the method will not raise an error if the Inference Endpoint is already running."
        ),
    ] = True,
    token: TokenOpt = None,
) -> None:
    api = get_hf_api(token=token)
    try:
        endpoint = api.resume_inference_endpoint(
            name=name,
            namespace=namespace,
            token=token,
            running_ok=running_ok,
        )
    except HfHubHTTPError as error:
        typer.echo(f"Resume failed: {error}")
        raise typer.Exit(code=error.response.status_code) from error
    _print_endpoint(endpoint)


@app.command(help="Scale an Inference Endpoint to zero.")
def scale_to_zero(
    name: NameArg,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    api = get_hf_api(token=token)
    try:
        endpoint = api.scale_to_zero_inference_endpoint(name=name, namespace=namespace, token=token)
    except HfHubHTTPError as error:
        typer.echo(f"Scale To Zero failed: {error}")
        raise typer.Exit(code=error.response.status_code) from error

    _print_endpoint(endpoint)


@app.command(help="List available Catalog models.")
def list_catalog(
    token: TokenOpt = None,
) -> None:
    api = get_hf_api(token=token)
    try:
        models = api.list_inference_catalog(token=token)
    except HfHubHTTPError as error:
        typer.echo(f"Catalog fetch failed: {error}")
        raise typer.Exit(code=error.response.status_code) from error

    typer.echo(json.dumps({"models": models}, indent=2, sort_keys=True))
