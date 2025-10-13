"""CLI commands for Hugging Face Inference Endpoints."""

import json
from typing import Annotated, Optional

import typer

from huggingface_hub._inference_endpoints import InferenceEndpoint
from huggingface_hub.errors import HfHubHTTPError

from ._cli_utils import TokenOpt, get_hf_api, typer_factory


app = typer_factory(help="Manage Hugging Face Inference Endpoints.")

catalog_app = typer_factory(help="Interact with the Inference Endpoints catalog.")

NameArg = Annotated[
    str,
    typer.Argument(help="Endpoint name."),
]

NamespaceOpt = Annotated[
    Optional[str],
    typer.Option(
        help="The namespace associated with the Inference Endpoint. Defaults to the current user's namespace.",
    ),
]


def _print_endpoint(endpoint: InferenceEndpoint) -> None:
    typer.echo(json.dumps(endpoint.raw, indent=2, sort_keys=True))


@app.command()
def ls(
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Lists all Inference Endpoints for the given namespace."""
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


@app.command(name="deploy", help="Deploy an Inference Endpoint from a Hub repository.")
def deploy(
    name: NameArg,
    repo: Annotated[
        str,
        typer.Option(
            help="The name of the model repository associated with the Inference Endpoint (e.g. 'openai/gpt-oss-120b').",
        ),
    ],
    framework: Annotated[
        str,
        typer.Option(
            help="The machine learning framework used for the model (e.g. 'vllm').",
        ),
    ],
    accelerator: Annotated[
        str,
        typer.Option(
            help="The hardware accelerator to be used for inference (e.g. 'cpu').",
        ),
    ],
    instance_size: Annotated[
        str,
        typer.Option(
            help="The size or type of the instance to be used for hosting the model (e.g. 'x4').",
        ),
    ],
    instance_type: Annotated[
        str,
        typer.Option(
            help="The cloud instance type where the Inference Endpoint will be deployed (e.g. 'intel-icl').",
        ),
    ],
    region: Annotated[
        str,
        typer.Option(
            help="The cloud region in which the Inference Endpoint will be created (e.g. 'us-east-1').",
        ),
    ],
    vendor: Annotated[
        str,
        typer.Option(
            help="The cloud provider or vendor where the Inference Endpoint will be hosted (e.g. 'aws').",
        ),
    ],
    *,
    namespace: NamespaceOpt = None,
    task: Annotated[
        Optional[str],
        typer.Option(
            help="The task on which to deploy the model (e.g. 'text-classification').",
        ),
    ] = None,
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


@catalog_app.command(name="deploy", help="Deploy an Inference Endpoint from the Model Catalog.")
def deploy_from_catalog(
    name: NameArg,
    repo: Annotated[
        str,
        typer.Option(
            help="The name of the model repository associated with the Inference Endpoint (e.g. 'openai/gpt-oss-120b').",
        ),
    ],
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


def list_catalog(
    token: TokenOpt = None,
) -> None:
    """List available Catalog models."""
    api = get_hf_api(token=token)
    try:
        models = api.list_inference_catalog(token=token)
    except HfHubHTTPError as error:
        typer.echo(f"Catalog fetch failed: {error}")
        raise typer.Exit(code=error.response.status_code) from error

    typer.echo(json.dumps({"models": models}, indent=2, sort_keys=True))


catalog_app.command(name="ls")(list_catalog)
app.command(name="list-catalog", help="List available Catalog models.", hidden=True)(list_catalog)


app.add_typer(catalog_app, name="catalog")


@app.command()
def describe(
    name: NameArg,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Get information about an existing endpoint."""
    api = get_hf_api(token=token)
    try:
        endpoint = api.get_inference_endpoint(name=name, namespace=namespace, token=token)
    except HfHubHTTPError as error:
        typer.echo(f"Fetch failed: {error}")
        raise typer.Exit(code=error.response.status_code) from error

    _print_endpoint(endpoint)


@app.command()
def update(
    name: NameArg,
    namespace: NamespaceOpt = None,
    repo: Annotated[
        Optional[str],
        typer.Option(
            help="The name of the model repository associated with the Inference Endpoint (e.g. 'openai/gpt-oss-120b').",
        ),
    ] = None,
    accelerator: Annotated[
        Optional[str],
        typer.Option(
            help="The hardware accelerator to be used for inference (e.g. 'cpu').",
        ),
    ] = None,
    instance_size: Annotated[
        Optional[str],
        typer.Option(
            help="The size or type of the instance to be used for hosting the model (e.g. 'x4').",
        ),
    ] = None,
    instance_type: Annotated[
        Optional[str],
        typer.Option(
            help="The cloud instance type where the Inference Endpoint will be deployed (e.g. 'intel-icl').",
        ),
    ] = None,
    framework: Annotated[
        Optional[str],
        typer.Option(
            help="The machine learning framework used for the model (e.g. 'custom').",
        ),
    ] = None,
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
    token: TokenOpt = None,
) -> None:
    """Update an existing endpoint."""
    api = get_hf_api(token=token)
    try:
        endpoint = api.update_inference_endpoint(
            name=name,
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


@app.command()
def delete(
    name: NameArg,
    namespace: NamespaceOpt = None,
    yes: Annotated[
        bool,
        typer.Option("--yes", help="Skip confirmation prompts."),
    ] = False,
    token: TokenOpt = None,
) -> None:
    """Delete an Inference Endpoint permanently."""
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


@app.command()
def pause(
    name: NameArg,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Pause an Inference Endpoint."""
    api = get_hf_api(token=token)
    try:
        endpoint = api.pause_inference_endpoint(name=name, namespace=namespace, token=token)
    except HfHubHTTPError as error:
        typer.echo(f"Pause failed: {error}")
        raise typer.Exit(code=error.response.status_code) from error

    _print_endpoint(endpoint)


@app.command()
def resume(
    name: NameArg,
    namespace: NamespaceOpt = None,
    fail_if_already_running: Annotated[
        bool,
        typer.Option(
            "--fail-if-already-running",
            help="If `True`, the method will raise an error if the Inference Endpoint is already running.",
        ),
    ] = False,
    token: TokenOpt = None,
) -> None:
    """Resume an Inference Endpoint."""
    api = get_hf_api(token=token)
    try:
        endpoint = api.resume_inference_endpoint(
            name=name,
            namespace=namespace,
            token=token,
            running_ok=not fail_if_already_running,
        )
    except HfHubHTTPError as error:
        typer.echo(f"Resume failed: {error}")
        raise typer.Exit(code=error.response.status_code) from error
    _print_endpoint(endpoint)


@app.command()
def scale_to_zero(
    name: NameArg,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Scale an Inference Endpoint to zero."""
    api = get_hf_api(token=token)
    try:
        endpoint = api.scale_to_zero_inference_endpoint(name=name, namespace=namespace, token=token)
    except HfHubHTTPError as error:
        typer.echo(f"Scale To Zero failed: {error}")
        raise typer.Exit(code=error.response.status_code) from error

    _print_endpoint(endpoint)
