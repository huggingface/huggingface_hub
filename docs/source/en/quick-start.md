<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Quickstart

The [Hugging Face Hub](https://huggingface.co/) is the go-to place for sharing machine learning
models, demos, datasets, and metrics. `huggingface_hub` library helps you interact with
the Hub without leaving your development environment. You can create and manage
repositories easily, download and upload files, and get useful model and dataset
metadata from the Hub.

## Installation

To get started, install the `huggingface_hub` library:

```bash
pip install --upgrade huggingface_hub
```

For more details, check out the [installation](installation) guide.

## Download files

Repositories on the Hub are git version controlled, and users can download a single file
or the whole repository. You can use the [`hf_hub_download`] function to download files.
This function will download and cache a file on your local disk. The next time you need
that file, it will load from your cache, so you don't need to re-download it.

You will need the repository id and the filename of the file you want to download. For
example, to download the [Pegasus](https://huggingface.co/google/pegasus-xsum) model
configuration file:

```py
>>> from huggingface_hub import hf_hub_download
>>> hf_hub_download(repo_id="google/pegasus-xsum", filename="config.json")
```

To download a specific version of the file, use the `revision` parameter to specify the
branch name, tag, or commit hash. If you choose to use the commit hash, it must be the
full-length hash instead of the shorter 7-character commit hash:

```py
>>> from huggingface_hub import hf_hub_download
>>> hf_hub_download(
...     repo_id="google/pegasus-xsum",
...     filename="config.json",
...     revision="4d33b01d79672f27f001f6abade33f22d993b151"
... )
```

For more details and options, see the API reference for [`hf_hub_download`].

<a id="login"></a> <!-- backward compatible anchor -->

## Authentication

In a lot of cases, you must be authenticated with a Hugging Face account to interact with
the Hub: download private repos, upload files, create PRs,...
[Create an account](https://huggingface.co/join) if you don't already have one, and then sign in
to get your [User Access Token](https://huggingface.co/docs/hub/security-tokens) from
your [Settings page](https://huggingface.co/settings/tokens). The User Access Token is
used to authenticate your identity to the Hub.

> [!TIP]
> Tokens can have `read` or `write` permissions. Make sure to have a `write` access token if you want to create or edit a repository. Otherwise, it's best to generate a `read` token to reduce risk in case your token is inadvertently leaked.

### Login command

The easiest way to authenticate is to save the token on your machine. You can do that from the terminal using the [`login`] command:

```bash
hf auth login
```

The command will tell you if you are already logged in and prompt you for your token. The token is then validated and saved in your `HF_HOME` directory (defaults to `~/.cache/huggingface/token`). Any script or library interacting with the Hub will use this token when sending requests.

Alternatively, you can programmatically login using [`login`] in a notebook or a script:

```py
>>> from huggingface_hub import login
>>> login()
```

You can only be logged in to one account at a time. Logging in to a new account will automatically log you out of the previous one. To determine your currently active account, simply run the `hf auth whoami` command.

> [!WARNING]
> Once logged in, all requests to the Hub - even methods that don't necessarily require authentication - will use your access token by default. If you want to disable the implicit use of your token, you should set `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` as an environment variable (see [reference](../package_reference/environment_variables#hfhubdisableimplicittoken)).

### Manage multiple tokens locally

You can save multiple tokens on your machine by simply logging in with the [`login`] command with each token. If you need to switch between these tokens locally, you can use the [`auth switch`] command:

```bash
hf auth switch
```

This command will prompt you to select a token by its name from a list of saved tokens. Once selected, the chosen token becomes the _active_ token, and it will be used for all interactions with the Hub.


You can list all available access tokens on your machine with `hf auth list`.

### Environment variable

The environment variable `HF_TOKEN` can also be used to authenticate yourself. This is especially useful in a Space where you can set `HF_TOKEN` as a [Space secret](https://huggingface.co/docs/hub/spaces-overview#managing-secrets).

> [!TIP]
> **NEW:** Google Colaboratory lets you define [private keys](https://twitter.com/GoogleColab/status/1719798406195867814) for your notebooks. Define a `HF_TOKEN` secret to be automatically authenticated!

Authentication via an environment variable or a secret has priority over the token stored on your machine.

### Method parameters

Finally, it is also possible to authenticate by passing your token to any method that accepts `token` as a parameter.

```
from huggingface_hub import whoami

user = whoami(token=...)
```

This is usually discouraged except in an environment where you don't want to store your token permanently or if you need to handle several tokens at once.

> [!WARNING]
> Please be careful when passing tokens as a parameter. It is always best practice to load the token from a secure vault instead of hardcoding it in your codebase or notebook. Hardcoded tokens present a major leak risk if you share your code inadvertently.

## Create a repository

Once you've registered and logged in, create a repository with the [`create_repo`]
function:

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.create_repo(repo_id="super-cool-model")
```

If you want your repository to be private, then:

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.create_repo(repo_id="super-cool-model", private=True)
```

Private repositories will not be visible to anyone except yourself.

> [!TIP]
> To create a repository or to push content to the Hub, you must provide a User Access
> Token that has the `write` permission. You can choose the permission when creating the
> token in your [Settings page](https://huggingface.co/settings/tokens).

## Upload files

Use the [`upload_file`] function to add a file to your newly created repository. You
need to specify:

1. The path of the file to upload.
2. The path of the file in the repository.
3. The repository id of where you want to add the file.

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.upload_file(
...     path_or_fileobj="/home/lysandre/dummy-test/README.md",
...     path_in_repo="README.md",
...     repo_id="lysandre/test-model",
... )
```

To upload more than one file at a time, take a look at the [Upload](./guides/upload) guide
which will introduce you to several methods for uploading files (with or without git).

## Next steps

The `huggingface_hub` library provides an easy way for users to interact with the Hub
with Python. To learn more about how you can manage your files and repositories on the
Hub, we recommend reading our [how-to guides](./guides/overview) to:

- [Manage your repository](./guides/repository).
- [Download](./guides/download) files from the Hub.
- [Upload](./guides/upload) files to the Hub.
- [Search the Hub](./guides/search) for your desired model or dataset.
- [Run Inference](./guides/inference) across multiple services for models hosted on the Hugging Face Hub.
