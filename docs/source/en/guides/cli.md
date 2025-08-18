<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Command Line Interface (CLI)

The `huggingface_hub` Python package comes with a built-in CLI called `hf`. This tool allows you to interact with the Hugging Face Hub directly from a terminal. For example, you can login to your account, create a repository, upload and download files, etc. It also comes with handy features to configure your machine or manage your cache. In this guide, we will have a look at the main features of the CLI and how to use them.

## Getting started

First of all, let's install the CLI:

```
>>> pip install -U "huggingface_hub[cli]"
```

<Tip>

In the snippet above, we also installed the `[cli]` extra dependencies to make the user experience better, especially when using the `cache delete` command.

</Tip>

Once installed, you can check that the CLI is correctly setup:

```
>>> hf --help
usage: hf <command> [<args>]

positional arguments:
  {auth,cache,download,repo,repo-files,upload,upload-large-folder,env,version,lfs-enable-largefiles,lfs-multipart-upload}
                        hf command helpers
    auth                Manage authentication (login, logout, etc.).
    cache               Manage local cache directory.
    download            Download files from the Hub
    repo                Manage repos on the Hub.
    repo-files          Manage files in a repo on the Hub.
    upload              Upload a file or a folder to the Hub. Recommended for single-commit uploads.
    upload-large-folder
                        Upload a large folder to the Hub. Recommended for resumable uploads.
    env                 Print information about the environment.
    version             Print information about the hf version.

options:
  -h, --help            show this help message and exit
```

If the CLI is correctly installed, you should see a list of all the options available in the CLI. If you get an error message such as `command not found: hf`, please refer to the [Installation](../installation) guide.

<Tip>

The `--help` option is very convenient for getting more details about a command. You can use it anytime to list all available options and their details. For example, `hf upload --help` provides more information on how to upload files using the CLI.

</Tip>

### Alternative install

#### Using pkgx

[Pkgx](https://pkgx.sh)  is a blazingly fast cross platform package manager that runs anything. You can install huggingface-cli using pkgx as follows:

```bash
>>> pkgx install huggingface-cli
```

Or you can run huggingface-cli directly:

```bash
>>> pkgx huggingface-cli --help
```

Check out the pkgx huggingface page [here](https://pkgx.dev/pkgs/huggingface.co/) for more details.

#### Using Homebrew

You can also install the CLI using [Homebrew](https://brew.sh/):

```bash
>>> brew install huggingface-cli
```

Check out the Homebrew huggingface page [here](https://formulae.brew.sh/formula/huggingface-cli) for more details.

## hf auth login

In many cases, you must be logged in to a Hugging Face account to interact with the Hub (download private repos, upload files, create PRs, etc.). To do so, you need a [User Access Token](https://huggingface.co/docs/hub/security-tokens) from your [Settings page](https://huggingface.co/settings/tokens). The User Access Token is used to authenticate your identity to the Hub. Make sure to set a token with write access if you want to upload or modify content.

Once you have your token, run the following command in your terminal:

```bash
>>> hf auth login
```

This command will prompt you for a token. Copy-paste yours and press *Enter*. Then, you'll be asked if the token should also be saved as a git credential. Press *Enter* again (default to yes) if you plan to use `git` locally. Finally, it will call the Hub to check that your token is valid and save it locally.

```
_|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
_|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
_|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
_|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
_|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|

To log in, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens .
Enter your token (input will not be visible):
Add token as git credential? (Y/n)
Token is valid (permission: write).
Your token has been saved in your configured git credential helpers (store).
Your token has been saved to /home/wauplin/.cache/huggingface/token
Login successful
```

Alternatively, if you want to log-in without being prompted, you can pass the token directly from the command line. To be more secure, we recommend passing your token as an environment variable to avoid pasting it in your command history.

```bash
# Or using an environment variable
>>> hf auth login --token $HF_TOKEN --add-to-git-credential
Token is valid (permission: write).
The token `token_name` has been saved to /home/wauplin/.cache/huggingface/stored_tokens
Your token has been saved in your configured git credential helpers (store).
Your token has been saved to /home/wauplin/.cache/huggingface/token
Login successful
The current active token is: `token_name`
```

For more details about authentication, check out [this section](../quick-start#authentication).

## hf auth whoami

If you want to know if you are logged in, you can use `hf auth whoami`. This command doesn't have any options and simply prints your username and the organizations you are a part of on the Hub:

```bash
hf auth whoami
Wauplin
orgs:  huggingface,eu-test,OAuthTesters,hf-accelerate,HFSmolCluster
```

If you are not logged in, an error message will be printed.

## hf auth logout

This command logs you out. In practice, it will delete all tokens stored on your machine. If you want to remove a specific token, you can specify the token name as an argument.

This command will not log you out if you are logged in using the `HF_TOKEN` environment variable (see [reference](../package_reference/environment_variables#hftoken)). If that is the case, you must unset the environment variable in your machine configuration.

## hf download


Use the `hf download` command to download files from the Hub directly. Internally, it uses the same [`hf_hub_download`] and [`snapshot_download`] helpers described in the [Download](./download) guide and prints the returned path to the terminal. In the examples below, we will walk through the most common use cases. For a full list of available options, you can run:

```bash
hf download --help
```

### Download a single file

To download a single file from a repo, simply provide the repo_id and filename as follow:

```bash
>>> hf download gpt2 config.json
downloading https://huggingface.co/gpt2/resolve/main/config.json to /home/wauplin/.cache/huggingface/hub/tmpwrq8dm5o
(…)ingface.co/gpt2/resolve/main/config.json: 100%|██████████████████████████████████| 665/665 [00:00<00:00, 2.49MB/s]
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10/config.json
```

The command will always print on the last line the path to the file on your local machine.

To download a file located in a subdirectory of the repo, you should provide the path of the file in the repo in posix format like this:

```bash
>>> hf download HiDream-ai/HiDream-I1-Full text_encoder/model.safetensors
```

### Download an entire repository

In some cases, you just want to download all the files from a repository. This can be done by just specifying the repo id:

```bash
>>> hf download HuggingFaceH4/zephyr-7b-beta
Fetching 23 files:   0%|                                                | 0/23 [00:00<?, ?it/s]
...
...
/home/wauplin/.cache/huggingface/hub/models--HuggingFaceH4--zephyr-7b-beta/snapshots/3bac358730f8806e5c3dc7c7e19eb36e045bf720
```

### Download multiple files

You can also download a subset of the files from a repository with a single command. This can be done in two ways. If you already have a precise list of the files you want to download, you can simply provide them sequentially:

```bash
>>> hf download gpt2 config.json model.safetensors
Fetching 2 files:   0%|                                                                        | 0/2 [00:00<?, ?it/s]
downloading https://huggingface.co/gpt2/resolve/11c5a3d5811f50298f278a704980280950aedb10/model.safetensors to /home/wauplin/.cache/huggingface/hub/tmpdachpl3o
(…)8f278a7049802950aedb10/model.safetensors: 100%|██████████████████████████████| 8.09k/8.09k [00:00<00:00, 40.5MB/s]
Fetching 2 files: 100%|████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  3.76it/s]
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10
```

The other approach is to provide patterns to filter which files you want to download using `--include` and `--exclude`. For example, if you want to download all safetensors files from [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), except the files in FP16 precision:

```bash
>>> hf download stabilityai/stable-diffusion-xl-base-1.0 --include "*.safetensors" --exclude "*.fp16.*"*
Fetching 8 files:   0%|                                                                         | 0/8 [00:00<?, ?it/s]
...
...
Fetching 8 files: 100%|█████████████████████████████████████████████████████████████████████████| 8/8 (...)
/home/wauplin/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b
```

### Download a dataset or a Space

The examples above show how to download from a model repository. To download a dataset or a Space, use the `--repo-type` option:

```bash
# https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
>>> hf download HuggingFaceH4/ultrachat_200k --repo-type dataset

# https://huggingface.co/spaces/HuggingFaceH4/zephyr-chat
>>> hf download HuggingFaceH4/zephyr-chat --repo-type space

...
```

### Download a specific revision

The examples above show how to download from the latest commit on the main branch. To download from a specific revision (commit hash, branch name or tag), use the `--revision` option:

```bash
>>> hf download bigcode/the-stack --repo-type dataset --revision v1.1
...
```

### Download to a local folder

The recommended (and default) way to download files from the Hub is to use the cache-system. However, in some cases you want to download files and move them to a specific folder. This is useful to get a workflow closer to what git commands offer. You can do that using the `--local-dir` option.

A `.cache/huggingface/` folder is created at the root of your local directory containing metadata about the downloaded files. This prevents re-downloading files if they're already up-to-date. If the metadata has changed, then the new file version is downloaded. This makes the `local-dir` optimized for pulling only the latest changes.

<Tip>

For more details on how downloading to a local file works, check out the [download](./download#download-files-to-a-local-folder) guide.

</Tip>

```bash
>>> hf download adept/fuyu-8b model-00001-of-00002.safetensors --local-dir fuyu
...
fuyu/model-00001-of-00002.safetensors
```

### Specify cache directory

If not using `--local-dir`, all files will be downloaded by default to the cache directory defined by the `HF_HOME` [environment variable](../package_reference/environment_variables#hfhome). You can specify a custom cache using `--cache-dir`:

```bash
>>> hf download adept/fuyu-8b --cache-dir ./path/to/cache
...
./path/to/cache/models--adept--fuyu-8b/snapshots/ddcacbcf5fdf9cc59ff01f6be6d6662624d9c745
```

### Specify a token

To access private or gated repositories, you must use a token. By default, the token saved locally (using `hf auth login`) will be used. If you want to authenticate explicitly, use the `--token` option:

```bash
>>> hf download gpt2 config.json --token=hf_****
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10/config.json
```

### Quiet mode

By default, the `hf download` command will be verbose. It will print details such as warning messages, information about the downloaded files, and progress bars. If you want to silence all of this, use the `--quiet` option. Only the last line (i.e. the path to the downloaded files) is printed. This can prove useful if you want to pass the output to another command in a script.

```bash
>>> hf download gpt2 --quiet
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10
```

### Download timeout

On machines with slow connections, you might encounter timeout issues like this one:
```bash
`requests.exceptions.ReadTimeout: (ReadTimeoutError("HTTPSConnectionPool(host='cdn-lfs-us-1.huggingface.co', port=443): Read timed out. (read timeout=10)"), '(Request ID: a33d910c-84c6-4514-8362-c705e2039d38)')`
```

To mitigate this issue, you can set the `HF_HUB_DOWNLOAD_TIMEOUT` environment variable to a higher value (default is 10):
```bash
export HF_HUB_DOWNLOAD_TIMEOUT=30
```

For more details, check out the [environment variables reference](../package_reference/environment_variables#hfhubdownloadtimeout). And rerun your download command.

## hf upload

Use the `hf upload` command to upload files to the Hub directly. Internally, it uses the same [`upload_file`] and [`upload_folder`] helpers described in the [Upload](./upload) guide. In the examples below, we will walk through the most common use cases. For a full list of available options, you can run:

```bash
>>> hf upload --help
```

### Upload an entire folder

The default usage for this command is:

```bash
# Usage:  hf upload [repo_id] [local_path] [path_in_repo]
```

To upload the current directory at the root of the repo, use:

```bash
>>> hf upload my-cool-model . .
https://huggingface.co/Wauplin/my-cool-model/tree/main/
```

<Tip>

If the repo doesn't exist yet, it will be created automatically.

</Tip>

You can also upload a specific folder:

```bash
>>> hf upload my-cool-model ./models .
https://huggingface.co/Wauplin/my-cool-model/tree/main/
```

Finally, you can upload a folder to a specific destination on the repo:

```bash
>>> hf upload my-cool-model ./path/to/curated/data /data/train
https://huggingface.co/Wauplin/my-cool-model/tree/main/data/train
```

### Upload a single file

You can also upload a single file by setting `local_path` to point to a file on your machine. If that's the case, `path_in_repo` is optional and will default to the name of your local file:

```bash
>>> hf upload Wauplin/my-cool-model ./models/model.safetensors
https://huggingface.co/Wauplin/my-cool-model/blob/main/model.safetensors
```

If you want to upload a single file to a specific directory, set `path_in_repo` accordingly:

```bash
>>> hf upload Wauplin/my-cool-model ./models/model.safetensors /vae/model.safetensors
https://huggingface.co/Wauplin/my-cool-model/blob/main/vae/model.safetensors
```

### Upload multiple files

To upload multiple files from a folder at once without uploading the entire folder, use the `--include` and `--exclude` patterns. It can also be combined with the `--delete` option to delete files on the repo while uploading new ones. In the example below, we sync the local Space by deleting remote files and uploading all files except the ones in `/logs`:

```bash
# Sync local Space with Hub (upload new files except from logs/, delete removed files)
>>> hf upload Wauplin/space-example --repo-type=space --exclude="/logs/*" --delete="*" --commit-message="Sync local Space with Hub"
...
```

### Upload to a dataset or Space

To upload to a dataset or a Space, use the `--repo-type` option:

```bash
>>> hf upload Wauplin/my-cool-dataset ./data /train --repo-type=dataset
...
```

### Upload to an organization

To upload content to a repo owned by an organization instead of a personal repo, you must explicitly specify it in the `repo_id`:

```bash
>>> hf upload MyCoolOrganization/my-cool-model . .
https://huggingface.co/MyCoolOrganization/my-cool-model/tree/main/
```

### Upload to a specific revision

By default, files are uploaded to the `main` branch. If you want to upload files to another branch or reference, use the `--revision` option:

```bash
# Upload files to a PR
>>> hf upload bigcode/the-stack . . --repo-type dataset --revision refs/pr/104
...
```

**Note:** if `revision` does not exist and `--create-pr` is not set, a branch will be created automatically from the `main` branch.

### Upload and create a PR

If you don't have the permission to push to a repo, you must open a PR and let the authors know about the changes you want to make. This can be done by setting the `--create-pr` option:

```bash
# Create a PR and upload the files to it
>>> hf upload bigcode/the-stack . . --repo-type dataset --revision refs/pr/104
https://huggingface.co/datasets/bigcode/the-stack/blob/refs%2Fpr%2F104/
```

### Upload at regular intervals

In some cases, you might want to push regular updates to a repo. For example, this is useful if you're training a model and you want to upload the logs folder every 10 minutes. You can do this using the `--every` option:

```bash
# Upload new logs every 10 minutes
hf upload training-model logs/ --every=10
```

### Specify a commit message

Use the `--commit-message` and `--commit-description` to set a custom message and description for your commit instead of the default one

```bash
>>> hf upload Wauplin/my-cool-model ./models . --commit-message="Epoch 34/50" --commit-description="Val accuracy: 68%. Check tensorboard for more details."
...
https://huggingface.co/Wauplin/my-cool-model/tree/main
```

### Specify a token

To upload files, you must use a token. By default, the token saved locally (using `hf auth login`) will be used. If you want to authenticate explicitly, use the `--token` option:

```bash
>>> hf upload Wauplin/my-cool-model ./models . --token=hf_****
...
https://huggingface.co/Wauplin/my-cool-model/tree/main
```

### Quiet mode

By default, the `hf upload` command will be verbose. It will print details such as warning messages, information about the uploaded files, and progress bars. If you want to silence all of this, use the `--quiet` option. Only the last line (i.e. the URL to the uploaded files) is printed. This can prove useful if you want to pass the output to another command in a script.

```bash
>>> hf upload Wauplin/my-cool-model ./models . --quiet
https://huggingface.co/Wauplin/my-cool-model/tree/main
```

## hf repo-files

If you want to delete files from a Hugging Face repository, use the `hf repo-files` command.

### Delete files

The `hf repo-files delete <repo_id>` sub-command allows you to delete files from a repository. Here are some usage examples.

Delete a folder :
```bash
>>> hf repo-files delete Wauplin/my-cool-model folder/
Files correctly deleted from repo. Commit: https://huggingface.co/Wauplin/my-cool-mo...
```

Delete multiple files:
```bash
>>> hf repo-files delete Wauplin/my-cool-model file.txt folder/pytorch_model.bin
Files correctly deleted from repo. Commit: https://huggingface.co/Wauplin/my-cool-mo...
```

Use Unix-style wildcards to delete sets of files:
```bash
>>> hf repo-files delete Wauplin/my-cool-model "*.txt" "folder/*.bin"
Files correctly deleted from repo. Commit: https://huggingface.co/Wauplin/my-cool-mo...
```

### Specify a token

To delete files from a repo you must be authenticated and authorized. By default, the token saved locally (using `hf auth login`) will be used. If you want to authenticate explicitly, use the `--token` option:

```bash
>>> hf repo-files delete --token=hf_**** Wauplin/my-cool-model file.txt
```

## hf cache scan

Scanning your cache directory is useful if you want to know which repos you have downloaded and how much space it takes on your disk. You can do that by running `hf cache scan`:

```bash
>>> hf cache scan
REPO ID                     REPO TYPE SIZE ON DISK NB FILES LAST_ACCESSED LAST_MODIFIED REFS                LOCAL PATH
--------------------------- --------- ------------ -------- ------------- ------------- ------------------- -------------------------------------------------------------------------
glue                        dataset         116.3K       15 4 days ago    4 days ago    2.4.0, main, 1.17.0 /home/wauplin/.cache/huggingface/hub/datasets--glue
google/fleurs               dataset          64.9M        6 1 week ago    1 week ago    refs/pr/1, main     /home/wauplin/.cache/huggingface/hub/datasets--google--fleurs
Jean-Baptiste/camembert-ner model           441.0M        7 2 weeks ago   16 hours ago  main                /home/wauplin/.cache/huggingface/hub/models--Jean-Baptiste--camembert-ner
bert-base-cased             model             1.9G       13 1 week ago    2 years ago                       /home/wauplin/.cache/huggingface/hub/models--bert-base-cased
t5-base                     model            10.1K        3 3 months ago  3 months ago  main                /home/wauplin/.cache/huggingface/hub/models--t5-base
t5-small                    model           970.7M       11 3 days ago    3 days ago    refs/pr/1, main     /home/wauplin/.cache/huggingface/hub/models--t5-small

Done in 0.0s. Scanned 6 repo(s) for a total of 3.4G.
Got 1 warning(s) while scanning. Use -vvv to print details.
```

For more details about how to scan your cache directory, please refer to the [Manage your cache](./manage-cache#scan-cache-from-the-terminal) guide.

## hf cache delete

`hf cache delete` is a tool that helps you delete parts of your cache that you don't use anymore. This is useful for saving and freeing disk space. To learn more about using this command, please refer to the [Manage your cache](./manage-cache#clean-cache-from-the-terminal) guide.

## hf repo tag create

The `hf repo tag create` command allows you to tag, untag, and list tags for repositories.

### Tag a model

To tag a repo, you need to provide the `repo_id` and the `tag` name:

```bash
>>> hf repo tag create Wauplin/my-cool-model v1.0
You are about to create tag v1.0 on model Wauplin/my-cool-model
Tag v1.0 created on Wauplin/my-cool-model
```

### Tag a model at a specific revision

If you want to tag a specific revision, you can use the `--revision` option. By default, the tag will be created on the `main` branch:

```bash
>>> hf repo tag create Wauplin/my-cool-model v1.0 --revision refs/pr/104
You are about to create tag v1.0 on model Wauplin/my-cool-model
Tag v1.0 created on Wauplin/my-cool-model
```

### Tag a dataset or a Space

If you want to tag a dataset or Space, you must specify the `--repo-type` option:

```bash
>>> hf repo tag create bigcode/the-stack v1.0 --repo-type dataset
You are about to create tag v1.0 on dataset bigcode/the-stack
Tag v1.0 created on bigcode/the-stack
```

### List tags

To list all tags for a repository, use the `-l` or `--list` option:

```bash
>>> hf repo tag create Wauplin/gradio-space-ci -l --repo-type space
Tags for space Wauplin/gradio-space-ci:
0.2.2
0.2.1
0.2.0
0.1.2
0.0.2
0.0.1
```

### Delete a tag

To delete a tag, use the `-d` or `--delete` option:

```bash
>>> hf repo tag create -d Wauplin/my-cool-model v1.0
You are about to delete tag v1.0 on model Wauplin/my-cool-model
Proceed? [Y/n] y
Tag v1.0 deleted on Wauplin/my-cool-model
```

You can also pass `-y` to skip the confirmation step.

## hf env

The `hf env` command prints details about your machine setup. This is useful when you open an issue on [GitHub](https://github.com/huggingface/huggingface_hub) to help the maintainers investigate your problem.

```bash
>>> hf env

Copy-and-paste the text below in your GitHub issue.

- huggingface_hub version: 0.19.0.dev0
- Platform: Linux-6.2.0-36-generic-x86_64-with-glibc2.35
- Python version: 3.10.12
- Running in iPython ?: No
- Running in notebook ?: No
- Running in Google Colab ?: No
- Token path ?: /home/wauplin/.cache/huggingface/token
- Has saved token ?: True
- Who am I ?: Wauplin
- Configured git credential helpers: store
- FastAI: N/A
- Tensorflow: 2.11.0
- Torch: 1.12.1
- Jinja2: 3.1.2
- Graphviz: 0.20.1
- Pydot: 1.4.2
- Pillow: 9.2.0
- hf_transfer: 0.1.3
- gradio: 4.0.2
- tensorboard: 2.6
- numpy: 1.23.2
- pydantic: 2.4.2
- aiohttp: 3.8.4
- ENDPOINT: https://huggingface.co
- HF_HUB_CACHE: /home/wauplin/.cache/huggingface/hub
- HF_ASSETS_CACHE: /home/wauplin/.cache/huggingface/assets
- HF_TOKEN_PATH: /home/wauplin/.cache/huggingface/token
- HF_HUB_OFFLINE: False
- HF_HUB_DISABLE_TELEMETRY: False
- HF_HUB_DISABLE_PROGRESS_BARS: None
- HF_HUB_DISABLE_SYMLINKS_WARNING: False
- HF_HUB_DISABLE_EXPERIMENTAL_WARNING: False
- HF_HUB_DISABLE_IMPLICIT_TOKEN: False
- HF_HUB_ENABLE_HF_TRANSFER: False
- HF_HUB_ETAG_TIMEOUT: 10
- HF_HUB_DOWNLOAD_TIMEOUT: 10
```

## hf jobs

Run compute jobs on Hugging Face infrastructure with a familiar Docker-like interface.

`hf jobs` is a command-line tool that lets you run anything on Hugging Face's infrastructure (including GPUs and TPUs!) with simple commands. Think `docker run`, but for running code on A100s.

```bash
# Directly run Python code
>>> hf jobs run python:3.12 python -c 'print("Hello from the cloud!")'

# Use GPUs without any setup
>>> hf jobs run --flavor a10g-small pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
... python -c "import torch; print(torch.cuda.get_device_name())"

# Run in an organization account
>>> hf jobs run --namespace my-org-name python:3.12 python -c "print('Running in an org account')"

# Run from Hugging Face Spaces
>>> hf jobs run hf.co/spaces/lhoestq/duckdb duckdb -c "select 'hello world'"

# Run a Python script with `uv` (experimental)
>>> hf jobs uv run my_script.py
```

### ✨ Key Features

- 🐳 **Docker-like CLI**: Familiar commands (`run`, `ps`, `logs`, `inspect`) to run and manage jobs
- 🔥 **Any Hardware**: From CPUs to A100 GPUs and TPU pods - switch with a simple flag
- 📦 **Run Anything**: Use Docker images, HF Spaces, or your custom containers
- 🔐 **Simple Auth**: Just use your HF token
- 📊 **Live Monitoring**: Stream logs in real-time, just like running locally
- 💰 **Pay-as-you-go**: Only pay for the seconds you use

<Tip>

**Hugging Face Jobs** are available only to [Pro users](https://huggingface.co/pro) and [Team or Enterprise organizations](https://huggingface.co/enterprise). Upgrade your plan to get started!

</Tip>

### Quick Start

#### 1. Run your first job

```bash
# Run a simple Python script
>>> hf jobs run python:3.12 python -c "print('Hello from HF compute!')"
```

This command runs the job and shows the logs. You can pass `--detach` to run the Job in the background and only print the Job ID.

#### 2. Check job status

```bash
# List your running jobs
>>> hf jobs ps

# Inspect the status of a job
>>> hf jobs inspect <job_id>

# View logs from a job
>>> hf jobs logs <job_id>

# Cancel a job
>>> hf jobs cancel <job_id>
```

#### 3. Run on GPU

You can also run jobs on GPUs or TPUs with the `--flavor` option. For example, to run a PyTorch job on an A10G GPU:

```bash
# Use an A10G GPU to check PyTorch CUDA
>>> hf jobs run --flavor a10g-small pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
... python -c "import torch; print(f"This code ran with the following GPU: {torch.cuda.get_device_name()}")"
```

Running this will show the following output!

```bash
This code ran with the following GPU: NVIDIA A10G
```

That's it! You're now running code on Hugging Face's infrastructure.

### Common Use Cases

- **Model Training**: Fine-tune or train models on GPUs (T4, A10G, A100) without managing infrastructure
- **Synthetic Data Generation**: Generate large-scale datasets using LLMs on powerful hardware
- **Data Processing**: Process massive datasets with high-CPU configurations for parallel workloads
- **Batch Inference**: Run offline inference on thousands of samples using optimized GPU setups
- **Experiments & Benchmarks**: Run ML experiments on consistent hardware for reproducible results
- **Development & Debugging**: Test GPU code without local CUDA setup

### Pass Environment variables and Secrets

You can pass environment variables to your job using 

```bash
# Pass environment variables
>>> hf jobs run -e FOO=foo -e BAR=bar python:3.12 python -c "import os; print(os.environ['FOO'], os.environ['BAR'])"
```

```bash
# Pass an environment from a local .env file
>>> hf jobs run --env-file .env python:3.12 python -c "import os; print(os.environ['FOO'], os.environ['BAR'])"
```

```bash
# Pass secrets - they will be encrypted server side
>>> hf jobs run -s MY_SECRET=psswrd python:3.12 python -c "import os; print(os.environ['MY_SECRET'])"
```

```bash
# Pass secrets from a local .env.secrets file - they will be encrypted server side
>>> hf jobs run --secrets-file .env.secrets python:3.12 python -c "import os; print(os.environ['MY_SECRET'])"
```

<Tip>

Use `--secrets HF_TOKEN` to pass your local Hugging Face token implicitly.
With this syntax, the secret is retrieved from the environment variable.
For `HF_TOKEN`, it may read the token file located in the Hugging Face home folder if the environment variable is unset.

</Tip>

### Hardware

Available `--flavor` options:

- CPU: `cpu-basic`, `cpu-upgrade`
- GPU: `t4-small`, `t4-medium`, `l4x1`, `l4x4`, `a10g-small`, `a10g-large`, `a10g-largex2`, `a10g-largex4`,`a100-large`
- TPU: `v5e-1x1`, `v5e-2x2`, `v5e-2x4`

(updated in 07/2025 from Hugging Face [suggested_hardware docs](https://huggingface.co/docs/hub/en/spaces-config-reference))

### UV Scripts (Experimental)

Run UV scripts (Python scripts with inline dependencies) on HF infrastructure:

```bash
# Run a UV script (creates temporary repo)
>>> hf jobs uv run my_script.py

# Run with persistent repo
>>> hf jobs uv run my_script.py --repo my-uv-scripts

# Run with GPU
>>> hf jobs uv run ml_training.py --flavor gpu-t4-small

# Pass arguments to script
>>> hf jobs uv run process.py input.csv output.parquet --repo data-scripts

# Run a script directly from a URL
>>> hf jobs uv run https://huggingface.co/datasets/username/scripts/resolve/main/example.py
```

UV scripts are Python scripts that include their dependencies directly in the file using a special comment syntax. This makes them perfect for self-contained tasks that don't require complex project setups. Learn more about UV scripts in the [UV documentation](https://docs.astral.sh/uv/guides/scripts/).
