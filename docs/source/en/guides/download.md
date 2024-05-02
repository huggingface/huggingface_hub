<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Download files from the Hub

The `huggingface_hub` library provides functions to download files from the repositories
stored on the Hub. You can use these functions independently or integrate them into your
own library, making it more convenient for your users to interact with the Hub. This
guide will show you how to:

* Download and cache a single file.
* Download and cache an entire repository.
* Download files to a local folder.

## Download a single file

The [`hf_hub_download`] function is the main function for downloading files from the Hub.
It downloads the remote file, caches it on disk (in a version-aware way), and returns its local file path.

<Tip>

The returned filepath is a pointer to the HF local cache. Therefore, it is important to not modify the file to avoid
having a corrupted cache. If you are interested in getting to know more about how files are cached, please refer to our
[caching guide](./manage-cache).

</Tip>

### From latest version

Select the file to download using the `repo_id`, `repo_type` and `filename` parameters. By default, the file will
be considered as being part of a `model` repo.

```python
>>> from huggingface_hub import hf_hub_download
>>> hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json")
'/root/.cache/huggingface/hub/models--lysandre--arxiv-nlp/snapshots/894a9adde21d9a3e3843e6d5aeaaf01875c7fade/config.json'

# Download from a dataset
>>> hf_hub_download(repo_id="google/fleurs", filename="fleurs.py", repo_type="dataset")
'/root/.cache/huggingface/hub/datasets--google--fleurs/snapshots/199e4ae37915137c555b1765c01477c216287d34/fleurs.py'
```

### From specific version

By default, the latest version from the `main` branch is downloaded. However, in some cases you want to download a file
at a particular version (e.g. from a specific branch, a PR, a tag or a commit hash).
To do so, use the `revision` parameter:

```python
# Download from the `v1.0` tag
>>> hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json", revision="v1.0")

# Download from the `test-branch` branch
>>> hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json", revision="test-branch")

# Download from Pull Request #3
>>> hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json", revision="refs/pr/3")

# Download from a specific commit hash
>>> hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json", revision="877b84a8f93f2d619faa2a6e514a32beef88ab0a")
```

**Note:** When using the commit hash, it must be the full-length hash instead of a 7-character commit hash.

### Construct a download URL

In case you want to construct the URL used to download a file from a repo, you can use [`hf_hub_url`] which returns a URL.
Note that it is used internally by [`hf_hub_download`].

## Download an entire repository

[`snapshot_download`] downloads an entire repository at a given revision. It uses internally [`hf_hub_download`] which
means all downloaded files are also cached on your local disk. Downloads are made concurrently to speed-up the process.

To download a whole repository, just pass the `repo_id` and `repo_type`:

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="lysandre/arxiv-nlp")
'/home/lysandre/.cache/huggingface/hub/models--lysandre--arxiv-nlp/snapshots/894a9adde21d9a3e3843e6d5aeaaf01875c7fade'

# Or from a dataset
>>> snapshot_download(repo_id="google/fleurs", repo_type="dataset")
'/home/lysandre/.cache/huggingface/hub/datasets--google--fleurs/snapshots/199e4ae37915137c555b1765c01477c216287d34'
```

[`snapshot_download`] downloads the latest revision by default. If you want a specific repository revision, use the
`revision` parameter:

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="lysandre/arxiv-nlp", revision="refs/pr/1")
```

### Filter files to download

[`snapshot_download`] provides an easy way to download a repository. However, you don't always want to download the
entire content of a repository. For example, you might want to prevent downloading all `.bin` files if you know you'll
only use the `.safetensors` weights. You can do that using `allow_patterns` and `ignore_patterns` parameters.

These parameters accept either a single pattern or a list of patterns. Patterns are Standard Wildcards (globbing
patterns) as documented [here](https://tldp.org/LDP/GNU-Linux-Tools-Summary/html/x11655.htm). The pattern matching is
based on [`fnmatch`](https://docs.python.org/3/library/fnmatch.html).

For example, you can use `allow_patterns` to only download JSON configuration files:

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="lysandre/arxiv-nlp", allow_patterns="*.json")
```

On the other hand, `ignore_patterns` can exclude certain files from being downloaded. The
following example ignores the `.msgpack` and `.h5` file extensions:

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="lysandre/arxiv-nlp", ignore_patterns=["*.msgpack", "*.h5"])
```

Finally, you can combine both to precisely filter your download. Here is an example to download all json and markdown
files except `vocab.json`.

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="gpt2", allow_patterns=["*.md", "*.json"], ignore_patterns="vocab.json")
```

## Download file(s) to a local folder

By default, we recommend using the [cache system](./manage-cache) to download files from the Hub. You can specify a custom cache location using the `cache_dir` parameter in [`hf_hub_download`] and [`snapshot_download`], or by setting the [`HF_HOME`](../package_reference/environment_variables#hf_home) environment variable.

However, if you need to download files to a specific folder, you can pass a `local_dir` parameter to the download function. This is useful to get a workflow closer to what the `git` command offers. The downloaded files will maintain their original file structure within the specified folder. For example, if `filename="data/train.csv"` and `local_dir="path/to/folder"`, the resulting filepath will be `"path/to/folder/data/train.csv"`.

A `./huggingface/` folder is created at the root of your local directory containing metadata about the downloaded files. This prevents re-downloading files if they're already up-to-date. If the metadata has changed, then the new file version is downloaded. This makes the `local_dir` optimized for pulling only the latest changes.

After completing the download, you can safely remove the `.cache/huggingface/` folder if you no longer need it. However, be aware that re-running your script without this folder may result in longer recovery times, as metadata will be lost. Rest assured that your local data will remain intact and unaffected.

<Tip>

Don't worry about the `.cache/huggingface/` folder when committing changes to the Hub! This folder is automatically ignored by both `git` and [`upload_folder`].

</Tip>

## Download from the CLI

You can use the `huggingface-cli download` command from the terminal to directly download files from the Hub.
Internally, it uses the same [`hf_hub_download`] and [`snapshot_download`] helpers described above and prints the
returned path to the terminal.

```bash
>>> huggingface-cli download gpt2 config.json
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10/config.json
```

You can download multiple files at once which displays a progress bar and returns the snapshot path in which the files
are located:

```bash
>>> huggingface-cli download gpt2 config.json model.safetensors
Fetching 2 files: 100%|████████████████████████████████████████████| 2/2 [00:00<00:00, 23831.27it/s]
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10
```

For more details about the CLI download command, please refer to the [CLI guide](./cli#huggingface-cli-download).

## Faster downloads

If you are running on a machine with high bandwidth,
you can increase your download speed with [`hf_transfer`](https://github.com/huggingface/hf_transfer),
a Rust-based library developed to speed up file transfers with the Hub.
To enable it:

1. Specify the `hf_transfer` extra when installing `huggingface_hub`
   (e.g. `pip install huggingface_hub[hf_transfer]`).
2. Set `HF_HUB_ENABLE_HF_TRANSFER=1` as an environment variable.

<Tip warning={true}>

`hf_transfer` is a power user tool!
It is tested and production-ready,
but it lacks user-friendly features like advanced error handling or proxies.
For more details, please take a look at this [section](https://huggingface.co/docs/huggingface_hub/hf_transfer).

</Tip>
