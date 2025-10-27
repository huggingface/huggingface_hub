# Hugging Face Hub Client library

## Download files from the Hub

The `hf_hub_download()` function is the main function to download files from the Hub. One
advantage of using it is that files are cached locally, so you won't have to
download the files multiple times. If there are changes in the repository, the
files will be automatically downloaded again.


### `hf_hub_download`

The function takes the following parameters, downloads the remote file,
stores it to disk (in a version-aware way) and returns its local file path.

Parameters:
- a `repo_id` (a user or organization name and a repo name, separated by `/`, like `julien-c/EsperBERTo-small`)
- a `filename` (like `pytorch_model.bin`)
- an optional Git revision id (can be a branch name, a tag, or a commit hash)
- a `cache_dir` which you can specify if you want to control where on disk the
  files are cached.

```python
from huggingface_hub import hf_hub_download
hf_hub_download("lysandre/arxiv-nlp", filename="config.json")
```

### `snapshot_download`

Using `hf_hub_download()` works well when you know which files you want to download;
for example a model file alongside a configuration file, both with static names.
There are cases in which you will prefer to download all the files of the remote
repository at a specified revision. That's what `snapshot_download()` does. It
downloads and stores a remote repository to disk (in a versioning-aware way) and
returns its local file path.

Parameters:
- a `repo_id` in the format `namespace/repository`
- a `revision` on which the repository will be downloaded
- a `cache_dir` which you can specify if you want to control where on disk the
  files are cached

### `hf_hub_url`

Internally, the library uses `hf_hub_url()` to return the URL to download the actual files:
`https://huggingface.co/julien-c/EsperBERTo-small/resolve/main/pytorch_model.bin`


Parameters:
- a `repo_id` (a user or organization name and a repo name separated by a `/`, like `julien-c/EsperBERTo-small`)
- a `filename` (like `pytorch_model.bin`)
- an optional `subfolder`, corresponding to a folder inside the model repo
- an optional `repo_type`, such as `dataset` or `space`
- an optional Git revision id (can be a branch name, a tag, or a commit hash)

If you check out this URL's headers with a `HEAD` http request (which you can do
from the command line with `curl -I`) for a few different files, you'll see
that:
- small files are returned directly
- large files (i.e. the ones stored through
  [git-lfs](https://git-lfs.github.com/)) are returned via a redirect to a
  Cloudfront URL. Cloudfront is a Content Delivery Network, or CDN, that ensures
  that downloads are as fast as possible from anywhere on the globe.

<br>

## Publish files to the Hub

If you've used Git before, this will be very easy since Git is used to manage
files in the Hub. You can find a step-by-step guide on how to upload your model
to the Hub: https://huggingface.co/docs/hub/adding-a-model.


### API utilities in `hf_api.py`

You don't need them for the standard publishing workflow (ie. using git command line), however, if you need a
programmatic way of creating a repo, deleting it (`⚠️ caution`), pushing a
single file to a repo or listing models from the Hub, you'll find helpers in
`hf_api.py`. Some example functionality available with the `HfApi` class:

* `whoami()`
* `create_repo()`
* `list_repo_files()`
* `list_repo_objects()`
* `delete_repo()`
* `update_repo_settings()`
* `create_commit()`
* `upload_file()`
* `delete_file()`
* `delete_folder()`

Those API utilities are also exposed through the `hf` CLI:

```bash
hf auth login
hf auth logout
hf auth whoami
hf repo create
```

With the `HfApi` class there are methods to query models, datasets, and Spaces by specific tags (e.g. if you want to list models compatible with your library):
- **Models**:
  - `list_models()`
  - `model_info()`
  - `get_model_tags()`
- **Datasets**:
  - `list_datasets()`
  - `dataset_info()`
  - `get_dataset_tags()`
- **Spaces**:
  - `list_spaces()`
  - `space_info()`

These lightly wrap around the API Endpoints. Documentation for valid parameters and descriptions can be found [here](https://huggingface.co/docs/hub/endpoints).
