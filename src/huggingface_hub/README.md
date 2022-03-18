# Hugging Face Hub Client library

## Download files from the Hub

Three utility functions are provided to dowload files from the Hub. One
advantage of using them is that files are cached locally, so you won't have to
download the files multiple times. If there are changes in the repository, the
files will be automatically downloaded again.

### `hf_hub_url`

`hf_hub_url()` returns the url we'll use to download the actual files:
`https://huggingface.co/julien-c/EsperBERTo-small/resolve/main/pytorch_model.bin`

Parameters:
- a `repo_id` (a user or organization name and a repo name seperated by a `/`, like `julien-c/EsperBERTo-small`)
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

### `cached_download`

`cached_download()` takes the following parameters, downloads the remote file,
stores it to disk (in a versioning-aware way) and returns its local file path.

Parameters:
- a remote `url`
- a `cache_dir` which you can specify if you want to control where on disk the
  files are cached.

A common use case is to download the files from a download url

```python
from huggingface_hub import hf_hub_url, cached_download
config_file_url = hf_hub_url("lysandre/arxiv-nlp", filename="config.json")
cached_download(config_file_url)
```

Check out the [source code](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/file_download.py) and search for `cached_download` for all possible params (we'll create a real doc page
in the future).

### `hf_hub_download`

Since the use case of combining `hf_hub_url()` and `cached_download()` is very
common, we also provide a wrapper that calls both functions.

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

Using `hf_hub_download()` works well when you have a fixed repository structure;
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

<br>

## Publish files to the Hub

If you've used Git before, this will be very easy since Git is used to manage
files in the Hub. You can find a step-by-step guide on how to upload your model
to the Hub: https://huggingface.co/docs/hub/adding-a-model. 


### API utilities in `hf_api.py`

You don't need them for the standard publishing workflow, however, if you need a
programmatic way of creating a repo, deleting it (`⚠️ caution`), pushing a
single file to a repo or listing models from the Hub, you'll find helpers in
`hf_api.py`. Some example functionality available with the `HfApi` class:

* `login()`
* `whoami()`
* `logout()`
* `create_repo()`
* `list_repo_files()`
* `list_repo_objects()`
* `delete_repo()`
* `update_repo_visibility()`
* `upload_file()`
* `delete_file()`

Those API utilities are also exposed through the `huggingface-cli` CLI:

```bash
huggingface-cli login
huggingface-cli logout
huggingface-cli whoami
huggingface-cli repo create
```

With the `HfApi` class there are methods to query models, datasets, and metrics by specific tags (e.g. if you want to list models compatible with your library):
- **Models**:
  - `list_models()`
  - `model_info()`
  - `get_model_tags()`
- **Datasets**:
  - `list_datasets()`
  - `dataset_info()`
  - `get_dataset_tags()`
  
These lightly wrap around the API Endpoints. Documentation for valid parameters and descriptions can be found [here](https://huggingface.co/docs/hub/endpoints).
  

### Advanced programmatic repository management 

The `Repository` class helps manage both offline Git repositories and Hugging
Face Hub repositories. Using the `Repository` class requires `git` and `git-lfs`
to be installed.

Instantiate a `Repository` object by calling it with a path to a local Git
clone/repository:

```python
>>> from huggingface_hub import Repository
>>> repo = Repository("<path>/<to>/<folder>")
```

The `Repository` takes a `clone_from` string as parameter. This can stay as
`None` for offline management, but can also be set to any URL pointing to a Git
repo to clone that repository in the specified directory:

```python
>>> repo = Repository("huggingface-hub", clone_from="https://github.com/huggingface/huggingface_hub")
```

The `clone_from` method can also take any Hugging Face model ID as input, and
will clone that repository:

```python
>>> repo = Repository("w2v2", clone_from="facebook/wav2vec2-large-960h-lv60")
```

If the repository you're cloning is one of yours or one of your organisation's,
then having the ability to commit and push to that repository is important. In
order to do that, you should make sure to be logged-in using `huggingface-cli
login`, and to have the `use_auth_token` parameter set to `True` (the default)
when  instantiating the `Repository` object:

```python
>>> repo = Repository("my-model", clone_from="<user>/<model_id>", use_auth_token=True)
```

This works for models, datasets and spaces repositories; but you will need to
explicitely specify the type for the last two options:

```python
>>> repo = Repository("my-dataset", clone_from="<user>/<dataset_id>", use_auth_token=True, repo_type="dataset")
```

You can also change between branches:

```python
>>> repo = Repository("huggingface-hub", clone_from="<user>/<dataset_id>", revision='branch1')
>>> repo.git_checkout("branch2")
```

The `clone_from` method can also take any Hugging Face model ID as input, and
will clone that repository:

```python
>>> repo = Repository("w2v2", clone_from="facebook/wav2vec2-large-960h-lv60")
```

Finally, you can choose to specify the Git username and email attributed to that
clone directly by using the `git_user` and `git_email` parameters. When
committing to that repository, Git will therefore be aware of who you are and
who will be the author of the commits:

```python
>>> repo = Repository(
...   "my-dataset", 
...   clone_from="<user>/<dataset_id>", 
...   use_auth_token=True, 
...   repo_type="dataset",
...   git_user="MyName",
...   git_email="me@cool.mail"
... )
```

The repository can be managed through this object, through wrappers of
traditional Git methods:

- `git_add(pattern: str, auto_lfs_track: bool)`. The `auto_lfs_track` flag
  triggers auto tracking of large files (>10MB) with `git-lfs`
- `git_commit(commit_message: str)`
- `git_pull(rebase: bool)`
- `git_push()`
- `git_checkout(branch)`

The `git_push` method has a parameter `blocking` which is `True` by default. When set to `False`, the push will
happen behind the scenes - which can be helpful if you would like your script to continue on while the push is 
happening.

LFS-tracking methods:

- `lfs_track(pattern: Union[str, List[str]], filename: bool)`. Setting
  `filename` to `True` will use the `--filename` parameter, which will consider
  the pattern(s) as filenames, even if they contain special glob characters.
- `lfs_untrack()`.
- `auto_track_large_files()`: automatically tracks files that are larger than
  10MB. Make sure to call this after adding files to the index.

On top of these unitary methods lie some useful additional methods:

- `push_to_hub(commit_message)`: consecutively does `git_add`, `git_commit` and
  `git_push`.
- `commit(commit_message: str, track_large_files: bool)`: this is a context
  manager utility that handles committing to a repository. This automatically
  tracks large files (>10Mb) with `git-lfs`. The `track_large_files` argument can
  be set to `False` if you wish to ignore that behavior.

These two methods also have support for the `blocking` parameter.

Examples using the `commit` context manager:
```python
>>> with Repository("text-files", clone_from="<user>/text-files", use_auth_token=True).commit("My first file :)"):
...     with open("file.txt", "w+") as f:
...         f.write(json.dumps({"hey": 8}))
```

```python
>>> import torch
>>> model = torch.nn.Transformer()
>>> with Repository("torch-model", clone_from="<user>/torch-model", use_auth_token=True).commit("My cool model :)"):
...     torch.save(model.state_dict(), "model.pt")
  ```

### Non-blocking behavior

The pushing methods have access to a `blocking` boolean parameter to indicate whether the push should happen
asynchronously.

In order to see if the push has finished or its status code (to spot a failure), one should use the `command_queue`
property on the `Repository` object.

For example:

```python
from huggingface_hub import Repository

repo = Repository("<local_folder>", clone_from="<user>/<model_name>")

with repo.commit("Commit message", blocking=False):
    # Save data

last_command = repo.command_queue[-1]

# Status of the push command
last_command.status  
# Will return the status code
#     -> -1 will indicate the push is still ongoing
#     -> 0 will indicate the push has completed successfully
#     -> non-zero code indicates the error code if there was an error

# if there was an error, the stderr may be inspected
last_command.stderr

# Whether the command finished or if it is still ongoing
last_command.is_done

# Whether the command errored-out.
last_command.failed
```

When using `blocking=False`, the commands will be tracked and your script will exit only when all pushes are done, even
if other errors happen in your script (a failed push counts as done).


### Need to upload very large (>5GB) files?

To upload large files (>5GB 🔥), you need to install the custom transfer agent
for git-lfs, bundled in this package. 

To install, just run:

```bash
$ huggingface-cli lfs-enable-largefiles
```

This should be executed once for each model repo that contains a model file
>5GB. If you just try to push a file bigger than 5GB without running that
command, you will get an error with a message reminding you to run it.

Finally, there's a `huggingface-cli lfs-multipart-upload` command but that one
is internal (called by lfs directly) and is not meant to be called by the user.

<br>

## Using the Inference API wrapper

`huggingface_hub` comes with a wrapper client to make calls to the Inference
API! You can find some examples below, but we encourage you to visit the
Inference API
[documentation](https://api-inference.huggingface.co/docs/python/html/detailed_parameters.html)
to review the specific parameters for the different tasks.

When you instantiate the wrapper to the Inference API, you specify the model
repository id. The pipeline (`text-classification`,  `text-to-speech`, etc) is
automatically extracted from the
[repository](https://huggingface.co/docs/hub/main#how-is-a-models-type-of-inference-api-and-widget-determined),
but you can also override it as shown below.


### Examples

Here is a basic example of calling the Inference API for a `fill-mask` task
using the `bert-base-uncased` model. The `fill-mask` task only expects a string
(or list of strings) as input.

```python
from huggingface_hub.inference_api import InferenceApi
inference = InferenceApi("bert-base-uncased", token=API_TOKEN)
inference(inputs="The goal of life is [MASK].")
>> [{'sequence': 'the goal of life is life.', 'score': 0.10933292657136917, 'token': 2166, 'token_str': 'life'}]
```

This is an example of a task (`question-answering`) which requires a dictionary
as input thas has the `question` and `context` keys.

```python
inference = InferenceApi("deepset/roberta-base-squad2", token=API_TOKEN)
inputs = {"question":"What's my name?", "context":"My name is Clara and I live in Berkeley."}
inference(inputs)
>> {'score': 0.9326569437980652, 'start': 11, 'end': 16, 'answer': 'Clara'}
```

Some tasks might also require additional params in the request. Here is an
example using a `zero-shot-classification` model.

```python
inference = InferenceApi("typeform/distilbert-base-uncased-mnli", token=API_TOKEN)
inputs = "Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!"
params = {"candidate_labels":["refund", "legal", "faq"]}
inference(inputs, params)
>> {'sequence': 'Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!', 'labels': ['refund', 'faq', 'legal'], 'scores': [0.9378499388694763, 0.04914155602455139, 0.013008488342165947]}
```

Finally, there are some models that might support multiple tasks. For example,
`sentence-transformers` models can do `sentence-similarity` and
`feature-extraction`. You can override the configured task when initializing the
API.

```python
inference = InferenceApi("bert-base-uncased", task="feature-extraction", token=API_TOKEN)
```

