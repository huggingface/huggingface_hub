# Hugging Face Client library

## Download files from the huggingface.co hub

Integration inside a library is super simple. We expose two functions, `hf_hub_url()` and `cached_download()`.

### `hf_hub_url`

`hf_hub_url()` takes:
- a repo id (e.g. a model id like `julien-c/EsperBERTo-small` i.e. a user or organization name and a repo name, separated by `/`),
- a filename (like `pytorch_model.bin`),
- and an optional git revision id (can be a branch name, a tag, or a commit hash)

and returns the url we'll use to download the actual files: `https://huggingface.co/julien-c/EsperBERTo-small/resolve/main/pytorch_model.bin`

If you check out this URL's headers with a `HEAD` http request (which you can do from the command line with `curl -I`) for a few different files, you'll see that:
- small files are returned directly
- large files (i.e. the ones stored through [git-lfs](https://git-lfs.github.com/)) are returned via a redirect to a Cloudfront URL. Cloudfront is a Content Delivery Network, or CDN, that ensures that downloads are as fast as possible from anywhere on the globe.

### `cached_download`

`cached_download()` takes the following parameters, downloads the remote file, stores it to disk (in a versioning-aware way) and returns its local file path.

Parameters:
- a remote `url`
- your library's name and version (`library_name` and `library_version`), which will be added to the HTTP requests' user-agent so that we can provide some usage stats.
- a `cache_dir` which you can specify if you want to control where on disk the files are cached.

Check out the source code for all possible params (we'll create a real doc page in the future).

### Bonus: `snapshot_download`

`snapshot_download()` downloads all the files from the remote repository at the specified revision, 
stores it to disk (in a versioning-aware way) and returns its local file path.

Parameters:
- a `repo_id` in the format `namespace/repository`
- a `revision` on which the repository will be downloaded
- a `cache_dir` which you can specify if you want to control where on disk the files are cached.

<br>

## Publish models to the huggingface.co hub

Uploading a model to the hub is super simple too:
- create a model repo directly from the website, at huggingface.co/new (models can be public or private, and are namespaced under either a user or an organization)
- clone it with git
- [download and install git lfs](https://git-lfs.github.com/) if you don't already have it on your machine (you can check by running a simple `git lfs`)
- add, commit and push your files, from git, as you usually do (or using the `Repository` class detailed below). 

**We are intentionally not wrapping git too much, so that you can go on with the workflow you’re used to and the tools you already know.**

> 👀 To see an example of how we document the model sharing process in `transformers`, check out https://huggingface.co/transformers/model_sharing.html

Users add tags into their README.md model cards (e.g. your `library_name`, a domain tag like `audio`, etc.) to make sure their models are discoverable.

**Documentation about the model hub itself is at https://huggingface.co/docs**

### API utilities in `hf_api.py`

You don't need them for the standard publishing workflow, however, if you need a programmatic way of creating a repo, deleting it (`⚠️ caution`), pushing a single file to a repo or listing models from the hub, you'll find helpers in `hf_api.py`.

We also have an API to query models by specific tags (e.g. if you want to list models compatible to your library)

### `huggingface-cli`

Those API utilities are also exposed through a CLI:

```bash
huggingface-cli login
huggingface-cli logout
huggingface-cli whoami
huggingface-cli repo create
```

### Need to upload large (>5GB) files?

To upload large files (>5GB 🔥), you need to install the custom transfer agent for git-lfs, bundled in this package. 

To install, just run:

```bash
$ huggingface-cli lfs-enable-largefiles
```

This should be executed once for each model repo that contains a model file >5GB. If you just try to push a file bigger than 5GB without running that command, you will get an error with a message reminding you to run it.

Finally, there's a `huggingface-cli lfs-multipart-upload` command but that one is internal (called by lfs directly) and is not meant to be called by the user.


## Managing a repository with `Repository`

The `Repository` class helps manage both offline git repositories, and huggingface hub repositories. Using the
`Repository` class requires `git` and `git-lfs` to be installed.

Instantiate a `Repository` object by calling it with a path to a local git clone/repository:

```python
>>> from huggingface_hub import Repository
>>> repo = Repository("<path>/<to>/<folder>")
```

The `Repository` takes a `clone_from` string as parameter. This can stay as `None` for offline management, but can
also be set to any URL pointing to a git repo to clone that repository in the specified directory:

```python
>>> repo = Repository("huggingface-hub", clone_from="https://github.com/huggingface/huggingface_hub")
```

The `clone_from` method can also take any Hugging Face model ID as input, and will clone that repository:

```python
>>> repo = Repository("w2v2", clone_from="facebook/wav2vec2-large-960h-lv60")
```

If the repository you're cloning is one of yours or one of your organisation's, then having the ability
to commit and push to that repository is important. In order to do that, you should make sure to be logged-in
using `huggingface-cli login`, and to have the `use_auth_token` parameter set to `True` (the default) when 
instantiating the `Repository` object:

```python
>>> repo = Repository("my-model", clone_from="<user>/<model_id>", use_auth_token=True)
```

This works for models, datasets and spaces repositories; but you will need to explicitely specify the type for the two
last options:

```python
>>> repo = Repository("my-dataset", clone_from="<user>/<dataset_id>", use_auth_token=True, repo_type="dataset")
```

Finally, you can choose to specify the git username and email attributed to that clone directly by using
the `git_user` and `git_email` parameters. When committing to that repository, git will therefore be aware
of who you are and who will be the author of the commits:

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

The repository can be managed through this object, through wrappers of traditional git methods:

- `git_add(pattern: str, auto_lfs_track: bool)`. The `auto_lfs_track` flag
  triggers auto tracking of large files (>10MB) with `git-lfs`.
- `git_commit(commit_message: str)`.
- `git_pull(rebase: bool)`.
- `git_push()`.

LFS-tracking methods:

- `lfs_track(pattern: Union[str, List[str]], filename: bool)`.
  Setting `filename` to `True` will use the `--filename` parameter, which will consider the pattern(s) as 
  filenames, even if they contain special glob characters.
- `lfs_untrack()`.
- `auto_track_large_files()`: automatically tracks files that are larger than 10MB. Make sure to call this
  after adding files to the index.
  
On top of these unitary methods lie some additional useful methods:

- `push_to_hub(commit_message)`: consecutively does `git_add`, `git_commit` and `git_push`.
- `commit(commit_message: str, track_large_files: bool)`: this is a context manager utility that handles
  committing to a repository. This automatically tracks large files (>10Mb) with git-lfs. The `track_large_files`
  argument can be set to `False` if you wish to ignore that behavior.


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

<br>

## Feedback (feature requests, bugs, etc.) is super welcome 💙💚💛💜♥️🧡
