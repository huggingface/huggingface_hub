---
title: How to create repositories and upload files to the Hub
---

# How to integrate upstream utilities in your library

*Upstream* utilities allow you to publish files to the Hub from your library. This guide will show you how to:

* Use the repository-management methods available in the `huggingface_hub` package.
* Use the `Repository` class to handle files and version control a repository with Git-like commands.

## `huggingface_hub` repository-management methods

The `huggingface_hub` package offers high-level methods that wraps around HTTP requests. There are many valuable tasks you can accomplish with it, including: 

- List and filter models and datasets.
- Inspect model or dataset metadata.
- Delete a repository.
- Change the visibility of a repository.

### List and filter

It can be helpful for users to see a list of available models and filter them according to a specific language or library. This can be especially useful for library and organization owners who want to view all their models. Use the `list_models` function with the `filter` parameter to search for specific models.

You can view all the available filters on the left of the [model Hub](http://hf.co/models).

![/docs/assets/hub/hub_filters.png](/docs/assets/hub/hub_filters.png)

```python
>>> from huggingface_hub import list_models

# List all models.
>>> list_models()

# List only text classification models.
>>> list_models(filter="text-classification")

# List only Russian models compatible with PyTorch.
>>> list_models(filter=("languages:ru", "pytorch"))

# List only the models trained on the "common_voice" dataset.
>>> list_models(filter="dataset:common_voice")

# List only the models from the spaCy library.
>>> list_models(filter="spacy")
```

Explore available public datasets with `list_datasets`:

```python
>>> from huggingface_hub import list_datasets

# List only text classification datasets.
>>> list_datasets(filter="task_categories:text-classification")

# List only datasets in Russian for language modeling.
>>> list_datasets(filter=("languages:ru", "task_ids:language-modeling"))
```

### Inspect model or dataset metadata

Get important information about a model or dataset as shown below:

```python
>>> from huggingface_hub import model_info, dataset_info

# Get metadata of a single model.
>>> model_info("distilbert-base-uncased")

# Get metadata of a single dataset.
>>> dataset_info("glue")
```

### Create a repository

Create a repository with `create_repo` and give it a name with the `name` parameter.

```python
>>> from huggingface_hub import create_repo
>>> create_repo("test-model")
'https://huggingface.co/lysandre/test-model'
```
### Delete a repository

Delete a repository with `delete_repo`. Make sure you are certain you want to delete a repository because this is an irreversible process!

Pass the full repository ID to `delete_repo`. The full repository ID looks like `{username_or_org}/{repo_name}`, and you can retrieve it with `get_full_repo_name()` as shown below:

```python
>>> from huggingface_hub import get_full_repo_name, delete_repo
>>> name = get_full_repo_name(repo_name)
>>> delete_repo(name=name)
```

Delete a dataset repository by adding the `repo_type` parameter:

```python
>>> delete_repo(name=REPO_NAME, repo_type="dataset")
```

### Change repository visibility

A repository can be public or private. A private repository is only visible to you or members of the organization in which the repository is located. Change a repository to private as shown in the following:

```python
>>> from huggingface_hub import update_repo_visibility
>>> update_repo_visibility(name=REPO_NAME, private=True)
```

## `Repository` 

The `Repository` class allows you to push models or other repositories to the Hub. `Repository` is a wrapper over Git and Git-LFS methods, so make sure you have Git-LFS installed (see [here](https://git-lfs.github.com/) for installation instructions) and set up before you begin. The `Repository` class should feel familiar if you are already familiar with common Git commands. 

### Clone a repository

The `clone_from` parameter clones a repository from a Hugging Face model ID to a directory specified by the `local_dir` argument:

```python
>>> repo = Repository(local_dir="w2v2", clone_from="facebook/wav2vec2-large-960h-lv60")
```

`clone_from` can also clone a repository from a specified directory using a URL (if you are working offline, this parameter should be `None`):

```python
>>> repo = Repository(local_dir="huggingface-hub", clone_from="https://github.com/huggingface/huggingface_hub")
```

Easily combine the `clone_from` parameter with `create_repo` to create and clone a repository:

```python
>>> repo_url = create_repo(name="repo_name")
>>> repo = Repository(local_dir="repo_local_path", clone_from=repo_url)
```

### Using a local clone

Instantiate a `Repository` object with a path to a local Git clone or repository:

```python
>>> from huggingface_hub import Repository
>>> repo = Repository(local_dir="<path>/<to>/<folder>")
```

### Commit and push to a cloned repository

If you want to commit or push to a cloned repository that belongs to you or your organizations:

1. Log in to your Hugging Face account with the following command:

   ```bash
   huggingface-cli login
   ```

2. Alternatively, if you prefer working from a Jupyter or Colaboratory notebook, login with `notebook_login`:

   ```python
   >>> from huggingface_hub import notebook_login
   >>> notebook_login()
   ```

   `notebook_login` will launch a widget in your notebook from which you can enter your Hugging Face credentials.

3. Instantiate a `Repository` class:
   
   ```python
   >>> repo = Repository(local_dir="my-model", clone_from="<user>/<model_id>")
   ```

You can also attribute a Git username and email to a cloned repository by specifying the `git_user` and `git_email` parameters. When users commit to that repository, Git will be aware of the commit author.

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

### Branch

Switch between branches with `git_checkout`. For example, if you want to switch from `branch1` to `branch2`:

```python
>>> repo = Repository(local_dir="huggingface-hub", clone_from="<user>/<dataset_id>", revision='branch1')
>>> repo.git_checkout("branch2")
```

### Pull

Update a current local branch with `git_pull`:

```python
>>> repo.git_pull()
```

Set `rebase=True` if you want your local commits to occur after your branch is updated with the new commits from the remote:

```python
>>> repo.git_pull(rebase=True)
```

### `commit` context manager

The `commit` context manager is a simple utility that handles four of the most common Git commands: pull, add, commit, and push. `git-lfs` automatically tracks any file larger than 10MB. In the following example, the `commit` context manager:

1. Pulls from the `text-files` repository.
2. Adds a change made to `file.txt`.
3. Commits the change.
4. Pushes the change to the `text-files` repository.

```python
>>> with Repository(local_dir="text-files", clone_from="<user>/text-files").commit(commit_message="My first file :)"):
...     with open("file.txt", "w+") as f:
...         f.write(json.dumps({"hey": 8}))
```

Here is another example of how to save and upload a file to a repository:

```python
>>> import torch
>>> model = torch.nn.Transformer()
>>> with Repository("torch-model", clone_from="<user>/torch-model", use_auth_token=True).commit(commit_message="My cool model :)"):
...     torch.save(model.state_dict(), "model.pt")
```

Set `blocking=False` if you would like to push your commits asynchronously. Non-blocking behavior is helpful when you want to continue running your script while you push your commits.

```python
>>> with repo.commit(commit_message="My cool model :)", blocking=False)
```

You can check the status of your push with the `command_queue` property:

```python
>>> last_command = repo.command_queue[-1]
>>> last_command.status
# -> -1 indicates the push is ongoing.
# -> 0 indicates the push has completed successfully.
# -> Non-zero code indicates the error code if there was an error.
```

When `blocking=False`, commands are tracked, and your script will only exit when all pushes are completed, even if other errors occur in your script. Some additional useful commands for checking the status of a push include:

```python
# Inspect an error.
>>> last_command.stderr

# Check whether a push is completed or ongoing.
>>> last_command.is_done

# Check whether a push command has errored.
>>> last_command.failed
```

### `push_to_hub`

The `Repository` class also has a `push_to_hub` utility to add files, make a commit, and push them to a repository. Unlike the `commit` context manager, `push_to_hub` requires you to pull from a repository first, save the files, and then call `push_to_hub`.

```python
>>> repo.git_pull()
>>> repo.push_to_hub(commit_message="Commit my-awesome-file to the Hub")
```

## Upload very large files

For huge files (>5GB), you need to install a custom transfer agent for Git-LFS:

```bash
huggingface-cli lfs-enable-largefiles
```

You should install this for each model repository that contains a model file. Once installed, you are now able to push files larger than 5GB.