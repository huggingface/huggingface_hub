<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Upload files to the Hub

Sharing your files and work is an important aspect of the Hub. The `huggingface_hub` offers several options for uploading your files to the Hub. You can use these functions independently or integrate them into your library, making it more convenient for your users to interact with the Hub. This guide will show you how to push files:

- without using Git.
- that are very large with [Git LFS](https://git-lfs.github.com/).
- with the `commit` context manager.
- with the [`~Repository.push_to_hub`] function.

Whenever you want to upload files to the Hub, you need to log in to your Hugging Face account:

- Log in to your Hugging Face account with the following command:

  ```bash
  huggingface-cli login
  # or using an environment variable
  huggingface-cli login --token $HUGGINGFACE_TOKEN
  ```

- Alternatively, you can programmatically login using [`login`] in a notebook or a script:

  ```python
  >>> from huggingface_hub import login
  >>> login()
  ```

  If ran in a Jupyter or Colaboratory notebook, [`login`] will launch a widget from
  which you can enter your Hugging Face access token. Otherwise, a message will be
  prompted in the terminal.

  It is also possible to login programmatically without the widget by directly passing
  the token to [`login`]. If you do so, be careful when sharing your notebook. It is
  best practice to load the token from a secure vault instead of saving it in plain in
  your Colaboratory notebook.

## Upload a file

Once you've created a repository with [`create_repo`], you can upload a file to your repository using [`upload_file`].

Specify the path of the file to upload, where you want to upload the file to in the repository, and the name of the repository you want to add the file to. Depending on your repository type, you can optionally set the repository type as a `dataset`, `model`, or `space`.

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.upload_file(
...     path_or_fileobj="/path/to/local/folder/README.md",
...     path_in_repo="README.md",
...     repo_id="username/test-dataset",
...     repo_type="dataset",
... )
```

## Upload a folder

Use the [`upload_folder`] function to upload a local folder to an existing repository. Specify the path of the local folder
to upload, where you want to upload the folder to in the repository, and the name of the repository you want to add the
folder to. Depending on your repository type, you can optionally set the repository type as a `dataset`, `model`, or `space`.

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()

# Upload all the content from the local folder to your remote Space.
# By default, files are uploaded at the root of the repo
>>> api.upload_folder(
...     folder_path="/path/to/local/space",
...     repo_id="username/my-cool-space",
...     repo_type="space",
... )
```

Use the `allow_patterns` and `ignore_patterns` arguments to specify which files to upload. These parameters accept either a single pattern or a list of patterns.
Patterns are Standard Wildcards (globbing patterns) as documented [here](https://tldp.org/LDP/GNU-Linux-Tools-Summary/html/x11655.htm).
If both `allow_patterns` and `ignore_patterns` are provided, both constraints apply. By default, all files from the folder are uploaded.

Any `.git/` folder present in any subdirectory will be ignored. However, please be aware that the `.gitignore` file is not taken into account.
This means you must use `allow_patterns` and `ignore_patterns` to specify which files to upload instead.

```py
>>> api.upload_folder(
...     folder_path="/path/to/local/folder",
...     path_in_repo="my-dataset/train", # Upload to a specific folder
...     repo_id="username/test-dataset",
...     repo_type="dataset",
...     ignore_patterns="**/logs/*.txt", # Ignore all text logs
... )
```

You can also use the `delete_patterns` argument to specify files you want to delete from the repo in the same commit.
This can prove useful if you want to clean a remote folder before pushing files in it and you don't know which files
already exists.

The example below uploads the local `./logs` folder to the remote `/experiment/logs/` folder. Only txt files are uploaded
but before that, all previous logs on the repo on deleted. All of this in a single commit.
```py
>>> api.upload_folder(
...     folder_path="/path/to/local/folder/logs",
...     repo_id="username/trained-model",
...     path_in_repo="experiment/logs/",
...     allow_patterns="*.txt", # Upload all local text files
...     delete_patterns="*.txt", # Delete all remote text files before
... )
```

## Upload from the CLI

You can use the `huggingface-cli upload` command from the terminal to directly upload files to the Hub. Internally
it uses the same [`upload_file`] and [`upload_folder`] helpers described above.

You can either upload a single file or an entire folder:

```bash
# Usage:  huggingface-cli upload [repo_id] [local_path] [path_in_repo]
>>> huggingface-cli upload Wauplin/my-cool-model ./models/model.safetensors model.safetensors
https://huggingface.co/Wauplin/my-cool-model/blob/main/model.safetensors

>>> huggingface-cli upload Wauplin/my-cool-model ./models .
https://huggingface.co/Wauplin/my-cool-model/tree/main
```

`local_path` and `path_in_repo` are optional and can be implicitly inferred. If `local_path` is not set, the tool will
check if a local folder or file has the same name as the `repo_id`. If that's the case, its content will be uploaded.
Otherwise, an exception is raised asking the user to explicitly set `local_path`. In any case, if `path_in_repo` is not
set, files are uploaded at the root of the repo.

```bash
# Upload file at root
huggingface-cli upload my-cool-model model.safetensors

# Upload directory at root
huggingface-cli upload my-cool-model ./models

# Upload `my-cool-model/` directory if it exist, raise otherwise
huggingface-cli upload my-cool-model
```

By default, the token saved locally (using `huggingface-cli login`) will be used. If you want to authenticate explicitly,
use the `--token` option:

```bash
huggingface-cli upload my-cool-model --token=hf_****
```

When uploading a folder, you can use the `--include` and `--exclude` arguments to filter the files to upload. You can
also use `--delete` to delete existing files on the Hub.

```bash
# Sync local Space with Hub (upload new files except from logs/, delete removed files)
huggingface-cli upload Wauplin/space-example --repo-type=space --exclude="/logs/*" --delete="*" --commit-message="Sync local Space with Hub"
```

Finally, you can also schedule a job that will upload your files regularly (see [scheduled uploads](#scheduled-uploads)).

```bash
# Upload new logs every 10 minutes
huggingface-cli upload training-model logs/ --every=10
```

## Advanced features

In most cases, you won't need more than [`upload_file`] and [`upload_folder`] to upload your files to the Hub.
However, `huggingface_hub` has more advanced features to make things easier. Let's have a look at them!


### Non-blocking uploads

In some cases, you want to push data without blocking your main thread. This is particularly useful to upload logs and
artifacts while continuing a training. To do so, you can use the `run_as_future` argument in both [`upload_file`] and
[`upload_folder`]. This will return a [`concurrent.futures.Future`](https://docs.python.org/3/library/concurrent.futures.html#future-objects)
object that you can use to check the status of the upload.

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> future = api.upload_folder( # Upload in the background (non-blocking action)
...     repo_id="username/my-model",
...     folder_path="checkpoints-001",
...     run_as_future=True,
... )
>>> future
Future(...)
>>> future.done()
False
>>> future.result() # Wait for the upload to complete (blocking action)
...
```

<Tip>

Background jobs are queued when using `run_as_future=True`. This means that you are guaranteed that the jobs will be
executed in the correct order.

</Tip>

Even though background jobs are mostly useful to upload data/create commits, you can queue any method you like using
[`run_as_future`]. For instance, you can use it to create a repo and then upload data to it in the background. The
built-in `run_as_future` argument in upload methods is just an alias around it.

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.run_as_future(api.create_repo, "username/my-model", exists_ok=True)
Future(...)
>>> api.upload_file(
...     repo_id="username/my-model",
...     path_in_repo="file.txt",
...     path_or_fileobj=b"file content",
...     run_as_future=True,
... )
Future(...)
```

### Upload a folder by chunks

[`upload_folder`] makes it easy to upload an entire folder to the Hub. However, for large folders (thousands of files or
hundreds of GB), it can still be challenging. If you have a folder with a lot of files, you might want to upload
it in several commits. If you experience an error or a connection issue during the upload, you would not have to resume
the process from the beginning.

To upload a folder in multiple commits, just pass `multi_commits=True` as argument. Under the hood, `huggingface_hub`
will list the files to upload/delete and split them in several commits. The "strategy" (i.e. how to split the commits)
is based on the number and size of the files to upload. A PR is open on the Hub to push all the commits. Once the PR is
ready, the commits are squashed into a single commit. If the process is interrupted before completing, you can rerun
your script to resume the upload. The created PR will be automatically detected and the upload will resume from where
it stopped. It is recommended to pass `multi_commits_verbose=True` to get a better understanding of the upload and its
progress.

The example below will upload the checkpoints folder to a dataset in multiple commits. A PR will be created on the Hub
and merged automatically once the upload is complete. If you prefer the PR to stay open and review it manually, you can
pass `create_pr=True`. 

```py
>>> upload_folder(
...     folder_path="local/checkpoints",
...     repo_id="username/my-dataset",
...     repo_type="dataset",
...     multi_commits=True,
...     multi_commits_verbose=True,
... )
```

If you want a better control on the upload strategy (i.e. the commits that are created), you can have a look at the
low-level [`plan_multi_commits`] and [`create_commits_on_pr`] methods.

<Tip warning={true}>

`multi_commits` is still an experimental feature. Its API and behavior is subject to change in the future without prior
notice.

</Tip>

### Scheduled uploads

The Hugging Face Hub makes it easy to save and version data. However, there are some limitations when updating the same file thousands of times. For instance, you might want to save logs of a training process or user
feedback on a deployed Space. In these cases, uploading the data as a dataset on the Hub makes sense, but it can be hard to do properly. The main reason is that you don't want to version every update of your data because it'll make the git repository unusable. The [`CommitScheduler`] class offers a solution to this problem.

The idea is to run a background job that regularly pushes a local folder to the Hub. Let's assume you have a
Gradio Space that takes as input some text and generates two translations of it. Then, the user can select their preferred translation. For each run, you want to save the input, output, and user preference to analyze the results. This is a
perfect use case for [`CommitScheduler`]; you want to save data to the Hub (potentially millions of user feedback), but
you don't _need_ to save in real-time each user's input. Instead, you can save the data locally in a JSON file and
upload it every 10 minutes. For example:

```py
>>> import json
>>> import uuid
>>> from pathlib import Path
>>> import gradio as gr
>>> from huggingface_hub import CommitScheduler

# Define the file where to save the data. Use UUID to make sure not to overwrite existing data from a previous run.
>>> feedback_file = Path("user_feedback/") / f"data_{uuid.uuid4()}.json"
>>> feedback_folder = feedback_file.parent

# Schedule regular uploads. Remote repo and local folder are created if they don't already exist.
>>> scheduler = CommitScheduler(
...     repo_id="report-translation-feedback",
...     repo_type="dataset",
...     folder_path=feedback_folder,
...     path_in_repo="data",
...     every=10,
... )

# Define the function that will be called when the user submits its feedback (to be called in Gradio)
>>> def save_feedback(input_text:str, output_1: str, output_2:str, user_choice: int) -> None:
...     """
...     Append input/outputs and user feedback to a JSON Lines file using a thread lock to avoid concurrent writes from different users.
...     """
...     with scheduler.lock:
...         with feedback_file.open("a") as f:
...             f.write(json.dumps({"input": input_text, "output_1": output_1, "output_2": output_2, "user_choice": user_choice}))
...             f.write("\n")

# Start Gradio
>>> with gr.Blocks() as demo:
>>>     ... # define Gradio demo + use `save_feedback`
>>> demo.launch()
```

And that's it! User input/outputs and feedback will be available as a dataset on the Hub. By using a unique JSON file name, you are guaranteed you won't overwrite data from a previous run or data from another
Spaces/replicas pushing concurrently to the same repository.

For more details about the [`CommitScheduler`], here is what you need to know:
- **append-only:**
    It is assumed that you will only add content to the folder. You must only append data to existing files or create
    new files. Deleting or overwriting a file might corrupt your repository.
- **git history**:
    The scheduler will commit the folder every `every` minutes. To avoid polluting the git repository too much, it is
    recommended to set a minimal value of 5 minutes. Besides, the scheduler is designed to avoid empty commits. If no
    new content is detected in the folder, the scheduled commit is dropped.
- **errors:**
    The scheduler run as background thread. It is started when you instantiate the class and never stops. In particular,
    if an error occurs during the upload (example: connection issue), the scheduler will silently ignore it and retry
    at the next scheduled commit.
- **thread-safety:**
    In most cases it is safe to assume that you can write to a file without having to worry about a lock file. The
    scheduler will not crash or be corrupted if you write content to the folder while it's uploading. In practice,
    _it is possible_ that concurrency issues happen for heavy-loaded apps. In this case, we advice to use the
    `scheduler.lock` lock to ensure thread-safety. The lock is blocked only when the scheduler scans the folder for
    changes, not when it uploads data. You can safely assume that it will not affect the user experience on your Space.

#### Space persistence demo

Persisting data from a Space to a Dataset on the Hub is the main use case for [`CommitScheduler`]. Depending on the use
case, you might want to structure your data differently. The structure has to be robust to concurrent users and
restarts which often implies generating UUIDs. Besides robustness, you should upload data in a format readable by the 🤗 Datasets library for later reuse. We created a [Space](https://huggingface.co/spaces/Wauplin/space_to_dataset_saver)
that demonstrates how to save several different data formats (you may need to adapt it for your own specific needs).

#### Custom uploads

[`CommitScheduler`] assumes your data is append-only and should be uploading "as is". However, you
might want to customize the way data is uploaded. You can do that by creating a class inheriting from [`CommitScheduler`]
and overwrite the `push_to_hub` method (feel free to overwrite it any way you want). You are guaranteed it will
be called every `every` minutes in a background thread. You don't have to worry about concurrency and errors but you
must be careful about other aspects, such as pushing empty commits or duplicated data.

In the (simplified) example below, we overwrite `push_to_hub` to zip all PNG files in a single archive to avoid
overloading the repo on the Hub:

```py
class ZipScheduler(CommitScheduler):
    def push_to_hub(self):
        # 1. List PNG files
          png_files = list(self.folder_path.glob("*.png"))
          if len(png_files) == 0:
              return None  # return early if nothing to commit

        # 2. Zip png files in a single archive
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "train.zip"
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zip:
                for png_file in png_files:
                    zip.write(filename=png_file, arcname=png_file.name)

            # 3. Upload archive
            self.api.upload_file(..., path_or_fileobj=archive_path)

        # 4. Delete local png files to avoid re-uploading them later
        for png_file in png_files:
            png_file.unlink()
```

When you overwrite `push_to_hub`, you have access to the attributes of [`CommitScheduler`] and especially:
- [`HfApi`] client: `api`
- Folder parameters: `folder_path` and `path_in_repo`
- Repo parameters: `repo_id`, `repo_type`, `revision`
- The thread lock: `lock`

<Tip>

For more examples of custom schedulers, check out our [demo Space](https://huggingface.co/spaces/Wauplin/space_to_dataset_saver)
containing different implementations depending on your use cases.

</Tip>

### create_commit

The [`upload_file`] and [`upload_folder`] functions are high-level APIs that are generally convenient to use. We recommend
trying these functions first if you don't need to work at a lower level. However, if you want to work at a commit-level,
you can use the [`create_commit`] function directly.

There are two types of operations supported by [`create_commit`]:

- [`CommitOperationAdd`] uploads a file to the Hub. If the file already exists, the file contents are overwritten. This operation accepts two arguments:

  - `path_in_repo`: the repository path to upload a file to.
  - `path_or_fileobj`: either a path to a file on your filesystem or a file-like object. This is the content of the file to upload to the Hub.

- [`CommitOperationDelete`] removes a file or a folder from a repository. This operation accepts `path_in_repo` as an argument.

- [`CommitOperationCopy`] copies a file within a repository. This operation accepts three arguments:

  - `src_path_in_repo`: the repository path of the file to copy.
  - `path_in_repo`: the repository path where the file should be copied.
  - `src_revision`: optional - the revision of the file to copy if your want to copy a file from a different branch/revision.

For example, if you want to upload two files and delete a file in a Hub repository:

1. Use the appropriate `CommitOperation` to add or delete a file and to delete a folder:

```py
>>> from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
>>> api = HfApi()
>>> operations = [
...     CommitOperationAdd(path_in_repo="LICENSE.md", path_or_fileobj="~/repo/LICENSE.md"),
...     CommitOperationAdd(path_in_repo="weights.h5", path_or_fileobj="~/repo/weights-final.h5"),
...     CommitOperationDelete(path_in_repo="old-weights.h5"),
...     CommitOperationDelete(path_in_repo="logs/"),
...     CommitOperationCopy(src_path_in_repo="image.png", path_in_repo="duplicate_image.png"),
... ]
```

2. Pass your operations to [`create_commit`]:

```py
>>> api.create_commit(
...     repo_id="lysandre/test-model",
...     operations=operations,
...     commit_message="Upload my model weights and license",
... )
```

In addition to [`upload_file`] and [`upload_folder`], the following functions also use [`create_commit`] under the hood:

- [`delete_file`] deletes a single file from a repository on the Hub.
- [`delete_folder`] deletes an entire folder from a repository on the Hub.
- [`metadata_update`] updates a repository's metadata.

For more detailed information, take a look at the [`HfApi`] reference.

## Tips and tricks for large uploads

There are some limitations to be aware of when dealing with a large amount of data in your repo. Given the time it takes to stream the data,
getting an upload/push to fail at the end of the process or encountering a degraded experience, be it on hf.co or when working locally, can be very annoying.
We gathered a list of tips and recommendations for structuring your repo.


| Characteristic     | Recommended        | Tips                                                   |
| ----------------   | ------------------ | ------------------------------------------------------ |
| Repo size          | -                  | contact us for large repos (TBs of data)               |
| Files per repo     | <100k              | merge data into fewer files                            |
| Entries per folder | <10k               | use subdirectories in repo                             |
| File size          | <5GB               | split data into chunked files                          |
| Commit size        | <100 files*        | upload files in multiple commits                       |
| Commits per repo   | -                  | upload multiple files per commit and/or squash history |

_* Not relevant when using `git` CLI directly_

Please read the next section to understand better those limits and how to deal with them.

### Hub repository size limitations

What are we talking about when we say "large uploads", and what are their associated limitations? Large uploads can be
very diverse, from repositories with a few huge files (e.g. model weights) to repositories with thousands of small files
(e.g. an image dataset).

Under the hood, the Hub uses Git to version the data, which has structural implications on what you can do in your repo.
If your repo is crossing some of the numbers mentioned in the previous section, **we strongly encourage you to check out [`git-sizer`](https://github.com/github/git-sizer)**,
which has very detailed documentation about the different factors that will impact your experience. Here is a TL;DR of factors to consider:

- **Repository size**: The total size of the data you're planning to upload. There is no hard limit on a Hub repository size. However, if you plan to upload hundreds of GBs or even TBs of data, we would appreciate it if you could let us know in advance so we can better help you if you have any questions during the process. You can contact us at datasets@huggingface.co or on [our Discord](http://hf.co/join/discord).
- **Number of files**:
    - For optimal experience, we recommend keeping the total number of files under 100k. Try merging the data into fewer files if you have more.
      For example, json files can be merged into a single jsonl file, or large datasets can be exported as Parquet files.
    - The maximum number of files per folder cannot exceed 10k files per folder. A simple solution is to
      create a repository structure that uses subdirectories. For example, a repo with 1k folders from `000/` to `999/`, each containing at most 1000 files, is already enough.
- **File size**: In the case of uploading large files (e.g. model weights), we strongly recommend splitting them **into chunks of around 5GB each**.
There are a few reasons for this:
    - Uploading and downloading smaller files is much easier both for you and the other users. Connection issues can always
      happen when streaming data and smaller files avoid resuming from the beginning in case of errors.
    - Files are served to the users using CloudFront. From our experience, huge files are not cached by this service
      leading to a slower download speed.
In all cases no single LFS file will be able to be >50GB. I.e. 50GB is the hard limit for single file size.
- **Number of commits**: There is no hard limit for the total number of commits on your repo history. However, from
our experience, the user experience on the Hub starts to degrade after a few thousand commits. We are constantly working to
improve the service, but one must always remember that a git repository is not meant to work as a database with a lot of
writes. If your repo's history gets very large, it is always possible to squash all the commits to get a
fresh start using [`super_squash_history`]. This is a non-revertible operation.
- **Number of operations per commit**: Once again, there is no hard limit here. When a commit is uploaded on the Hub, each
git operation (addition or delete) is checked by the server. When a hundred LFS files are committed at once,
each file is checked individually to ensure it's been correctly uploaded. When pushing data through HTTP with `huggingface_hub`,
a timeout of 60s is set on the request, meaning that if the process takes more time, an error is raised
client-side. However, it can happen (in rare cases) that even if the timeout is raised client-side, the process is still
completed server-side. This can be checked manually by browsing the repo on the Hub. To prevent this timeout, we recommend
adding around 50-100 files per commit.

### Practical tips

Now that we've seen the technical aspects you must consider when structuring your repository, let's see some practical
tips to make your upload process as smooth as possible.

- **Start small**: We recommend starting with a small amount of data to test your upload script. It's easier to iterate
on a script when failing takes only a little time.
- **Expect failures**: Streaming large amounts of data is challenging. You don't know what can happen, but it's always
best to consider that something will fail at least once -no matter if it's due to your machine, your connection, or our
servers. For example, if you plan to upload a large number of files, it's best to keep track locally of which files you
already uploaded before uploading the next batch. You are ensured that an LFS file that is already committed will never
be re-uploaded twice but checking it client-side can still save some time.
- **Use `hf_transfer`**: this is a Rust-based [library](https://github.com/huggingface/hf_transfer) meant to speed up
uploads on machines with very high bandwidth. To use it, you must install it (`pip install hf_transfer`) and enable it
by setting `HF_HUB_ENABLE_HF_TRANSFER=1` as an environment variable. You can then use `huggingface_hub` normally.
Disclaimer: this is a power user tool. It is tested and production-ready but lacks user-friendly features like progress
bars or advanced error handling.

## (legacy) Upload files with Git LFS

All the methods described above use the Hub's API to upload files. This is the recommended way to upload files to the Hub.
However, we also provide [`Repository`], a wrapper around the git tool to manage a local repository.

<Tip warning={true}>

Although [`Repository`] is not formally deprecated, we recommend using the HTTP-based methods described above instead.
For more details about this recommendation, please have a look at [this guide](../concepts/git_vs_http) explaining the
core differences between HTTP-based and Git-based approaches.

</Tip>

Git LFS automatically handles files larger than 10MB. But for very large files (>5GB), you need to install a custom transfer agent for Git LFS:

```bash
huggingface-cli lfs-enable-largefiles
```

You should install this for each repository that has a very large file. Once installed, you'll be able to push files larger than 5GB.

### commit context manager

The `commit` context manager handles four of the most common Git commands: pull, add, commit, and push. `git-lfs` automatically tracks any file larger than 10MB. In the following example, the `commit` context manager:

1. Pulls from the `text-files` repository.
2. Adds a change made to `file.txt`.
3. Commits the change.
4. Pushes the change to the `text-files` repository.

```python
>>> from huggingface_hub import Repository
>>> with Repository(local_dir="text-files", clone_from="<user>/text-files").commit(commit_message="My first file :)"):
...     with open("file.txt", "w+") as f:
...         f.write(json.dumps({"hey": 8}))
```

Here is another example of how to use the `commit` context manager to save and upload a file to a repository:

```python
>>> import torch
>>> model = torch.nn.Transformer()
>>> with Repository("torch-model", clone_from="<user>/torch-model", token=True).commit(commit_message="My cool model :)"):
...     torch.save(model.state_dict(), "model.pt")
```

Set `blocking=False` if you would like to push your commits asynchronously. Non-blocking behavior is helpful when you want to continue running your script while your commits are being pushed.

```python
>>> with repo.commit(commit_message="My cool model :)", blocking=False)
```

You can check the status of your push with the `command_queue` method:

```python
>>> last_command = repo.command_queue[-1]
>>> last_command.status
```

Refer to the table below for the possible statuses:

| Status   | Description                          |
| -------- | ------------------------------------ |
| -1       | The push is ongoing.                 |
| 0        | The push has completed successfully. |
| Non-zero | An error has occurred.               |

When `blocking=False`, commands are tracked, and your script will only exit when all pushes are completed, even if other errors occur in your script. Some additional useful commands for checking the status of a push include:

```python
# Inspect an error.
>>> last_command.stderr

# Check whether a push is completed or ongoing.
>>> last_command.is_done

# Check whether a push command has errored.
>>> last_command.failed
```

### push_to_hub

The [`Repository`] class has a [`~Repository.push_to_hub`] function to add files, make a commit, and push them to a repository. Unlike the `commit` context manager, you'll need to pull from a repository first before calling [`~Repository.push_to_hub`].

For example, if you've already cloned a repository from the Hub, then you can initialize the `repo` from the local directory:

```python
>>> from huggingface_hub import Repository
>>> repo = Repository(local_dir="path/to/local/repo")
```

Update your local clone with [`~Repository.git_pull`] and then push your file to the Hub:

```py
>>> repo.git_pull()
>>> repo.push_to_hub(commit_message="Commit my-awesome-file to the Hub")
```

However, if you aren't ready to push a file yet, you can use [`~Repository.git_add`] and [`~Repository.git_commit`] to only add and commit your file:

```py
>>> repo.git_add("path/to/file")
>>> repo.git_commit(commit_message="add my first model config file :)")
```

When you're ready, push the file to your repository with [`~Repository.git_push`]:

```py
>>> repo.git_push()
```
