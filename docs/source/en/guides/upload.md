<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Upload files to the Hub

Sharing your files and work is an important aspect of the Hub. The `huggingface_hub` offers several options for uploading your files to the Hub. You can use these functions independently or integrate them into your library, making it more convenient for your users to interact with the Hub.

Whenever you want to upload files to the Hub, you need to log in to your Hugging Face account. For more details about authentication, check out [this section](../quick-start#authentication).

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

By default, the `.gitignore` file will be taken into account to know which files should be committed or not. By default we check if a `.gitignore` file is present in a commit, and if not, we check if it exists on the Hub. Please be aware that only a `.gitignore` file present at the root of the directory will be used. We do not check for `.gitignore` files in subdirectories.

If you don't want to use an hardcoded `.gitignore` file, you can use the `allow_patterns` and `ignore_patterns` arguments to filter which files to upload. These parameters accept either a single pattern or a list of patterns. Patterns are Standard Wildcards (globbing patterns) as documented [here](https://tldp.org/LDP/GNU-Linux-Tools-Summary/html/x11655.htm). If both `allow_patterns` and `ignore_patterns` are provided, both constraints apply.

Beside the `.gitignore` file and allow/ignore patterns, any `.git/` folder present in any subdirectory will be ignored.

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

You can use the `hf upload` command from the terminal to directly upload files to the Hub. Internally it uses the same [`upload_file`] and [`upload_folder`] helpers described above.

You can either upload a single file or an entire folder:

```bash
# Usage:  hf upload [repo_id] [local_path] [path_in_repo]
>>> hf upload Wauplin/my-cool-model ./models/model.safetensors model.safetensors
https://huggingface.co/Wauplin/my-cool-model/blob/main/model.safetensors

>>> hf upload Wauplin/my-cool-model ./models .
https://huggingface.co/Wauplin/my-cool-model/tree/main
```

`local_path` and `path_in_repo` are optional and can be implicitly inferred. If `local_path` is not set, the tool will
check if a local folder or file has the same name as the `repo_id`. If that's the case, its content will be uploaded.
Otherwise, an exception is raised asking the user to explicitly set `local_path`. In any case, if `path_in_repo` is not
set, files are uploaded at the root of the repo.

For more details about the CLI upload command, please refer to the [CLI guide](./cli#hf-upload).

## Upload a large folder

In most cases, the [`upload_folder`] method and `hf upload` command should be the go-to solutions to upload files to the Hub. They ensure a single commit will be made, handle a lot of use cases, and fail explicitly when something wrong happens. However, when dealing with a large amount of data, you will usually prefer a resilient process even if it leads to more commits or requires more CPU usage. The [`upload_large_folder`] method has been implemented in that spirit:
- it is resumable: the upload process is split into many small tasks (hashing files, pre-uploading them, and committing them). Each time a task is completed, the result is cached locally in a `./cache/huggingface` folder inside the folder you are trying to upload. By doing so, restarting the process after an interruption will resume all completed tasks.
- it is multi-threaded: hashing large files and pre-uploading them benefits a lot from multithreading if your machine allows it.
- it is resilient to errors: a high-level retry-mechanism has been added to retry each independent task indefinitely until it passes (no matter if it's a OSError, ConnectionError, PermissionError, etc.). This mechanism is double-edged. If transient errors happen, the process will continue and retry. If permanent errors happen (e.g. permission denied), it will retry indefinitely without solving the root cause.

If you want more technical details about how `upload_large_folder` is implemented under the hood, please have a look to the [`upload_large_folder`] package reference.

Here is how to use [`upload_large_folder`] in a script. The method signature is very similar to [`upload_folder`]:

```py
>>> api.upload_large_folder(
...     repo_id="HuggingFaceM4/Docmatix",
...     repo_type="dataset",
...     folder_path="/path/to/local/docmatix",
... )
```

You will see the following output in your terminal:
```
Repo created: https://huggingface.co/datasets/HuggingFaceM4/Docmatix
Found 5 candidate files to upload
Recovering from metadata files: 100%|█████████████████████████████████████| 5/5 [00:00<00:00, 542.66it/s]

---------- 2024-07-22 17:23:17 (0:00:00) ----------
Files:   hashed 5/5 (5.0G/5.0G) | pre-uploaded: 0/5 (0.0/5.0G) | committed: 0/5 (0.0/5.0G) | ignored: 0
Workers: hashing: 0 | get upload mode: 0 | pre-uploading: 5 | committing: 0 | waiting: 11
---------------------------------------------------
```

First, the repo is created if it didn't exist before. Then, the local folder is scanned for files to upload. For each file, we try to recover metadata information (from a previously interrupted upload). From there, it is able to launch workers and print an update status every 1 minute. Here, we can see that 5 files have already been hashed but not pre-uploaded. 5 workers are pre-uploading files while the 11 others are waiting for a task.

A command line is also provided. You can define the number of workers and the level of verbosity in the terminal:

```sh
hf upload-large-folder HuggingFaceM4/Docmatix --repo-type=dataset /path/to/local/docmatix --num-workers=16
```

> [!TIP]
> For large uploads, you have to set `repo_type="model"` or `--repo-type=model` explicitly. Usually, this information is implicit in all other `HfApi` methods. This is to avoid having data uploaded to a repository with a wrong type. If that's the case, you'll have to re-upload everything.

> [!WARNING]
> While being much more robust to upload large folders, `upload_large_folder` is more limited than [`upload_folder`] feature-wise. In practice:
> - you cannot set a custom `path_in_repo`. If you want to upload to a subfolder, you need to set the proper structure locally.
> - you cannot set a custom `commit_message` and `commit_description` since multiple commits are created.
> - you cannot delete from the repo while uploading. Please make a separate commit first.
> - you cannot create a PR directly. Please create a PR first (from the UI or using [`create_pull_request`]) and then commit to it by passing `revision`.

### Tips and tricks for large uploads

There are some limitations to be aware of when dealing with a large amount of data in your repo. Given the time it takes to stream the data, getting an upload/push to fail at the end of the process or encountering a degraded experience, be it on hf.co or when working locally, can be very annoying.

Check out our [Repository limitations and recommendations](https://huggingface.co/docs/hub/repositories-recommendations) guide for best practices on how to structure your repositories on the Hub. Let's move on with some practical tips to make your upload process as smooth as possible.

- **Start small**: We recommend starting with a small amount of data to test your upload script. It's easier to iterate on a script when failing takes only a little time.
- **Expect failures**: Streaming large amounts of data is challenging. You don't know what can happen, but it's always best to consider that something will fail at least once -no matter if it's due to your machine, your connection, or our servers. For example, if you plan to upload a large number of files, it's best to keep track locally of which files you already uploaded before uploading the next batch. You are ensured that an LFS file that is already committed will never be re-uploaded twice but checking it client-side can still save some time. This is what [`upload_large_folder`] does for you.
- **Use `hf_xet`**: this leverages the new storage backend for the Hub, is written in Rust, and is now available for everyone to use. In fact, `hf_xet` is already enabled by default when using `huggingface_hub`! For maximum performance, set [`HF_XET_HIGH_PERFORMANCE=1`](../package_reference/environment_variables.md#hf_xet_high_performance) as an environment variable. Be aware that when high performance mode is enabled, the tool will try to use all available bandwidth and CPU cores.

## Advanced features

In most cases, you won't need more than [`upload_file`] and [`upload_folder`] to upload your files to the Hub.
However, `huggingface_hub` has more advanced features to make things easier. Let's have a look at them!

### Faster Uploads

Take advantage of faster uploads through `hf_xet`, the Python binding to the [`xet-core`](https://github.com/huggingface/xet-core) library that enables chunk-based deduplication for faster uploads and downloads. `hf_xet` integrates seamlessly with `huggingface_hub`, but uses the Rust `xet-core` library and Xet storage instead of LFS. 

`hf_xet` uses the Xet storage system, which breaks files down into immutable chunks, storing collections of these chunks (called blocks or xorbs) remotely and retrieving them to reassemble the file when requested. When uploading, after confirming the user is authorized to write to this repo, `hf_xet` will scan the files, breaking them down into their chunks and collecting those chunks into xorbs (and deduplicating across known chunks), and then will be upload these xorbs to the Xet content-addressable service (CAS), which will verify the integrity of the xorbs, register the xorb metadata along with the LFS SHA256 hash (to support lookup/download), and write the xorbs to remote storage.

To enable it, simply install the latest version of `huggingface_hub`:

```bash
pip install -U "huggingface_hub"
```

As of `huggingface_hub` 0.32.0, this will also install `hf_xet`. 

All other `huggingface_hub` APIs will continue to work without any modification. To learn more about the benefits of Xet storage and `hf_xet`, refer to this [section](https://huggingface.co/docs/hub/storage-backends).

**Cluster / Distributed Filesystem Upload Considerations**

When uploading from a cluster, the files being uploaded often reside on a distributed or networked filesystem (NFS, EBS, Lustre, Fsx, etc). Xet storage will chunk those files and write them into blocks (also called xorbs) locally, and once the block is completed will upload them. For better performance when uploading from a distributed filesystem, make sure to set [`HF_XET_CACHE`](../package_reference/environment_variables#hfxetcache) to a directory that is on a local disk (ex. a local NVMe or SSD disk). The default location for the Xet cache is under `HF_HOME` at (`~/.cache/huggingface/xet`) and this being in the user's home directory is often also located on the distributed filesystem.

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

> [!TIP]
> Background jobs are queued when using `run_as_future=True`. This means that you are guaranteed that the jobs will be
> executed in the correct order.

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
hundreds of GB), we recommend using [`upload_large_folder`], which splits the upload into multiple commits. See the [Upload a large folder](#upload-a-large-folder) section for more details.


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

> [!TIP]
> For more examples of custom schedulers, check out our [demo Space](https://huggingface.co/spaces/Wauplin/space_to_dataset_saver)
> containing different implementations depending on your use cases.

### create_commit

The [`upload_file`] and [`upload_folder`] functions are high-level APIs that are generally convenient to use. We recommend
trying these functions first if you don't need to work at a lower level. However, if you want to work at a commit-level,
you can use the [`create_commit`] function directly.

There are three types of operations supported by [`create_commit`]:

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

### Preupload LFS files before commit

In some cases, you might want to upload huge files to S3 **before** making the commit call. For example, if you are
committing a dataset in several shards that are generated in-memory, you would need to upload the shards one by one
to avoid an out-of-memory issue. A solution is to upload each shard as a separate commit on the repo. While being
perfectly valid, this solution has the drawback of potentially messing the git history by generating tens of commits.
To overcome this issue, you can upload your files one by one to S3 and then create a single commit at the end. This
is possible using [`preupload_lfs_files`] in combination with [`create_commit`].

> [!WARNING]
> This is a power-user method. Directly using [`upload_file`], [`upload_folder`] or [`create_commit`] instead of handling
> the low-level logic of pre-uploading files is the way to go in the vast majority of cases. The main caveat of
> [`preupload_lfs_files`] is that until the commit is actually made, the upload files are not accessible on the repo on
> the Hub. If you have a question, feel free to ping us on our Discord or in a GitHub issue.

Here is a simple example illustrating how to pre-upload files:

```py
>>> from huggingface_hub import CommitOperationAdd, preupload_lfs_files, create_commit, create_repo

>>> repo_id = create_repo("test_preupload").repo_id

>>> operations = [] # List of all `CommitOperationAdd` objects that will be generated
>>> for i in range(5):
...     content = ... # generate binary content
...     addition = CommitOperationAdd(path_in_repo=f"shard_{i}_of_5.bin", path_or_fileobj=content)
...     preupload_lfs_files(repo_id, additions=[addition])
...     operations.append(addition)

>>> # Create commit
>>> create_commit(repo_id, operations=operations, commit_message="Commit all shards")
```

First, we create the [`CommitOperationAdd`] objects one by one. In a real-world example, those would contain the
generated shards. Each file is uploaded before generating the next one. During the [`preupload_lfs_files`] step, **the
`CommitOperationAdd` object is mutated**. You should only use it to pass it directly to [`create_commit`]. The main
update of the object is that **the binary content is removed** from it, meaning that it will be garbage-collected if
you don't store another reference to it. This is expected as we don't want to keep in memory the content that is
already uploaded. Finally we create the commit by passing all the operations to [`create_commit`]. You can pass
additional operations (add, delete or copy) that have not been processed yet and they will be handled correctly.
