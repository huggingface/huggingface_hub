<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Buckets

Buckets are a flat file storage system on the Hugging Face Hub. Unlike regular repositories (models, datasets, Spaces), buckets have no git history, no versioning, and no branches. They are designed for use cases where you need simple, fast, mutable storage — such as storing training checkpoints, logs, intermediate artifacts, or any large collection of files that doesn't need version control.

You can interact with buckets using the Python API ([`HfApi`]) or the CLI (`hf buckets`). In this guide, we will walk through all the operations available.

> [!TIP]
> All CLI commands are available under `hf buckets <command>`. You can run `hf buckets --help` to see all available subcommands.

## Create and manage buckets

### Create a bucket

Create a bucket with [`create_bucket`]. You need to provide a bucket name. If you don't specify a namespace, the bucket is created under your username.

```py
>>> from huggingface_hub import create_bucket

# Create a bucket under your namespace
>>> url = create_bucket("my-bucket")
>>> url.bucket_id
'username/my-bucket'
>>> url.handle
'hf://buckets/username/my-bucket'

# Create a private bucket
>>> create_bucket("my-private-bucket", private=True)
BucketUrl(...)

# Don't error if bucket already exists
>>> create_bucket("my-bucket", exist_ok=True)
BucketUrl(...)
```

Or via CLI:

```bash
>>> hf buckets create my-bucket
Bucket created: https://huggingface.co/buckets/username/my-bucket (handle: hf://buckets/username/my-bucket)

# Create a private bucket
>>> hf buckets create my-bucket --private

# Don't error if bucket already exists
>>> hf buckets create my-bucket --exist-ok
```

You can also specify the full `namespace/bucket_name` format or the `hf://buckets/namespace/bucket_name` handle:

```bash
>>> hf buckets create my-org/shared-bucket
>>> hf buckets create hf://buckets/my-org/shared-bucket
```

### Get bucket info

Use [`bucket_info`] to get metadata about a bucket, including its visibility, total size, file count, and creation date.

```py
>>> from huggingface_hub import bucket_info
>>> info = bucket_info("username/my-bucket")
>>> info.id
'username/my-bucket'
>>> info.private
False
>>> info.size
551879671
>>> info.total_files
12
>>> info.created_at
datetime.datetime(2026, 2, 6, 17, 37, 57, tzinfo=datetime.timezone.utc)
```

Or via CLI:

```bash
# JSON output
>>> hf buckets info username/my-bucket
{
  "id": "username/my-bucket",
  "private": false,
  ...
}

# Quiet mode: prints only the bucket ID
>>> hf buckets info username/my-bucket --quiet
username/my-bucket
```

### List buckets

Use [`list_buckets`] to list all buckets in a namespace. By default, it lists buckets in the current user's namespace.

```py
>>> from huggingface_hub import list_buckets

# List your own buckets
>>> for bucket in list_buckets():
...     print(bucket.id, bucket.size, bucket.total_files)

# List buckets in an organization
>>> for bucket in list_buckets(namespace="huggingface"):
...     print(bucket.id)
```

Or via CLI:

```bash
# Table format (default)
>>> hf buckets list

# JSON format
>>> hf buckets list --format json

# List buckets in a specific namespace
>>> hf buckets list huggingface

# Quiet mode: prints one bucket ID per line
>>> hf buckets list --quiet
```

### Delete a bucket

Use [`delete_bucket`] to delete a bucket. This operation is irreversible.

```py
>>> from huggingface_hub import delete_bucket
>>> delete_bucket("username/my-bucket")

# Don't error if bucket doesn't exist
>>> delete_bucket("username/my-bucket", missing_ok=True)
```

Or via CLI:

```bash
# Prompts for confirmation
>>> hf buckets delete username/my-bucket

# Skip confirmation
>>> hf buckets delete username/my-bucket --yes

# Don't error if bucket doesn't exist
>>> hf buckets delete username/my-bucket --yes --missing-ok
```

## Browse bucket contents

### List files

Use [`list_bucket_tree`] to list files and directories in a bucket. By default, it lists only the top-level entries.

```py
>>> from huggingface_hub import list_bucket_tree

# List top-level entries
>>> for item in list_bucket_tree("username/my-bucket"):
...     print(item.type, item.path)
file file.txt
file big.bin
directory sub

# List all files recursively
>>> for item in list_bucket_tree("username/my-bucket", recursive=True):
...     if item.type == "file":
...         print(item.path, item.size)
file.txt 5
big.bin 2048
sub/nested.txt 14
sub/deep/file.txt 4

# Filter by prefix
>>> for item in list_bucket_tree("username/my-bucket", prefix="sub"):
...     print(item.path)
```

Or via CLI, with support for table, human-readable, and ASCII tree formats:

```bash
# Default table format
>>> hf buckets tree username/my-bucket

# Recursive listing
>>> hf buckets tree username/my-bucket -R

# Human-readable sizes
>>> hf buckets tree username/my-bucket -h -R

# ASCII tree format
>>> hf buckets tree username/my-bucket --tree -R
├── big.bin
├── file.txt
└── sub/
    ├── deep/
    │   └── file.txt
    └── nested.txt

# Filter by prefix
>>> hf buckets tree username/my-bucket/sub -R
```

> [!TIP]
> The `hf buckets tree` command accepts both short format (`username/my-bucket/sub`) and full handle (`hf://buckets/username/my-bucket/sub`) as arguments.

## Upload files

### Upload with Python

Use [`batch_bucket_files`] to upload files to a bucket. You can upload from local file paths or from raw bytes:

```py
>>> from huggingface_hub import batch_bucket_files

# Upload from local file paths
>>> batch_bucket_files(
...     "username/my-bucket",
...     add=[
...         ("./model.safetensors", "models/model.safetensors"),
...         ("./config.json", "models/config.json"),
...     ],
... )

# Upload from raw bytes
>>> batch_bucket_files(
...     "username/my-bucket",
...     add=[
...         (b'{"key": "value"}', "config.json"),
...     ],
... )

# Upload and delete in one batch
>>> batch_bucket_files(
...     "username/my-bucket",
...     add=[("./new-model.safetensors", "model.safetensors")],
...     delete=["old-model.bin"],
... )
```

> [!WARNING]
> [`batch_bucket_files`] is non-transactional. If an error occurs during the process, some files may have been uploaded or deleted while others haven't.

### Upload a single file with the CLI

Use `hf buckets cp` to upload a single file:

```bash
# Upload to bucket root (uses local filename as remote name)
>>> hf buckets cp ./config.json hf://buckets/username/my-bucket

# Upload to a subdirectory
>>> hf buckets cp ./data.csv hf://buckets/username/my-bucket/logs/

# Upload with a different remote filename
>>> hf buckets cp ./model.safetensors hf://buckets/username/my-bucket/v2/model.safetensors

# Upload from stdin
>>> echo "hello" | hf buckets cp - hf://buckets/username/my-bucket/hello.txt
>>> cat model.safetensors | hf buckets cp - hf://buckets/username/my-bucket/model.safetensors
```

### Upload a directory with the CLI

Use `hf buckets sync` to upload an entire local directory to a bucket:

```bash
# Upload a local directory to a bucket
>>> hf buckets sync ./data hf://buckets/username/my-bucket

# Upload to a specific prefix in the bucket
>>> hf buckets sync ./data hf://buckets/username/my-bucket/train
```

See the [Sync directories](#sync-directories) section below for the full set of sync options.

## Download files

### Download with Python

Use [`download_bucket_files`] to download files from a bucket:

```py
>>> from huggingface_hub import download_bucket_files

# Download specific files by path
>>> download_bucket_files(
...     "username/my-bucket",
...     files=[
...         ("models/model.safetensors", "./local/model.safetensors"),
...         ("config.json", "./local/config.json"),
...     ],
... )
```

For better performance, you can pass [`BucketFile`] objects obtained from [`list_bucket_tree`] instead of string paths. This skips the metadata fetching step:

```py
>>> from huggingface_hub import list_bucket_tree, download_bucket_files

# List and filter files, then download
>>> parquet_files = [
...     item for item in list_bucket_tree("username/my-bucket", recursive=True)
...     if item.type == "file" and item.path.endswith(".parquet")
... ]
>>> download_bucket_files(
...     "username/my-bucket",
...     files=[(f, f"./local/{f.path}") for f in parquet_files],
... )
```

### Download a single file with the CLI

Use `hf buckets cp` to download a single file:

```bash
# Download to a specific file
>>> hf buckets cp hf://buckets/username/my-bucket/config.json ./config.json

# Download to a directory (uses original filename)
>>> hf buckets cp hf://buckets/username/my-bucket/config.json ./data/

# Download to current directory (omit destination)
>>> hf buckets cp hf://buckets/username/my-bucket/config.json

# Download to stdout
>>> hf buckets cp hf://buckets/username/my-bucket/config.json -
```

### Download a directory with the CLI

Use `hf buckets sync` to download all files from a bucket to a local directory:

```bash
# Download bucket contents to a local directory
>>> hf buckets sync hf://buckets/username/my-bucket ./data

# Download only a specific prefix
>>> hf buckets sync hf://buckets/username/my-bucket/models ./local-models
```

## Sync directories

The `hf buckets sync` command (also available as the top-level `hf sync` alias) is the most powerful way to transfer files between a local directory and a bucket. It compares source and destination, and only transfers files that have changed.

### Basic sync

```bash
# Upload: local directory -> bucket
>>> hf buckets sync ./data hf://buckets/username/my-bucket

# Download: bucket -> local directory
>>> hf buckets sync hf://buckets/username/my-bucket ./data
```

> [!TIP]
> `hf sync` is a convenient alias for `hf buckets sync`. Both commands are identical.
> ```bash
> >>> hf sync ./data hf://buckets/username/my-bucket
> ```

### Delete extraneous files

By default, sync only adds or updates files. Use `--delete` to also remove files in the destination that don't exist in the source:

```bash
# Upload and remove remote files not present locally
>>> hf buckets sync ./data hf://buckets/username/my-bucket --delete

# Download and remove local files not present in bucket
>>> hf buckets sync hf://buckets/username/my-bucket ./data --delete
```

### Filtering

You can control which files are synced using include/exclude patterns:

```bash
# Only sync .txt files
>>> hf buckets sync ./data hf://buckets/username/my-bucket --include "*.txt"

# Exclude log files
>>> hf buckets sync ./data hf://buckets/username/my-bucket --exclude "*.log"

# Combine include and exclude
>>> hf buckets sync ./data hf://buckets/username/my-bucket --include "*.safetensors" --exclude "*.tmp"
```

For more complex filtering, use a filter file with `--filter-from`:

```bash
>>> hf buckets sync ./data hf://buckets/username/my-bucket --filter-from filters.txt
```

The filter file uses `+` (include) and `-` (exclude) prefixes. Lines starting with `#` are comments. Rules are evaluated in order — the first matching rule wins:

```text
# filters.txt
- *.log
- *.tmp
+ *.safetensors
+ *.json
```

### Comparison modes

By default, sync compares files using both size and modification time. You can customize this behavior:

```bash
# Only compare sizes (ignore modification times)
>>> hf buckets sync ./data hf://buckets/username/my-bucket --ignore-times

# Only compare modification times (ignore sizes)
>>> hf buckets sync ./data hf://buckets/username/my-bucket --ignore-sizes

# Only update files that already exist on the receiver (skip new files)
>>> hf buckets sync ./data hf://buckets/username/my-bucket --existing

# Only create new files (skip files that already exist on the receiver)
>>> hf buckets sync ./data hf://buckets/username/my-bucket --ignore-existing
```

### Plan and apply

For critical operations, you can review the sync plan before executing it:

```bash
# Step 1: Generate a plan file (nothing is transferred)
>>> hf buckets sync ./data hf://buckets/username/my-bucket --plan sync-plan.jsonl
Sync plan: ./data -> hf://buckets/username/my-bucket
  Uploads: 3
  Downloads: 0
  Deletes: 0
  Skips: 1
Plan saved to: sync-plan.jsonl

# Step 2: Review the plan file (JSONL format)
>>> cat sync-plan.jsonl

# Step 3: Apply the plan
>>> hf buckets sync --apply sync-plan.jsonl
```

> [!TIP]
> The plan file is a JSONL file with a header line followed by one line per operation. Each operation includes the action (`upload`, `download`, `delete`, or `skip`), the file path, and the reason for the action.

### Verbose and quiet modes

```bash
# Show per-file operations
>>> hf buckets sync ./data hf://buckets/username/my-bucket --verbose

# Suppress all output
>>> hf buckets sync ./data hf://buckets/username/my-bucket --quiet
```

## Advanced

For lower-level use cases, the following methods are also available:

- [`get_bucket_paths_info`]: Fetch information about specific paths in a bucket in a single batch request. Useful when you know exactly which files you need metadata for.

```py
>>> from huggingface_hub import get_bucket_paths_info
>>> for info in get_bucket_paths_info("username/my-bucket", ["file.txt", "models/model.safetensors"]):
...     print(info.path, info.size, info.xet_hash)
```

- [`get_bucket_file_metadata`]: Fetch metadata (size and xet data) for a single file. Used internally by [`download_bucket_files`].

```py
>>> from huggingface_hub import get_bucket_file_metadata
>>> metadata = get_bucket_file_metadata("username/my-bucket", "models/model.safetensors")
>>> metadata.size
42000
```
