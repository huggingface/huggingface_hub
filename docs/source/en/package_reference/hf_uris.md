<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# HF URIs

A *HF URI* is a URI-like string that identifies a location on the Hugging Face Hub. Throughout the library and the CLI, `hf://...` strings are used to point at:

- a model, dataset, space or kernel repository (optionally pinned at a revision);
- a file or sub-folder inside such a repository;
- a [bucket](../guides/buckets) or a sub-folder inside a bucket;
- a [Spaces](../guides/manage-spaces) or [Jobs](../guides/jobs) volume to mount,
  with an optional `:ro` / `:rw` flag.

This page documents the canonical syntax of HF URIs. The same parser is used everywhere in the library, so a URI that is valid in one context (e.g. [`HfFileSystem`]) is parsed identically in another (e.g. `hf jobs run -v`).

## Canonical syntax

```text
hf://[<TYPE>/]<ID>[@<REVISION>][/<PATH>][:<MOUNT_PATH>[:ro|:rw]]
```

| Component       | Required | Allowed values                                                      |
| --------------- | -------- | ------------------------------------------------------------------- |
| `hf://`         | yes      | Literal protocol prefix.                                            |
| `<TYPE>/`       | no       | `models/`, `datasets/`, `spaces/`, `kernels/`, `buckets/` (plural). |
| `<ID>`          | yes      | `<repo_id>` for repos, `<namespace>/<name>` for buckets.            |
| `@<REVISION>`   | no       | Branch, tag, commit SHA, or special ref (`refs/pr/N`, `refs/convert/...`). Repos only. |
| `/<PATH>`       | no       | Path inside the repo or bucket.                                     |
| `:<MOUNT_PATH>` | no       | Absolute mount path (volume use case).                              |
| `:ro` / `:rw`   | no       | Read-only / read-write flag (mount URIs only).                      |

## What is a HF URI

The following are **all valid HF URIs**:

```text
# Models (type prefix is optional)
hf://gpt2                                       # canonical model
hf://my-org/my-model                            # namespaced model
hf://models/my-org/my-model                     # explicit type prefix
hf://models/my-org/my-model/config.json         # file inside a model repo
hf://models/my-org/my-model@v1.0/config.json    # pinned to a revision

# Datasets, Spaces, Kernels (type prefix is required)
hf://datasets/squad                             # canonical dataset
hf://datasets/my-org/my-dataset@dev/train.csv
hf://spaces/my-user/my-space
hf://kernels/my-org/my-kernel

# Special revisions (preserved as-is)
hf://datasets/my-org/my-dataset@refs/pr/10/data.csv
hf://datasets/my-org/my-dataset@refs/convert/parquet/data.parquet

# Buckets (always 'namespace/name', no revision)
hf://buckets/my-org/my-bucket
hf://buckets/my-org/my-bucket/sub/folder

# Volume URIs (append `:<MOUNT_PATH>[:ro|:rw]`)
hf://gpt2:/data
hf://datasets/my-org/my-dataset:/mnt:ro
hf://datasets/my-org/my-dataset/train:/mnt:rw    # mount a sub-folder
hf://buckets/my-org/my-bucket:/storage:rw
```

## What is **not** a HF URI

The parser is strict on purpose. The following are **rejected**:

| Invalid URI                               | Reason                                                                  |
| ----------------------------------------- | ----------------------------------------------------------------------- |
| `gpt2`, `huggingface.co/gpt2`             | Missing `hf://` protocol prefix.                                        |
| `hf://dataset/foo/bar`, `hf://model/gpt2` | Singular type forms are forbidden, use the plural (`datasets/`, ...).  |
| `hf://datasets`, `hf://buckets/`          | A type prefix alone is not a valid URI, an `<ID>` is required.         |
| `hf://buckets/single-segment`             | Buckets must always be `namespace/name`.                                |
| `hf://buckets/org/b@v1`                   | Buckets do not support a revision marker.                               |
| `hf://gpt2@`, `hf://datasets/foo/bar@/x`  | Empty revision after `@`.                                               |
| `hf://a/b/c@v1`                           | A repo id can be at most `namespace/name`, extra segments are paths.    |
| `hf://gpt2:ro`, `hf://gpt2:rw`            | `:ro`/`:rw` requires a mount path (`hf://gpt2:/data:ro`).               |
| `hf://gpt2:/`                             | Mount path must be a non-empty absolute path.                           |

## Parsing in Python

[`parse_hf_uri`] is the centralized parser. It is a pure string parser (no network calls) and returns a frozen [`HfUri`] dataclass.

```python
>>> from huggingface_hub import parse_hf_uri
>>> parse_hf_uri("hf://datasets/squad@refs/pr/3/train.json")
HfUri(type='dataset', id='squad', revision='refs/pr/3', path_in_repo='train.json', mount_path=None, read_only=None)

>>> parse_hf_uri("hf://buckets/my-org/my-bucket/sub/dir:/mnt:ro")
HfUri(type='bucket', id='my-org/my-bucket', revision=None, path_in_repo='sub/dir', mount_path='/mnt', read_only=True)
```

[`HfUri`] is round-trippable via [`HfUri.to_uri`], which always emits the canonical form (with an explicit type prefix):

```python
>>> uri = parse_hf_uri("hf://gpt2@v1/config.json")
>>> uri.to_uri()
'hf://models/gpt2@v1/config.json'
```

Use the `type` and `id` fields directly. The boolean properties [`is_repo`] and [`is_bucket`] disambiguate between repository URIs and bucket URIs when needed.

## Reference

[[autodoc]] huggingface_hub.utils.HfUri

[[autodoc]] huggingface_hub.utils.parse_hf_uri
