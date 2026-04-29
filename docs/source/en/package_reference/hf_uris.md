<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# HF URIs

A *HF URI* is a URI-like string that identifies a location on the Hugging Face Hub. Throughout the library and the CLI, `hf://...` strings are used to point at:

- a model, dataset, space or kernel repository (optionally pinned at a revision);
- a file or sub-folder inside such a repository;
- a [bucket](../guides/buckets) or a sub-folder inside a bucket.

A *HF mount* wraps a HF URI with a local mount path and an optional `:ro` / `:rw` flag, used by [Spaces](../guides/manage-spaces) and [Jobs](../guides/jobs) volumes.

This page documents the canonical syntax of HF URIs and HF mounts. The same parser is used everywhere in the library, so a URI that is valid in one context (e.g. [`HfFileSystem`]) is parsed identically in another.

## HF URI syntax

```text
hf://[<TYPE>/]<ID>[@<REVISION>][/<PATH>]
```

| Component       | Required | Allowed values                                                      |
| --------------- | -------- | ------------------------------------------------------------------- |
| `hf://`         | yes      | Literal protocol prefix.                                            |
| `<TYPE>/`       | no       | `models/`, `datasets/`, `spaces/`, `kernels/`, `buckets/` (plural). |
| `<ID>`          | yes      | `<namespace>/<name>`                                                |
| `@<REVISION>`   | no       | Branch, tag, commit SHA, or special ref (`refs/pr/N`, `refs/convert/...`). Repos only. |
| `/<PATH>`       | no       | Path inside the repo or bucket.                                     |

## HF mount syntax

```text
hf://[<TYPE>/]<ID>[@<REVISION>][/<PATH>]:<MOUNT_PATH>[:ro|:rw]
```

A mount is a HF URI followed by `:<MOUNT_PATH>` and an optional `:ro` / `:rw` flag.

| Component       | Required | Allowed values                                                      |
| --------------- | -------- | ------------------------------------------------------------------- |
| `<MOUNT_PATH>`  | yes      | Absolute mount path (must start with `/`).                          |
| `:ro` / `:rw`   | no       | Read-only / read-write flag.                                        |

## What is a HF URI

The following are **all valid HF URIs**:

```text
# Models (type prefix is optional, but the id is always 'namespace/name')
hf://my-org/my-model                            # implicit type prefix
hf://models/my-org/my-model                     # explicit type prefix
hf://models/my-org/my-model/config.json         # file inside a model repo
hf://models/my-org/my-model@v1.0/config.json    # pinned to a revision

# Datasets, Spaces, Kernels (type prefix is required)
hf://datasets/my-org/my-dataset
hf://datasets/my-org/my-dataset@dev/train.csv
hf://spaces/my-user/my-space
hf://kernels/my-org/my-kernel

# Special revisions (preserved as-is)
hf://datasets/my-org/my-dataset@refs/pr/10/data.csv
hf://datasets/my-org/my-dataset@refs/convert/parquet/data.parquet

# Buckets (always 'namespace/name', no revision)
hf://buckets/my-org/my-bucket
hf://buckets/my-org/my-bucket/sub/folder
```

The following are **valid HF mounts** (volume specifications):

```text
hf://my-org/my-model:/data
hf://datasets/my-org/my-dataset:/mnt:ro
hf://datasets/my-org/my-dataset/train:/mnt:rw    # mount a sub-folder
hf://buckets/my-org/my-bucket:/storage:rw
```

## What is **not** a HF URI

The parser is strict on purpose. The following are **rejected**:

| Invalid URI                                       | Reason                                                                  |
| ------------------------------------------------- | ----------------------------------------------------------------------- |
| `my-org/my-model`, `huggingface.co/org/m`         | Missing `hf://` protocol prefix.                                        |
| `hf://dataset/org/m`, `hf://model/org/m`          | Singular type forms are forbidden, use the plural (`datasets/`, ...).   |
| `hf://datasets`, `hf://buckets/`                  | A type prefix alone is not a valid URI, an `<ID>` is required.          |
| `hf://gpt2`, `hf://datasets/squad`                | Canonical repos (without a namespace) are not supported.                |
| `hf://buckets/single-segment`                     | Buckets must always be `namespace/name`.                                |
| `hf://buckets/org/b@v1`                           | Buckets do not support a revision marker.                               |
| `hf://org/m@`, `hf://datasets/foo/bar@/x`         | Empty revision after `@`.                                               |
| `hf://a/b/c@v1`                                   | A repo id must be `namespace/name`, extra segments are paths.           |
| `hf://org/m:/`                                    | Mount path must be a non-empty absolute path.                           |

## Parsing in Python

### Parsing URIs

[`parse_hf_uri`] is the centralized URI parser. It is a pure string parser (no network calls) and returns a frozen [`HfUri`] dataclass.

```python
>>> from huggingface_hub import parse_hf_uri
>>> parse_hf_uri("hf://datasets/my-org/my-dataset@refs/pr/3/train.json")
HfUri(type='dataset', id='my-org/my-dataset', revision='refs/pr/3', path_in_repo='train.json')
```

[`HfUri`] is round-trippable via [`HfUri.to_uri`], which always emits the canonical form (with an explicit type prefix):

```python
>>> uri = parse_hf_uri("hf://my-org/my-model@v1/config.json")
>>> uri.to_uri()
'hf://models/my-org/my-model@v1/config.json'
```

Use the `type` and `id` fields directly. The boolean properties [`is_repo`] and [`is_bucket`] disambiguate between repository URIs and bucket URIs when needed.

### Parsing mounts

[`parse_hf_mount`] parses a mount specification (a HF URI with a local mount path and optional `:ro`/`:rw` flag) and returns a frozen [`HfMount`] dataclass. It uses [`parse_hf_uri`] under the hood.

```python
>>> from huggingface_hub import parse_hf_mount
>>> parse_hf_mount("hf://buckets/my-org/my-bucket/sub/dir:/mnt:ro")
HfMount(source=HfUri(type='bucket', id='my-org/my-bucket', revision=None, path_in_repo='sub/dir'), mount_path='/mnt', read_only=True)
```

[`HfMount`] is round-trippable via [`HfMount.to_uri`]:

```python
>>> mount = parse_hf_mount("hf://my-org/my-model:/data:ro")
>>> mount.to_uri()
'hf://models/my-org/my-model:/data:ro'
```

## Reference

[[autodoc]] huggingface_hub.utils.HfUri

[[autodoc]] huggingface_hub.utils.parse_hf_uri

[[autodoc]] huggingface_hub.utils.HfMount

[[autodoc]] huggingface_hub.utils.parse_hf_mount
