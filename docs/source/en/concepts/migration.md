# Migrating to huggingface_hub v1.0

The `huggingface_hub` library has undergone significant changes in the v1.0 release. This guide is intended to help you migrate your existing code to the new version.

The v1.0 release is a major milestone for the library. It marks our commitment to API stability and the maturity of the library. We have made several improvements and breaking changes to make the library more robust and easier to use.

This guide is divided into two sections:
- [Main changes](#main-changes): A list of the most important new features and improvements.
- [Breaking changes](#breaking-changes): A comprehensive list of all breaking changes and how to adapt your code.

We hope this guide will help you to migrate to the new version of `huggingface_hub` smoothly. If you have any questions or feedback, please open an issue on the [GitHub repository](https://github.com/huggingface/huggingface_hub/issues).

## Main changes

### HTTPX migration

The `huggingface_hub` library now uses [`httpx`](https://www.python-httpx.org/) instead of `requests` for HTTP requests. This change was made to improve performance and to support asynchronous requests.

This is a major change that affects the entire library. While we have tried to make this change as transparent as possible, you may need to update your code in some cases. Please see the [Breaking changes](#breaking-changes) section for more details.

### Python 3.9+

`huggingface_hub` now requires Python 3.9 or higher. Python 3.8 is no longer supported.

### Built-in generics for type annotations

The library now uses built-in generics for type annotations (e.g. `list` instead of `typing.List`). This is a new feature in Python 3.9 and makes the code cleaner and easier to read.

## Breaking changes

This section lists the breaking changes introduced in v1.0.

### Python 3.8 support dropped

`huggingface_hub` v1.0 drops support for Python 3.8. You will need to upgrade to Python 3.9 or higher to use the new version of the library.

### HTTPX migration

The migration to `httpx` has introduced a few breaking changes.

- **Proxy configuration**: "per method" proxies are no longer supported. Proxies must be configured globally using the `HTTP_PROXY` and `HTTPS_PROXY` environment variables.
- **Custom HTTP backend**: The `configure_http_backend` function has been removed. You can now use `set_client_factory` and `set_async_client_factory` to configure the HTTP client.
- **Error handling**: `requests.HTTPError` is no longer raised. Instead, `httpx.HTTPError` is raised. We recommend catching `HfHubHttpError` which is a subclass of `httpx.HTTPError` and will ensure your code is compatible with both old and new versions of the library.
- **SSLError**: `httpx` does not have the concept of `SSLError`. It is now a generic `httpx.ConnectError`.
- **`LocalEntryNotFoundError`**: This error no longer inherits from `HTTPError`.
- **`InferenceClient`**: The `InferenceClient` can now be used as a context manager. This is especially useful when streaming tokens from a language model to ensure that the connection is closed properly.
- **`AsyncInferenceClient`**: The `trust_env` parameter has been removed from the `AsyncInferenceClient`'s constructor.

### Deprecated features removed

A number of deprecated functions and parameters have been removed in v1.0.

- `hf_cache_home` is removed. Please use `HF_HOME` instead.
- `use_auth_token` is removed. Please use `token` instead.
- `get_token_permission` is removed.
- `update_repo_visibility` is removed. Please use `update_repo_settings` instead.
- `is_write_action` parameter is removed from `build_hf_headers`.
- `write_permission` parameter is removed from `login`.
- `new_session` parameter in `login` is renamed to `skip_if_logged_in`.
- `resume_download`, `force_filename`, and `local_dir_use_symlinks` parameters are removed from `hf_hub_download` and `snapshot_download`.
- `library`, `language`, `tags`, and `task` parameters are removed from `list_models`.

### Return value of `upload_file` and `upload_folder`

The `upload_file` and `upload_folder` functions now return the URL of the commit created on the Hub. Previously, they returned the URL of the file or folder.

### `Repository` class removed

The `Repository` class has been removed in v1.0. This class was a git-based wrapper to manage repositories. The recommended way to interact with the Hub is now to use the HTTP-based functions in the `huggingface_hub` library.

The `Repository` class was mostly a wrapper around the `git` command-line interface. You can still use `git` directly to manage your repositories. However, we recommend using the `huggingface_hub` library's HTTP-based API for a better experience, especially when dealing with large files.

Here is a mapping from the old `Repository` methods to the new functions:

| `Repository` method | New function |
| --- | --- |
| `repo.clone_from(...)` | `snapshot_download(...)` |
| `repo.git_add(...)` | `upload_file(...)` or `upload_folder(...)` |
| `repo.git_commit(...)` | `create_commit(...)` |
| `repo.git_push(...)` | `create_commit(...)` |
| `repo.git_pull(...)` | `snapshot_download(...)` |
| `repo.git_checkout(...)` | `snapshot_download(..., revision=...)` |
| `repo.git_tag(...)` | `create_tag(...)` |
| `repo.git_branch(...)` | `create_branch(...)` |

### `HfFolder` and `InferenceApi` classes removed

The `HfFolder` and `InferenceApi` classes have been removed in v1.0.

- `HfFolder` was used to manage the Hugging Face cache directory and the user's token. It is now recommended to use the following functions instead:
  - [`login`] and [`logout`] to manage the user's token.
  - [`hf_hub_download`] and [`snapshot_download`] to download and cache files from the Hub.

- `InferenceApi` was a class to interact with the Inference API. It is now recommended to use the [`InferenceClient`] class instead.

### TensorFlow support removed

All TensorFlow-related code and dependencies have been removed in v1.0. This includes the following breaking changes:

- The `split_tf_state_dict_into_shards` and `get_tf_storage_size` utility functions have been removed.
- The `tensorflow`, `fastai`, and `fastcore` versions are no longer included in the built-in headers.

### Keras 2 integration removed

The Keras 2 integration has been removed in v1.0. This includes the `KerasModelHubMixin` class and the `save_pretrained_keras`, `from_pretrained_keras`, and `push_to_hub_keras` functions.

Keras 3 is now tightly integrated with the Hub. You can use the [`ModelHubMixin`] to integrate your Keras 3 models with the Hub. Please refer to the [Integrate any ML framework with the Hub](./integrations.md) guide for more details.
