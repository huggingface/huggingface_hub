# Migrating to huggingface_hub v1.0

The v1.0 release is a major milestone for the `huggingface_hub` library. It marks our commitment to API stability and the maturity of the library. We have made several improvements and breaking changes to make the library more robust and easier to use.

This guide is intended to help you migrate your existing code to the new version. If you have any questions or feedback, please let us know by [opening an issue on GitHub](https://github.com/huggingface/huggingface_hub/issues).

## Python 3.9+

`huggingface_hub` now requires Python 3.9 or higher. Python 3.8 is no longer supported.

## HTTPX migration

The `huggingface_hub` library now uses [`httpx`](https://www.python-httpx.org/) instead of `requests` for HTTP requests. This change was made to improve performance and to support both synchronous and asynchronous requests the same way. We therefore dropped both `requests` and `aiohttp` dependencies.

### Breaking changes

This is a major change that affects the entire library. While we have tried to make this change as transparent as possible, you may need to update your code in some cases. Here is a list of breaking changes introduced in the process:

- **Proxy configuration**: "per method" proxies are no longer supported. Proxies must be configured globally using the `HTTP_PROXY` and `HTTPS_PROXY` environment variables.
- **Custom HTTP backend**: The `configure_http_backend` function has been removed. You should now use [`set_client_factory`] and [`set_async_client_factory`] to configure the HTTP clients.
- **Error handling**: HTTP errors are not inherited from `requests.HTTPError` anymore, but from `httpx.HTTPError`. We recommend catching `huggingface_hub.HfHubHttpError` which is a subclass of `requests.HTTPError` in v0.x and of `httpx.HTTPError` in v1.x. Catching from the `huggingface_hub` error ensures your code is compatible with both the old and new versions of the library.
- **SSLError**: `httpx` does not have the concept of `SSLError`. It is now a generic `httpx.ConnectError`.
- **`LocalEntryNotFoundError`**: This error no longer inherits from `HTTPError`. We now define a `EntryNotFoundError` (new) that is inherited by both [`LocalEntryNotFoundError`] (if file not found in local cache) and [`RemoteEntryNotFoundError`] (if file not found in repo on the Hub). Only the remote error inherits from `HTTPError`.
- **`InferenceClient`**: The `InferenceClient` can now be used as a context manager. This is especially useful when streaming tokens from a language model to ensure that the connection is closed properly.
- **`AsyncInferenceClient`**: The `trust_env` parameter has been removed from the `AsyncInferenceClient`'s constructor. Environment variables are trusted by default by `httpx`. If you explicitly don't want to trust the environment, you must configure it with [`set_client_factory`].

For more details, you can check [PR #3328](https://github.com/huggingface/huggingface_hub/pull/3328) that introduced `httpx`.

### Why `httpx`?


The migration from `requests` to `httpx` brings several key improvements that enhance the library's performance, reliability, and maintainability:

**Thread Safety and Connection Reuse**: `httpx` is thread-safe by design, allowing us to safely reuse the same client across multiple threads. This connection reuse reduces the overhead of establishing new connections for each HTTP request, improving performance especially when making frequent requests to the Hub.

**HTTP/2 Support**: `httpx` provides native HTTP/2 support, which offers better efficiency when making multiple requests to the same server (exactly our use case). This translates to lower latency and reduced resource consumption compared to HTTP/1.1.

**Unified Sync/Async API**: Unlike our previous setup with separate `requests` (sync) and `aiohttp` (async) dependencies, `httpx` provides both synchronous and asynchronous clients with identical behavior. This ensures that `InferenceClient` and `AsyncInferenceClient` have consistent functionality and eliminates subtle behavioral differences that previously existed between the two implementations.

**Improved SSL Error Handling**: `httpx` handles SSL errors more gracefully, making debugging connection issues easier and more reliable.

**Future-Proof Architecture**: `httpx` is actively maintained and designed for modern Python applications. In contrast, `requests` is in maintenance mode and won't receive major updates like thread-safety improvements or HTTP/2 support.

**Better Environment Variable Handling**: `httpx` provides more consistent handling of environment variables across both sync and async contexts, eliminating previous inconsistencies where `requests` would read local environment variables by default while `aiohttp` would not.

The transition to `httpx` positions `huggingface_hub` with a modern, efficient, and maintainable HTTP backend. While most users should experience seamless operation, the underlying improvements provide better performance and reliability for all Hub interactions.

## `hf_transfer`

Now that all repositories on the Hub are Xet-enabled and that `hf_xet` is the default way to download/upload files, we've removed support for the `hf_transfer` optional package. The `HF_HUB_ENABLE_HF_TRANSFER` environment variable is therefore ignored. Use [`HF_XET_HIGH_PERFORMANCE`](../package_reference/environment_variables.md) instead.

## `Repository` class

The `Repository` class has been removed in v1.0. It was a thin wrapper around the `git` CLI for managing repositories. You can still use `git` directly in the terminal, but the recommended approach is to use the HTTP-based API in the `huggingface_hub` library for a smoother experience, especially when dealing with large files.

Here is a mapping from the legacy `Repository` class to the new `HfApi` one:

| `Repository` method                        | `HfApi` method                                        |
| ------------------------------------------ | ----------------------------------------------------- |
| `repo.clone_from`                          | `snapshot_download`                                   |
| `repo.git_add` + `git_commit` + `git_push` | [`upload_file`], [`upload_folder`], [`create_commit`] |
| `repo.git_tag`                             | `create_tag`                                          |
| `repo.git_branch`                          | `create_branch`                                       |

## `HfFolder` class

`HfFolder` was used to manage the user access token. Use [`login`] to save a new token, [`logout`] to delete it and [`whoami`] to check the user associated to the current token. Finally, use [`get_token`] to retrieve user's token in a script.


## `InferenceApi` class

`InferenceApi` was a class to interact with the Inference API. It is now recommended to use the [`InferenceClient`] class instead.

## Other deprecated features

Some methods and parameters have been removed in v1.0. The ones listed below have already been deprecated with a warning message in v0.x.

- `constants.hf_cache_home` has been removed. Please use `HF_HOME` instead.
- `use_auth_token` parameters have been removed from all methods. Please use `token` instead.
- `get_token_permission` method has been removed.
- `update_repo_visibility` method has been removed. Please use `update_repo_settings` instead.
- `is_write_action` parameter has been removed from `build_hf_headers` as well as `write_permission` from `login`. The concept of "write permission" has been removed and is no longer relevant now that fine-grained tokens are the recommended approach.
- `new_session` parameter in `login` has been renamed to `skip_if_logged_in` for better clarity.
- `resume_download`, `force_filename`, and `local_dir_use_symlinks` parameters have been removed from `hf_hub_download` and `snapshot_download`.
- `library`, `language`, `tags`, and `task` parameters have been removed from `list_models`.

## CLI cache commands

Cache management from the CLI has been redesigned to follow a Docker-inspired workflow. The legacy `hf cache scan` and `hf cache delete` commands are removed in v1.0 and are replaced with the new trio below:

- `hf cache ls` lists cache entries with concise table, JSON, or CSV output. Use `--revisions` to inspect individual revisions, add `--filter` expressions such as `size>1GB` or `accessed>30d`, and combine them with `--quiet` when you only need the identifiers.
- `hf cache rm` deletes selected cache entries. Pass one or more repo IDs (for example `model/bert-base-uncased`) or revision hashes, and optionally add `--dry-run` to preview or `--yes` to skip the confirmation prompt. This replaces both the interactive TUI and `--disable-tui` workflows from the previous command.
- `hf cache prune` performs the common cleanup task of deleting unreferenced revisions in one shot. Add `--dry-run` or `--yes` in the same way as with `hf cache rm`.


## TensorFlow and Keras 2.x support

All TensorFlow-related code and dependencies have been removed in v1.0. This includes the following breaking changes:

- `huggingface_hub[tensorflow]` is no longer a supported extra dependency
- The `split_tf_state_dict_into_shards` and `get_tf_storage_size` utility functions have been removed.
- The `tensorflow`, `fastai`, and `fastcore` versions are no longer included in the built-in headers.

The Keras 2.x integration has also been removed. This includes the `KerasModelHubMixin` class and the `save_pretrained_keras`, `from_pretrained_keras`, and `push_to_hub_keras` utilities. Keras 2.x is a legacy and unmaintained library. The recommended approach is to use Keras 3.x which is tightly integrated with the Hub (i.e. it contains built-in method to load/push to Hub). If you still want to work with Keras 2.x, you should downgrade `huggingface_hub` to v0.x version.

## `upload_file` and `upload_folder` return values

The [`upload_file`] and [`upload_folder`] functions now return the URL of the commit created on the Hub. Previously, they returned the URL of the file or folder. This is to align with the return value of [`create_commit`], [`delete_file`] and [`delete_folder`].
