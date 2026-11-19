# Migrating to huggingface_hub v2.0

`huggingface_hub` v2.0 replaces its HTTP dependency with [`httpx2`](https://httpx2.pydantic.dev/) and removes deprecated 1.x APIs. HTTP clients and exceptions now come from a separate package.

To support both `huggingface_hub` v1.x (starting with v1.30.0) and v2.x, import the HTTP module from `huggingface_hub.utils`:

```python
from huggingface_hub.utils import get_session, httpx

try:
    response = get_session().get("https://huggingface.co/api/models/gpt2")
    response.raise_for_status()
except httpx.HTTPError:
    ...
```

Use this same import when creating custom clients for [`set_client_factory`] or [`set_async_client_factory`]. Clients and exceptions from the old HTTP package are distinct from those in `httpx2` and cannot be used interchangeably.

`httpx2` uses the operating system's certificate trust store by default. Custom CA bundles configured with `SSL_CERT_FILE` or `SSL_CERT_DIR` are still supported. Logging configuration should target `httpx2` and `httpcore2` instead of the old logger names. See the [upstream migration guide](https://httpx2.pydantic.dev/migration/) for details.

The `oauth` extra now requires `authlib>=1.8.0` for `httpx2` support.

## Removed deprecated APIs

- Use `upload_folder` instead of `upload_large_folder` or `HfApi.upload_large_folder`.
- Use `duplicate_repo`, `set_space_volumes`, and `delete_space_volumes` instead of `duplicate_space`, `request_space_storage`, and `delete_space_storage`.
- Use `parse_hf_uri` instead of `repo_type_and_id_from_hf_id`, and `list_models(search=...)` instead of `model_name=...`.
- Use `space_volumes` instead of `space_storage` in `create_repo` and `duplicate_repo`.
- Use `InferenceEndpointType.AUTHENTICATED` or `type="authenticated"` instead of `PROTECTED` or `type="protected"`.
- Use `text_generation(stop=...)` instead of `stop_sequences=...`, and pass a token string or `None` to `InferenceClient` instead of a boolean.
- Use `hf` instead of `huggingface-cli`; `hf repo` and `hf repo-files delete` are replaced by `hf repos` and `hf repos delete-files`.
- Use `hf upload` instead of `hf upload-large-folder`; use `--volume` instead of `--storage` for repository creation and duplication, and `--status` or `--label` instead of `hf jobs ps --filter`.
- The deprecated `--claude` option on `hf skills add` and `hf skills update` is removed. Skills are installed for Claude Code automatically.

For more context, see the [`httpx2` project](https://github.com/pydantic/httpx2), the
[`httpx2` documentation](https://pydantic.dev/docs/httpx2/), the upstream
[transition guide from `httpx` to `httpx2`](https://httpx2.pydantic.dev/migration/), and the
[`huggingface_hub` transition plan](https://github.com/huggingface/huggingface_hub/issues/4802).
