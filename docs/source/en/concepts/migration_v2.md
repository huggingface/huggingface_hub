# Migrating to huggingface_hub v2.0

`huggingface_hub` v2.0 replaces its HTTP dependency with [`httpx2`](https://httpx2.pydantic.dev/). The Python API stays the same, but HTTP clients and exceptions now come from a separate package.

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

For more context, see the [`httpx2` project](https://github.com/pydantic/httpx2), the
[`httpx2` documentation](https://pydantic.dev/docs/httpx2/), the upstream
[transition guide from `httpx` to `httpx2`](https://httpx2.pydantic.dev/migration/), and the
[`huggingface_hub` transition plan](https://github.com/huggingface/huggingface_hub/issues/4802).
