import json
from typing import Iterator, Optional

import httpx

from ..utils._headers import build_hf_headers
from ..utils._sse_client import SSEClient
from .types import ApiGetReloadEventSourceData, ApiGetReloadRequest


HOT_RELOADING_PORT = 7887


class ReloadClient:
    def __init__(
        self,
        *,
        host: str,
        subdomain: str,
        replica_hash: str,
        token: Optional[str],
    ):
        base_host = host.replace(subdomain, f"{subdomain}--{HOT_RELOADING_PORT}")
        self.replica_hash = replica_hash
        self.client = httpx.Client(
            base_url=f"{base_host}/--replicas/+{replica_hash}",
            headers=build_hf_headers(token=token),
        )

    def get_reload(self, reload_id: str) -> Iterator[ApiGetReloadEventSourceData]:
        req = ApiGetReloadRequest(reloadId=reload_id)
        with self.client.stream("POST", "/get-reload", json=req) as res:
            assert res.status_code == 200, res.status_code  # TODO: Raise specific error ? Return ?
            for event in SSEClient(res.iter_bytes()).events():
                if event.event == "message":
                    yield json.loads(event.data)
