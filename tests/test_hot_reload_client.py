import httpx2
import pytest

from huggingface_hub._hot_reload import client as client_module
from huggingface_hub._hot_reload.client import ReloadClient, multi_replica_reload_events


def _mock_replica(monkeypatch, status_code: int) -> list[str]:
    """Make every ReloadClient talk to a mock /get-reload endpoint answering `status_code`."""
    calls: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(request.url.path)
        return httpx2.Response(status_code)

    real_client = httpx2.Client

    def fake_client(**kwargs):
        return real_client(transport=httpx2.MockTransport(handler), **kwargs)

    monkeypatch.setattr(client_module.httpx2, "Client", fake_client)
    monkeypatch.setattr(client_module.time, "sleep", lambda _: None)
    return calls


@pytest.mark.parametrize("status_code", [204, 404])
def test_get_reload_returns_status_code_on_non_200(monkeypatch, status_code):
    _mock_replica(monkeypatch, status_code)
    client = ReloadClient(host="https://user-space.hf.space", subdomain="user-space", replica_hash="abc", token=None)

    result = client.get_reload("deadbeef")

    # Declared return type is `Iterator[...] | int`: a non-200 answer must come back as the int status code.
    assert result == status_code


def test_multi_replica_reload_events_raises_on_404(monkeypatch):
    _mock_replica(monkeypatch, 404)

    with pytest.raises(Exception, match="status_code=404"):
        list(
            multi_replica_reload_events(
                commit_sha="deadbeef",
                host="https://user-space.hf.space",
                subdomain="user-space",
                replica_hashes=["abc"],
                token=None,
            )
        )


def test_multi_replica_reload_events_retries_on_204(monkeypatch):
    calls = _mock_replica(monkeypatch, 204)

    events = []
    with pytest.raises(Exception, match="Too many retries reached"):
        for event in multi_replica_reload_events(
            commit_sha="deadbeef",
            host="https://user-space.hf.space",
            subdomain="user-space",
            replica_hashes=["abc"],
            token=None,
            max_retries=2,
        ):
            events.append(event)

    assert events[0]["kind"] == "warning"
    assert len(calls) == 3


def test_get_reload_streams_events_on_200(monkeypatch):
    calls = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(request.url.path)
        sse_body = b'event: message\ndata: {"status": "ok"}\n\n'
        return httpx2.Response(200, content=sse_body)

    real_client = httpx2.Client

    def fake_client(**kwargs):
        return real_client(transport=httpx2.MockTransport(handler), **kwargs)

    monkeypatch.setattr(client_module.httpx2, "Client", fake_client)

    client = ReloadClient(host="https://user-space.hf.space", subdomain="user-space", replica_hash="abc", token=None)
    events = client.get_reload("deadbeef")

    assert not isinstance(events, int)
    events_list = list(events)
    assert len(events_list) == 1
    assert events_list[0] == {"status": "ok"}
