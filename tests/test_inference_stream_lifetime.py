import asyncio
import gc
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from huggingface_hub import AsyncInferenceClient, InferenceClient
from huggingface_hub.inference._common import RequestParameters


_BODY = b"data: one\n\ndata: two\n\n"


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(_BODY)))
        self.end_headers()
        self.wfile.write(_BODY)

    def log_message(self, *_args):
        return


def _server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread_started = threading.Thread(target=server.serve_forever, daemon=True)
    thread_started.start()
    return server, f"http://127.0.0.1:{port}/"


def _params(url: str) -> RequestParameters:
    return RequestParameters(url=url, task="conversational", model=None, json={"inputs": "hi"}, data=None, headers={})


def test_sync_stream_outlives_the_client_that_created_it():
    server, url = _server()
    try:

        def open_stream():
            client = InferenceClient(model=url, token="hf_test")
            return client._inner_post(_params(url), stream=True)

        lines = open_stream()
        gc.collect()
        assert list(lines) == ["data: one", "", "data: two", ""]
    finally:
        server.shutdown()


def test_async_stream_outlives_the_client_that_created_it():
    server, url = _server()

    async def open_stream():
        client = AsyncInferenceClient(model=url, token="hf_test")
        return await client._inner_post(_params(url), stream=True)

    async def run():
        try:
            lines = await open_stream()
            gc.collect()
            collected = [line async for line in lines]
            assert collected == ["data: one", "", "data: two", ""]
        finally:
            server.shutdown()

    asyncio.run(run())
