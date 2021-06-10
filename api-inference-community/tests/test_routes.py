import os
from unittest import TestCase
import logging
from api_inference_community.routes import status_ok, pipeline_route
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient


class ValidationTestCase(TestCase):
    def test_invalid_pipeline(self):
        os.environ["TASK"] = "invalid"

        def get_pipeline():
            raise Exception("We cannot load the pipeline")

        routes = [
            Route("/{whatever:path}", status_ok),
            Route("/{whatever:path}", pipeline_route, methods=["POST"]),
        ]

        app = Starlette(routes=routes)

        @app.on_event("startup")
        async def startup_event():
            logger = logging.getLogger("uvicorn.access")
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            logger.handlers = [handler]

            # Link between `api-inference-community` and framework code.
            app.get_pipeline = get_pipeline

        with TestClient(app) as client:
            response = client.post("/", data=b"")
        self.assertEqual(
            response.status_code,
            500,
        )
        self.assertEqual(response.content, b'{"error":"We cannot load the pipeline"}')
