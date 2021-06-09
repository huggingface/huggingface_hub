import logging
import os
from typing import Dict, Type

from api_inference_community.routes import pipeline_route, status_ok
from app.pipelines import AutomaticSpeechRecognitionPipeline, Pipeline, AudioToAudioPipeline
from starlette.applications import Starlette
from starlette.routing import Route


TASK = os.getenv("TASK")
MODEL_ID = os.getenv("MODEL_ID")


logger = logging.getLogger(__name__)


# Add the allowed tasks
# Supported tasks are:
# - text-generation
# - text-classification
# - token-classification
# - translation
# - summarization
# - automatic-speech-recognition
# - ...
# For instance
# from app.pipelines import AutomaticSpeechRecognitionPipeline
# ALLOWED_TASKS = {"automatic-speech-recognition": AutomaticSpeechRecognitionPipeline}
# You can check the requirements and expectations of each pipelines in their respective
# directories. Implement directly within the directories.
ALLOWED_TASKS: Dict[str, Type[Pipeline]] = {
    "automatic-speech-recognition": AutomaticSpeechRecognitionPipeline,
    "audio-to-audio": AudioToAudioPipeline
}


def get_pipeline(task: str, model_id: str) -> Pipeline:
    if task not in ALLOWED_TASKS:
        raise EnvironmentError(f"{task} is not a valid pipeline for model : {model_id}")
    return ALLOWED_TASKS[task](model_id)


routes = [
    Route("/{whatever:path}", status_ok),
    Route("/{whatever:path}", pipeline_route, methods=["POST"]),
]

app = Starlette(routes=routes)
if os.environ.get("DEBUG", "") == "1":
    from starlette.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_headers=["*"], allow_methods=["*"]
    )


@app.on_event("startup")
async def startup_event():
    logger = logging.getLogger("uvicorn.access")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.handlers = [handler]

    task = os.environ["TASK"]
    model_id = os.environ["MODEL_ID"]
    app.pipeline = get_pipeline(task, model_id)


if __name__ == "__main__":
    task = os.environ["TASK"]
    model_id = os.environ["MODEL_ID"]

    get_pipeline(task, model_id)
