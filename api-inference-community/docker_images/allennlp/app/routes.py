import logging
import os
import time
from typing import Dict

from app.pipelines import Pipeline, PipelineException
from app.validation import ffmpeg_convert, normalize_payload
from pydantic import ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


HF_HEADER_COMPUTE_TIME = "x-compute-time"
HF_HEADER_COMPUTE_TYPE = "x-compute-type"
HF_HEADER_COMPUTE_CHARACTERS = "x-compute-characters"
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "cpu")

logger = logging.getLogger(__name__)


async def pipeline_route(request: Request) -> Response:
    start = time.time()
    payload = await request.body()
    task = os.environ["TASK"]
    try:
        inputs, params = normalize_payload(payload, task)
    except ValidationError as e:
        errors = []
        for error in e.errors():
            errors.append(f'{error["msg"]}: `{error["loc"][0]}` in `parameters`')
        return JSONResponse({"error": errors}, status_code=400)
    except (EnvironmentError, ValueError) as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    pipe = request.app.pipeline
    return call_pipe(pipe, inputs, params, start)


def call_pipe(pipe: Pipeline, inputs, params: Dict, start: float) -> Response:
    root_logger = logging.getLogger()
    warnings = set()

    class RequestsHandler(logging.Handler):
        def emit(self, record):
            """Send the log records (created by loggers) to
            the appropriate destination.
            """
            warnings.add(record.getMessage())

    handler = RequestsHandler()
    handler.setLevel(logging.WARNING)
    root_logger.addHandler(handler)
    for _logger in logging.root.manager.loggerDict.values():  # type: ignore
        try:
            _logger.addHandler(handler)
        except Exception:
            pass

    status_code = 200
    n_characters = 0
    try:
        outputs = pipe(inputs)
        n_characters = get_input_characters(inputs)
    except (PipelineException, AssertionError, ValueError) as e:
        outputs = {"error": str(e)}
        status_code = 400
    except Exception as e:
        outputs = {"error": "unknown error"}
        status_code = 500
        logger.error(f"There was an inference error: {e}")

    if warnings and isinstance(outputs, dict):
        outputs["warnings"] = list(sorted(warnings))

    compute_type = COMPUTE_TYPE
    headers = {
        HF_HEADER_COMPUTE_TIME: "{:.3f}".format(time.time() - start),
        HF_HEADER_COMPUTE_TYPE: compute_type,
        # https://stackoverflow.com/questions/43344819/reading-response-headers-with-fetch-api/44816592#44816592
        "access-control-expose-headers": f"{HF_HEADER_COMPUTE_TYPE}, {HF_HEADER_COMPUTE_TIME}",
    }
    if status_code == 200:
        headers[HF_HEADER_COMPUTE_CHARACTERS] = f"{n_characters}"

    if os.getenv("TASK") in {"text-to-speech", "audio-source-separation"}:
        # Special case, right now everything is flac audio we can output
        waveform, sampling_rate = outputs
        data = ffmpeg_convert(waveform, sampling_rate)
        headers["content-type"] = "audio/flac"
        return Response(data, headers=headers, status_code=status_code)
    else:
        return JSONResponse(
            outputs,
            headers=headers,
            status_code=status_code,
        )


def get_input_characters(inputs) -> int:
    if isinstance(inputs, str):
        return len(inputs)
    elif isinstance(inputs, (tuple, list)):
        return sum(get_input_characters(input_) for input_ in inputs)
    elif isinstance(inputs, dict):
        return sum(get_input_characters(input_) for input_ in inputs.values())
    return -1


async def status_ok(request):
    return JSONResponse({"ok": "ok"})
