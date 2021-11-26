import base64
import functools
import io
import logging
import os
from typing import Dict, Type

import numpy as np
import tensorflow as tf
from api_inference_community.routes import pipeline_route, status_ok
from app.pipelines import ImageClassificationPipeline, Pipeline
from huggingface_hub import cached_download, hf_hub_url
from PIL import Image
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.routing import Route
from tensorflow import keras


TASK = os.getenv("TASK")
MODEL_ID = os.getenv("MODEL_ID")


class MyPipeline(Pipeline):
    def __init__(self, model_id):
        filename = cached_download(hf_hub_url(model_id, "tf_model.h5"))
        self.model = keras.models.load_model(filename)

    def __call__(self, inputs):
        img = np.array(inputs)
        im = tf.image.resize(img, (128, 128))
        im = tf.cast(im, tf.float32) / 255.0

        import datetime

        start = datetime.datetime.now()

        pred_mask = self.model.predict(im[tf.newaxis, ...])

        pred_mask_arg = tf.argmax(pred_mask, axis=-1)
        print("Inference", datetime.datetime.now() - start)

        labels = []

        binary_masks = {}

        mask_codes = {}

        for cls in range(pred_mask.shape[-1]):

            binary_masks[f"mask_{cls}"] = np.zeros(
                shape=(pred_mask.shape[1], pred_mask.shape[2])
            )

            for row in range(pred_mask_arg[0][1].get_shape().as_list()[0]):

                for col in range(pred_mask_arg[0][2].get_shape().as_list()[0]):

                    if pred_mask_arg[0][row][col] == cls:

                        binary_masks[f"mask_{cls}"][row][col] = 1

                    else:

                        binary_masks[f"mask_{cls}"][row][col] = 0

            mask = binary_masks[f"mask_{cls}"]

            mask *= 255

            img = Image.fromarray(mask.astype(np.int8), mode="L")

            with io.BytesIO() as out:

                img.save(out, format="PNG")

                png_string = out.getvalue()

                mask = base64.b64encode(png_string).decode("utf-8")

            mask_codes[f"mask_{cls}"] = mask

            labels.append(
                {
                    "label": f"LABEL_{cls}",
                    "mask": mask_codes[f"mask_{cls}"],
                    "score": 1.0,
                }
            )
        print("Postprocess", datetime.datetime.now() - start)

        return labels
        import ipdb

        ipdb.set_trace()


logger = logging.getLogger(__name__)

ALLOWED_TASKS: Dict[str, Type[Pipeline]] = {
    "image-classification": ImageClassificationPipeline,
    "image-segmentation": MyPipeline,
}


@functools.lru_cache()
def get_pipeline() -> Pipeline:
    task = os.environ["TASK"]
    model_id = os.environ["MODEL_ID"]
    if task not in ALLOWED_TASKS:
        raise EnvironmentError(f"{task} is not a valid pipeline for model : {model_id}")
    return ALLOWED_TASKS[task](model_id)


routes = [
    Route("/{whatever:path}", status_ok),
    Route("/{whatever:path}", pipeline_route, methods=["POST"]),
]

middleware = [Middleware(GZipMiddleware, minimum_size=1000)]
if os.environ.get("DEBUG", "") == "1":
    from starlette.middleware.cors import CORSMiddleware

    middleware.append(
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_headers=["*"],
            allow_methods=["*"],
        )
    )

app = Starlette(routes=routes, middleware=middleware)


@app.on_event("startup")
async def startup_event():
    logger = logging.getLogger("uvicorn.access")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.handlers = [handler]

    # Link between `api-inference-community` and framework code.
    app.get_pipeline = get_pipeline
    try:
        get_pipeline()
    except Exception:
        # We can fail so we can show exception later.
        pass


if __name__ == "__main__":
    try:
        get_pipeline()
    except Exception:
        # We can fail so we can show exception later.
        pass
