## Use Cases
Image classification models can be used when we are not interested in specific instances of objects with location information or their shape.

### Keyword Classification
Image classification models are used widely in stock photography to assign each image a keyword.

### Image Search

Models trained in image classification can improve user experience by organizing and categorizing photo galleries on the phone or in the cloud, on multiple keywords or tags.

## Inference

With the `transformers` library, you can use the `image-classification` pipeline to infer with image classification models. You can initialize the pipeline with a model id from the Hub. If you do not provide a model id it will initialize with [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) by default. When calling the pipeline you just need to specify a path, http link or an image loaded in PIL. You can also provide a `top_k` parameter which determines how many results it should return.

```python
from transformers import pipeline
clf = pipeline("image-classification")
clf("path_to_a_cat_image")

[{'label': 'tabby cat', 'score': 0.731},
...
]
```

## Useful Resources

### Creating your own image classifier in just a few minutes

With [HuggingPics](https://github.com/nateraw/huggingpics), you can fine-tune Vision Transformers for anything using images found on the web. This project downloads images of classes defined by you, trains a model, and pushes it to the Hub. You even get to try out the model directly with a working widget in the browser, ready to be shared with all your friends!
