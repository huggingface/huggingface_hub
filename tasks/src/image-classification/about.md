## Use Cases
Image classification is used for wide range of problems, from medical imaging to remote sensing. An example application is determining whether a Computed Tomography scan contains cancerous tissue or not.

## Inference
With the `transformers` library, you can use the `image-classification` pipeline to infer with image classification models. You can initialize the pipeline with a model id from the Hub. If you don't provide a model id, it will initialize with [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) by default. When calling the pipeline, you just need to specify a path, http link or an image loaded in PIL to this pipeline. You can also provide a `top_k` parameter which determines how many results it should return.

```python
!pip install transformers

from transformers import pipeline
clf = pipeline("image-classification")
clf("path_to_a_cat_image")

[{'label': 'tabby, tabby cat', 'score': 0.7319658398628235},
{'label': 'Egyptian cat', 'score': 0.14533261954784393},
{'label': 'tiger cat', 'score': 0.11755114793777466},
{'label': 'lynx, catamount', 'score': 0.002333116251975298},
{'label': 'quilt, comforter, comfort, puff', 'score': 0.000367624219506979}]
## Useful Resources
```

## Useful Resources
#### Creating your own image classifier in just a few minutes
With [HuggingPics](https://github.com/nateraw/huggingpics), you can fine-tune Vision Transformers for anything using images found on the web. This project downloads images of classes defined by you, trains a model, and pushes it to the Hub. You even get to try out the model directly with a working widget in the browser, ready to be shared with all your friends!
