## Use Cases
The most important use of image segmentation is in computer vision for robotics, such as autonomous driving. Segmentation models are used to identify pedestrians, lanes and other necessary information. Image segmentation models are also used in cameras to erase background of portraits in images.

## Task Variants
### Semantic Segmentation 
Semantic segmentation is the task of segmenting parts of an image together which belong to the same class. Semantic segmentation models make predictions for each pixel and return the probabilities of classes for each pixel. These models are evaluated on Mean Intersection Over Union (Mean IoU).

### Instance Segmentation
Instance segmentation is the variant of image segmentation where every distinct object is segmented, instead of one segment per class. 

### Panoptic Segmentation
Panoptic Segmentation is the image segmentation task that segments the image both instance-wise and class-wise, it has assigns every pixel a distinct instance of the class.

## Inference
You can infer with the image segmentation models using The Inference API.  

```python
import json
import requests

headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = f"https://api-inference.huggingface.co/models/{model_id}"

def query(*filename*):
    with open(filename, "rb") as f:
        data = f.read()

    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))

data = query(path_to_image)

## [{'label': 'Cat',
## 'mask': mask_code,
## 'score': 1.0},
## ...]
```

