Object detection is a computer vision task which allows to detect instances of objects in different parts of a given image. Object detection models receive an image as an input and output the images including bounding boxes and labels on the detected objects.

## Use Cases
Object detection systems are used for crowd counting for social distancing, and face mask detection. It is also used in crowd surveillance to detect anomalities. 

## Inference
You can infer with object detection models through "object-detection" pipeline. Initialize the pipeline and give the path or http link to image when calling to infer the model. 

```python
model = pipeline("object-detection")

model("path_to_cat_image")

[{'label': 'blanket',
  'mask': mask_string,
  'score': 0.9171056747436523},
...]
```