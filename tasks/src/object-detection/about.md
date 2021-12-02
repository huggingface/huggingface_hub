Object detection is a computer vision task which allows to detect instances of objects in different parts of a given image. Object detection models receive an image as an input and output the images including bounding boxes and labels on the detected objects.

## Use Cases

### Autonomous Driving
Perhaps one of the most significant use cases of object detection is autonomous driving. Self-driving cars use object detection to detect pedestrians, bicycles traffic lights and road signs to decide on which step it should take. 

### Object Tracking in Matches
Object detection systems are widely used in sports, where the ball, or a player is tracked for monitoring and refereeing during matches. 

### Image Search
Object detection is used for image search. Smart phones use object detection models to detect entities (such as places, or objects) and let the user search for the entity on the internet.

### Object Counting 
Object detection models are used to count instances of objects in a given picture, this can include counting the objects in warehouses or stores, or counting the number of visitors in a store. This is also used to manage crowds in events to prevent disasters.

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