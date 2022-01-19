## Use Cases

### Autonomous Driving

Segmentation models are used to identify road patterns such as lanes and obstacles for safer driving.

### Background Removal

Image Segmentation models are used in cameras to erase the background of certain objects and apply filters to them.

### Medical Imaging

Image Segmentation models are used to distinguish organs or tissues, improving medical imaging workflows. Models are used to segment dental instances, analyze X-Ray scans or even segment cells for pathological diagnosis. This [dataset](<[https://github.com/v7labs/covid-19-xray-dataset](https://github.com/v7labs/covid-19-xray-dataset)>) contains images of lungs of healthy patients and patients with COVID-19 segmented with masks. Another [segmentation dataset](<[https://ivdm3seg.weebly.com/data.html](https://ivdm3seg.weebly.com/data.html)>) contains segmented MRI data of the lower spine to analyze the effect of spaceflight simulation.

## Task Variants

### Semantic Segmentation

Semantic Segmentation is the task of segmenting parts of an image that belong to the same class. Semantic Segmentation models make predictions for each pixel and return the probabilities of the classes for each pixel. These models are evaluated on Mean Intersection Over Union (Mean IoU).

### Instance Segmentation

Instance Segmentation is the variant of Image Segmentation where every distinct object is segmented, instead of one segment per class.

### Panoptic Segmentation

Panoptic Segmentation is the Image Segmentation task that segments the image both by instance and by class, assigning each pixel a different instance of the class.

## Inference

You can infer with Image Segmentation models using the `image-segmentation` pipeline. You need to install [timm](https://github.com/rwightman/pytorch-image-models) first.

```python
!pip install timm
model = pipeline("image-segmentation")
model("cat.png")
#[{'label': 'cat',
#  'mask': mask_code,
#  'score': 0.999}
# ...]
```
