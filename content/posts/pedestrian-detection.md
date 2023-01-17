+++

author = "pme0"
title = "Pedestrian Detection"
date = "2022-01-10"
description = ""
tags = [
    "Image & Video",
    "Object Detection",
    "YOLO", 
]
math = true

+++



{{< figure src="/images/pedestrian-detection/thumbnail_pedestrians.png" width="25%" >}}


## Introduction



## Model


## Inference

The most important inference hyperparameters are the *confidence threshold* determining the minimum confidence for a sucessful detection, and the *IoU threshold* determining the intersection of union for considering proximal bounding boxes as different objects. Additionally, a *class filter* can be used to exclude unwanted object classes from the detections. The standard *non max suppression* is applied in order to clean-up the raw detections produces by the model:
```python
conf_thres = 0.20
iou_thres=0.45
classes = [0]  # class '0' corresponds to 'person'

predictions = model(images)
predictions = non_max_suppression(predictions, conf_thres, iou_thres, classes)
```
The model returns a set predictions, each representing a detection where the first four elements are the bounding box coordinates and the last two are the confidence score and the class index, respectively:
```python
tensor([[2.42451e+02, 2.63408e+02, 3.22044e+02, 4.49350e+02, 9.22621e-01, 0.00000e+00],
        [9.62039e+01, 2.63770e+02, 1.76632e+02, 4.87993e+02, 9.18263e-01, 0.00000e+00],
        [6.66377e+02, 2.69026e+02, 7.47873e+02, 4.56455e+02, 8.92933e-01, 0.00000e+00],
        ...
    ])
```



## Results

In the following **single object detection** example we detect pedestrians:

{{< figure src="/images/pedestrian-detection/pexels-kate-trifo-4019405_bboxes.png" width="60%" >}}


The detection framework can be extended to detect more than one class. In the following **multiple object detection** example we detect pedestrians as well as bycicles and cars:

{{< figure src="/images/pedestrian-detection/pexels-aleks-magnusson-2962589_bboxes.png" width="60%" >}}



The model loses accuracy when presented with very distant object, as depicted in the next example:

{{< figure src="/images/pedestrian-detection/pexels-luis-dalvan-1770808_bboxes.png" width="60%" >}}
 
The reason is two-fold: firstly, the training data contains larger instances of pedestrians, and the multi-scale detector is only able to resolve the features up to a point;
and secondly, the rescaling of the image used for inference (1000x660) to fit the expected input size of the model (640x640) causes a distortion in the case of non-square images.
A crop of a square region previously containing undetected instances shows that the model performs much better in this case. 

{{< figure src="/images/pedestrian-detection/pexels-luis-dalvan-1770808-detail.jpg" width="60%" >}}
 


