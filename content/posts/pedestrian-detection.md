+++

author = "pme0"
title = "Object Detection"
date = "2022-01-10"
description = ""
tags = [
    "Image & Video",
    "Object Detection",
    "YOLO", 
]
math = true

+++




<iframe src="/media/out.mp4"
        scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"></iframe>

{{< video src="/media/pedestrian-detection/pexels-people-walking-2670-cut_bboxes.mp4" type="video/mp4" preload="auto" >}}

## Object Detection


## Pedestrian Detection


{{< figure src="/media/pedestrian-detection/thumbnail_pedestrians.png" width="35%" >}}



Detecting pedestrians is an important area of application for Artificial Intelligence. Some examples include:
- **public safety:** monitoring overcrowding at public events or in closed public spaces---such as underground stations and tight pedestrian areas---where emergency exits are limited and/or overcrowding can lead to panic, stampedes and crushes (e.g. Seoul Halloween crowd crush in 2022);
- **road safety:** ensuring autonomous vehicles are aware of pedestrians---such as people crossing the road or exiting parked cars---and are able to take emergency action if a dangerous situation arises;
- **human-robit interaction:** As AI and Robots become more prevalent, so will interactions between humans and machines, and it is therefore important for machines to be aware of sorrounding humans, for example when human and robots share working spaces or even work together in certain tasks;

In the following we analyze some examples of pedestrian detection using a detector and a visualizer. The code can be found in 



[YOLOv5](https://github.com/ultralytics/yolov5)

YOLOv5 has three main components:
1. **Backbone** using a combination of cross stage partial network (CSPNet) and Darknet which is termed CSPDarknet and which works as a *feature extraction* network;
2. **Neck** using a path aggregation network (PANet) which works as a *feature fusion* network;
3. **Head** using a convolutional network which works as a *multi-scale detector* network by outputting feature maps of different sizes;

Some diagrams illustrating the overall network architecture can be found [here](https://github.com/ultralytics/yolov5/issues/280)


## Inference

The most important inference hyperparameters are the *confidence threshold* determining the minimum confidence for a sucessful detection, and the *IoU threshold* determining the intersection of union for considering proximal bounding boxes as different objects. Additionally, a *class filter* can be used to exclude unwanted object classes from the detections. The standard *non max suppression* is applied in order to clean-up the raw detections produces by the model:
```python
# detection parameters
conf_thres = 0.25
iou_thres=0.45
classes = [0]

# detection outputs
predictions = model(images)
predictions = non_max_suppression(predictions, conf_thres, iou_thres, classes)
```
The list `class` defines the set of objects to be detected, with '0' in this example corresponding to 'person'. This must be one of the classes of objects that the model has been trained on.

The model returns a set predictions, each representing a detection where the first four elements are the bounding box coordinates and the last two are the confidence score and the class index, respectively:
```python
tensor([[2.42451e+02, 2.63408e+02, 3.22044e+02, 4.49350e+02, 9.22621e-01, 0.00000e+00],
        [9.62039e+01, 2.63770e+02, 1.76632e+02, 4.87993e+02, 9.18263e-01, 0.00000e+00],
        [6.66377e+02, 2.69026e+02, 7.47873e+02, 4.56455e+02, 8.92933e-01, 0.00000e+00],
        ...
    ])
```



## Visualization

In the following **single object detection** example we detect pedestrians:

{{< figure src="/media/pedestrian-detection/pexels-kate-trifo-4019405_bboxes_0.png" width="60%" >}}

In some situation we may wish to safeguard privacy while still being able to monitor pedestrians, in which case a bluring filter can be applied:

{{< figure src="/media/pedestrian-detection/pexels-kate-trifo-4019405_bboxes_0_blur.png" width="60%" >}}


The model performs when large objects are present but loses accuracy when presented with distant object, as depicted in the next example:

{{< figure src="/media/pedestrian-detection/pexels-luis-dalvan-1770808_bboxes.png" width="80%" >}}
 
The reason is two-fold: 
Firstly, the training data contains larger instances of pedestrians, and the multi-scale detector is only able to resolve the features up to a point. This is a common problem in small-object detection tasks.
Secondly, the rescaling of the image to fit the expected input size of the model (from 1000x660 to 640x640) causes a distortion in the case of non-square input images.
A crop of a square region previously containing undetected instances shows that the model performs much better in this case. 

{{< figure src="/media/pedestrian-detection/pexels-luis-dalvan-1770808-detail.jpg" width="80%" >}}
 
Alternatively, the confidence threshold can be decreasing so that less confident prediction are not excluded, which typically happens to very small objects.
We can see an example of this approach in the following image. With a high threshold only the instances for which the model is very certain are picked up

{{< figure src="/media/pedestrian-detection/pedestrians_abrd_conf90.png" width="60%" >}}

whereas with a low threshold the model is able to pick up even the three very small and well camouflaged observers in the background on the left side of the road---impressive!
{{< figure src="/media/pedestrian-detection/pedestrians_abrd_conf3.png" width="60%" >}}

{{< figure src="/media/pedestrian-detection/pedestrians_abrd-crop.png" width="80%" >}}

The optimal value for the Confidence and IoU thresholds will depend on the primary objective of the application. For example, in applications where detecting the number of pedestrians is critical for safety reasons and crowds may form---thus causing partial oclusions---we want to lower both Confidence and IoU thresholds; the cost of overestimation is negligible in this scenario, compared to the risks of overcrowing. In contrast, if the application demands focus on the foreground, we want higher Confidence threshold in order to abstract from the background 



## Conclusions

......

The detection framework can be extended to detect more than one class. In the following **multiple object detection** example we detect pedestrians as well as bicycles and cars:

{{< figure src="/media/pedestrian-detection/pexels-aleks-magnusson-2962589_bboxes.png" width="70%" >}}
