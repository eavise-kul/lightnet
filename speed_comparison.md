Comparing speeds
================
In this document we will compare the speed of various tasks, performed between networks of this library, of [pytorch-yolo2](https://github.com/marvis/pytorch-yolo2) and of [darknet](https://github.com/pjreddie/darknet).

## Single image test
This comparison was created by running a single image for inference through the network (Yolo-Voc with 1 class).  
To measurements were performed 25 times for each network.

| Measurement    | Statistic | Pytorch-Yolo2 | Lightnet             |
|:--------------:|:---------:|:-------------:|:--------------------:|
|                | Minimum   | 0.3427147865  | 0.3436274528         |
| Create network | Average   | 0.3885846233  | __0.3805842685__     |
|                | Maximum   | 0.4711854457  | 0.4875929355         |
|                |           |               |                      |
|                | Minimum   | 0.0798597335  | 0.0721385479         |
| Load weights   | Average   | 0.0838706016  | __0.0754387092__     |
|                | Maximum   | 0.1107821464  | 0.0901753902         |
|                |           |               |                      |
|                | Minimum   | 0.0034887790  | 0.0031592845         |
| Run network    | Average   | 0.0268937778  | __0.0261361885__     |
|                | Maximum   | 0.5856533050  | 0.5749838352         |
|                |           |               |                      |
|                | Minimum   | 0.0027985572  | 0.0016674995         |
| Region box     | Average   | 0.0046967124  | __0.0044290447__     |
|                | Maximum   | 0.0050539970  | 0.0049638748         |
|                |           |               |                      |
|                | Minimum   | 3.3855438232  | 3.3855438232e-05     |
| Perform nms    | Average   | 3.8003921508  | __3.7145614624e-05__ |
|                | Maximum   | 6.6518783569  | 7.2240829467e-05     |
|                |           |               |                      |
|                | Minimum   | 0.4309587478  | 0.4317719936         |
| TOTAL TIME     | Average   | 0.5040837192  | __0.4933306980__     |
|                | Maximum   | 1.1589803695  | 1.1574261188         |

## Yolo-VOC test
We compared running the Yolo-VOC network for inference against each other for 1000 images from the VOC2007 test data.
Inference was timed as run network + region box + non-maxima suppresion. Loading of the images, preprocessing and postprocessing were not included.  
__NOTE 1:__ The inference was tested one image at a time. This can be sped up quite a bit in lightnet by using a dataloader and setting the batch size higher.  
__NOTE 2:__ Darknet & Lightnet letterbox the image to the input dimension of the network (416,416), whilst Pytorch-Yolo2 resizes the image. This does not impact runtimes.  
[weight file](https://pjreddie.com/media/files/yolo-voc.weights) - [dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)

| Measurement    | Statistic | Pytorch-Yolo2 | Lightnet         | Darknet          | Lightnet CPU |
|:--------------:|:---------:|:-------------:|:----------------:|:----------------:|:------------:|
|                | Minimum   | 0.0082077980  | __0.0066645145__ | 0.0127110481     | 0.5437238216 |
| Inference time | Average   | 0.0093573399  | __0.0075396869__ | 0.0145603297     | 0.5992722511 |
|                | Maximum   | 0.8010835648  | 0.5802781582     | __0.0217719078__ | 0.7122807503 |
