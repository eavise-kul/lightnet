Comparing speeds
================
In this document we will compare the speed of various tasks, performed between networks of this library, of `pytorch-yolo2`_ and of `darknet`_.

Inference speed
---------------
We compared the Yolo-VOC network for inference in the different frameworks. |br|
Inference was timed as :math:`T_{forward\ pass\ network} + T_{get\ region\ box} + T_{non-maxima\ suppression}`.
Loading of the images, preprocessing and further postprocessing were not included. |br|
Before starting the measurements, an image was passed 500 times through the network as warm-up.
We then measured the inference time for 100 images and measured each image 50 times. |br|
The following table shows the results of the experiment on a Nvidia GTX 1080 Ti. |br|
`weight file`_ - `dataset`_

.. Note::
   The inference was tested one image at a time.
   This can be sped up in Lightnet by using a dataloader and setting the batch size higher.  

.. Note::
   Darknet & Lightnet letterbox the image to the input dimension of the network (416x416),
   whilst Pytorch-Yolo2 resizes the image. This should not impact runtimes significantly.

============  =======  =============  ==========  ============
Metric        Darknet  Pytorch-Yolo2  Lightnet    Lightnet-CPU
============  =======  =============  ==========  ============
Minimum (ms)  14.296   10.905         7.420       542.872
Average (ms)  13.735   8.748          7.182       502.948
Maximum (ms)  21.151   19.832         10.192      697.079
------------  -------  -------------  ----------  ------------
Average FPS   69.95    91.70          **134.77**    1.84   
============  =======  =============  ==========  ============

Training speed
--------------
.. Todo::
   Add speed of training lightnet on GTX 1080 Ti
   68463.08374524 sec -> 49300 batches (1.389 s/b)
   50288.76161551 sec -> 30700 batches (1.638 s/b)
   -----------------------------------------------
   1.484 s/b

We measured the time it took to train the Yolo-COV network on the Pascal VOC dataset. |br|
For this we measured total time of training and divided it by the number of batches (64 images) trained for.
You can find the method of measuring in the `training example script`_. |br|
The following table shows the results of the experiment on a Nvidia GTX 1080 Ti. |br|

.. Note::
   If anyone has measured the time needed to train on Pascal VOC in darknet,
   feel free to send me the results on gitlab. (Preferably on a Nvidia GTX 1080 Ti)

================  =======  ========
Metric            Darknet  Lightnet
================  =======  ========
Time/Batch (s/b)  *TBD*    1.484
================  =======  ========


.. include:: ../links.rst
.. _weight file: https://pjreddie.com/media/files/yolo-voc.weights
.. _dataset: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
.. _training example script: https://gitlab.com/EAVISE/lightnet/blob/master/examples/yolo-voc/train.py
