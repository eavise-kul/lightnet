.. _accuracy:

Performance
===========
In this document we will compare the accuracy of networks running in this library and in darknet_.

ImageNet
--------

We went through the hassle of training on ImageNet from scratch, to be able to compare results with the darknet framework.
As you can see, lightnet achieves about the same results as darknet for this task. |br|
The lightnet trained weights of these networks can be downloaded via the following links:

- `darknet19 448x448 <https://mega.nz/#!fwUkRC5D!m-8cn5zfE_5-GxB_uzLNiWlfV72MMYN99WAso_dCQZo>`_
- `darknet53 448x448 <https://mega.nz/#!mlcm1ajJ!Ets7W_fNWP3_-zgfFoaxAm_fXonzbF8NrUKkQ7-uUSw>`_

.. image:: 04-imagenet.svg


.. _acc-voc:

Pascal VOC
----------

We compared the Yolo V2 network on the Pascal VOC imageset (train/test split). |br|
For more information on how to recreate these results, take a look at the :ref:`Pascal VOC <pascalvoc>` page. |br|

.. note::
   In order to generate these results, we mimicked the darknet training routine in Lightnet. |br|
   This means that we used the "traintest" split and not "trainvaltest",
   as explained in the Pascal VOC tutorial.

   We also ignore difficult annotations when computing mAP for these models.

========= ======= ===================== ===============================================
Framework mAP (%) Weight File           Note
========= ======= ===================== ===============================================
Darknet   76.2    `weights <dnw-voc_>`_ Darknet weights with Lightnet evaluation code
--------- ------- --------------------- -----------------------------------------------
Lightnet  75.6    `weights <lnw-voc_>`_ Transfer learned from Lightnet ImageNet weights
========= ======= ===================== ===============================================

.. _dnw-voc: https://pjreddie.com/media/files/yolov2-voc.weights
.. _lnw-voc: https://mega.nz/#!PoVCgCqQ!A0RGBpkLAOSXWkg-UZvCEayQSzllmdQlC7-H6uigyNE

.. figure:: 04-voc.svg
   :width: 100%
   :alt: PR-curve for each of the 20 classes in Pascal VOC

   PR curve for each of the 20 classes in Pascal VOC. Click on the image for a more detailed look.


.. _acc-coco:

MS COCO
-------
We compared YoloV2 and YoloV3 on the COCO dataset. |br|
Do note that we used our own code to evaluate the lightnet networks, and thus cannot guarantee that we have the same way of computing.
The darknet metrics come from the YoloV3 paper.

+--------------+---------+--------+------+------+-----------------------------+
|Model         |Framework|mAP_coco|mAP_50|mAP_75|Weight File                  |
+==============+=========+========+======+======+=============================+
|YoloV2 416x416|Darknet  |21.6%   |44.0% |19.2% |                             |
+              +---------+--------+------+------+-----------------------------+
|              |Lightnet |21.6%   |41.2% |21.0% |`weights <lnw-v2-coco_>`_    |
+--------------+---------+--------+------+------+-----------------------------+
|YoloV2 608x608|Darknet  |\-      |48.1% |\-    |`weights <dnw-v2-coco_>`_    |
+              +---------+--------+------+------+-----------------------------+
|              |Lightnet |24.5%   |46.9% |23.5% |`weights <lnw-v2-coco_>`_    |
+--------------+---------+--------+------+------+-----------------------------+
|YoloV3 608x608|Darknet  |33.0%   |57.9% |34.4% |`weights <dnw-v3-coco_>`_    |
+              +---------+--------+------+------+-----------------------------+
|              |Lightnet |32.7%   |56.4% |34.1% |`weights <lnw-v3-coco_>`_    |
+--------------+---------+--------+------+------+-----------------------------+

.. _lnw-v2-coco: https://mega.nz/#!ntVmhaxL!yVjLPPaphAY2sMnf8VbZqiLK_7RgSjmf9lSNlo7anbY
.. _lnw-v3-coco: https://mega.nz/#!r0dSgCTY!laOU6R8xinS9exhoOaUfg0hym3MmfShClm5ZRkb-jAM
.. _dnw-v2-coco: https://pjreddie.com/media/files/yolov2.weights
.. _dnw-v3-coco: https://pjreddie.com/media/files/yolov3.weights


.. include:: /links.rst
