Comparing speeds
================
In this document we will compare the speed of various tasks, performed between networks of this library, of `pytorch-yolo2`_ and of `darknet`_.

Yolo-VOC test
-------------
We compared running the Yolo-VOC network for inference against each other for 1000 images from the VOC2007 test data.
Inference was timed as run network + region box + non-maxima suppresion. Loading of the images, preprocessing and postprocessing were not included. |br|
`weight file`_ - `dataset`_

**NOTE 1:** The inference was tested one image at a time. This can be sped up quite a bit in lightnet by using a dataloader and setting the batch size higher.  

**NOTE 2:** Darknet & Lightnet letterbox the image to the input dimension of the network (416,416), whilst Pytorch-Yolo2 resizes the image. This does not impact runtimes.  

===================  =========  =============  =========  ==========  ============
Measurement          Statistic  Pytorch-Yolo2  Lightnet   Darknet     Lightnet CPU
===================  =========  =============  =========  ==========  ============
Inference time (ms)  Minimum    8.207          **6.664**  12.711      543.723     
                                                                                  
                     Average    9.357          **7.539**  14.560      599.272     
                                                                                  
                     Maximum    801.083        580.278    **21.771**  712.280     
===================  =========  =============  =========  ==========  ============

Single image test
-----------------
This comparison was created by running a single image for inference through the network (Yolo-Voc with 1 class).  
To measurements were performed 25 times for each network.

.. warning::
    These measurements have been made on an old version of the package.

===================  =========  =============  =============
Measurement          Statistic  Pytorch-Yolo2  Lightnet     
===================  =========  =============  =============
Create network (ms)  Minimum    342.714        343.6274528  

                     Average    388.584        **380.584**  

                     Maximum    471.185        487.592      
-------------------  ---------  -------------  -------------
Load weights (ms)    Minimum    79.859         72.138       

                     Average    83.870         **75.438**   

                     Maximum    110.782        90.175       
-------------------  ---------  -------------  -------------
Run network (ms)     Minimum    3.488          3.159        

                     Average    26.893         **26.136**   

                     Maximum    585.653        574.983      
-------------------  ---------  -------------  -------------
Region box (ms)      Minimum    2.798          1.667        

                     Average    4.696          **4.429**    

                     Maximum    5.053          4.963        
-------------------  ---------  -------------  -------------
Perform nms (ms)     Minimum    3.385e-02      3.385e-02    

                     Average    3.800e-02      **3.714e-02**

                     Maximum    6.651e-02      7.224e-02    
-------------------  ---------  -------------  -------------
TOTAL TIME (ms)      Minimum    430.958        431.771      

                     Average    504.083        **493.330**  

                     Maximum    158.980        157.426      
===================  =========  =============  =============


.. include:: ../links.rst
.. _weight file: https://pjreddie.com/media/files/yolo-voc.weights
.. _dataset: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
