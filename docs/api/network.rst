Network
=======
.. automodule:: lightnet.network

Layer
------
.. automodule:: lightnet.network.layer

Containers
~~~~~~~~~~
.. autoclass:: lightnet.network.layer.Fusion
.. autoclass:: lightnet.network.layer.HourGlass
.. autoclass:: lightnet.network.layer.Parallel
.. autoclass:: lightnet.network.layer.ParallelCat
.. autoclass:: lightnet.network.layer.ParallelSum
.. autoclass:: lightnet.network.layer.Residual
.. autoclass:: lightnet.network.layer.SequentialSelect

Convolution
~~~~~~~~~~~~~
.. autoclass:: lightnet.network.layer.Conv2dBatchReLU
.. autoclass:: lightnet.network.layer.Conv2dDepthWise
.. autoclass:: lightnet.network.layer.CornerPool
.. autoclass:: lightnet.network.layer.InvertedBottleneck

Pooling
~~~~~~~
.. autoclass:: lightnet.network.layer.BottomPool
.. autoclass:: lightnet.network.layer.GlobalAvgPool2d
.. autoclass:: lightnet.network.layer.LeftPool
.. autoclass:: lightnet.network.layer.PaddedMaxPool2d
.. autoclass:: lightnet.network.layer.RightPool
.. autoclass:: lightnet.network.layer.TopPool

Others
~~~~~~
.. autoclass:: lightnet.network.layer.Flatten
.. autoclass:: lightnet.network.layer.Reorg


Loss
----
.. automodule:: lightnet.network.loss
.. autoclass:: lightnet.network.loss.RegionLoss
   :members: forward
.. autoclass:: lightnet.network.loss.MultiScaleRegionLoss

Module
------
.. automodule:: lightnet.network.module
.. autoclass:: lightnet.network.module.Lightnet
   :members:
.. autoclass:: lightnet.network.module.Darknet
   :members:


.. include:: ../links.rst
