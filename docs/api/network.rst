Network
=======
.. automodule:: lightnet.network

Layer
------

Containers
~~~~~~~~~~
Container modules are used to structure the networks more easily and dont necessarily contain actual computation layers themselves.
You can think of them as more advanced :class:`torch.nn.Sequential` classes.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: nomember-template.rst

   lightnet.network.layer.Fusion
   lightnet.network.layer.HourGlass
   lightnet.network.layer.Parallel
   lightnet.network.layer.ParallelCat
   lightnet.network.layer.ParallelSum
   lightnet.network.layer.Residual
   lightnet.network.layer.SequentialSelect

Convolution
~~~~~~~~~~~~~
These layers are convenience layers that group together one or more convolutions with some other layer (eg. batchnorm, dropout, relu, etc.).

.. Note::
   Most of the convolution layers have a `relu` class argument. |br|
   If you require the `relu` class to get extra parameters, you can use a `lambda` or `functools.partial`:

   >>> conv = ln.network.layer.Conv2dBatchReLU(
   ...     3, 32, 3, 1, 1,
   ...     relu=lambda: torch.nn.LeakyReLU(0.1, inplace=True)
   ... )
   >>> print(conv)
   Conv2dBatchReLU(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), LeakyReLU(negative_slope=0.1, inplace=True))

   >>> import functools
   >>> conv = ln.network.layer.Conv2dDepthWise(
   ...     32, 64, 1, 1, 0,
   ...     relu=functools.partial(torch.nn.ELU, 0.5, inplace=True)
   ... )
   >>> print(conv)
   Conv2dDepthWise(32, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), ELU(alpha=0.5, inplace=True))

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: nomember-template.rst

   lightnet.network.layer.Conv2dBatchReLU
   lightnet.network.layer.Conv2dDepthWise
   lightnet.network.layer.CornerPool
   lightnet.network.layer.InvertedBottleneck

Pooling
~~~~~~~
These layers perform some kind of specialised pooling operation.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: nomember-template.rst

   lightnet.network.layer.BottomPool
   lightnet.network.layer.LeftPool
   lightnet.network.layer.PaddedMaxPool2d
   lightnet.network.layer.RightPool
   lightnet.network.layer.TopPool

Others
~~~~~~
Miscellaneous layers that don't fit in any other category.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: nomember-template.rst

   lightnet.network.layer.Flatten
   lightnet.network.layer.Reorg

----

Loss
----
.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: forwardmember-template.rst

   lightnet.network.loss.RegionLoss
   lightnet.network.loss.MultiScaleRegionLoss
   lightnet.network.loss.CornerLoss

----

Module
------
.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: member-template.rst

   lightnet.network.module.Lightnet
   lightnet.network.module.Darknet


.. include:: /links.rst
