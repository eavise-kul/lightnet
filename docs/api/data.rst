Data
====
.. automodule:: lightnet.data

Preprocessing
-------------
These classes work just like :mod:`torchvision transforms <pytorch:torchvision>`. First, you create an object, and the you call the object with the image or annotation object as parameter.

.. autoclass:: lightnet.data.Letterbox
.. autoclass:: lightnet.data.RandomCrop
.. autoclass:: lightnet.data.RandomFlip
.. autoclass:: lightnet.data.HSVShift
.. autoclass:: lightnet.data.AnnoToTensor

Postprocessing
--------------
These classes and functions help to parse the output of a network to understandable data structures.

.. autoclass:: lightnet.data.BBoxConverter
.. autofunction:: lightnet.data.bbox_to_brambox

Dataset
-------
.. autoclass:: lightnet.data.BramboxData
.. autofunction:: lightnet.data.bbb_collate


.. include:: ../links.rst
