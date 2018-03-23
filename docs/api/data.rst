Data
====
.. automodule:: lightnet.data

Preprocessing
-------------
These classes perform data augmentation and conversion on your input.
They work just like the :mod:`torchvision transforms <pytorch:torchvision>`. |br|
First you create an object and then you call the object with the image or annotation object as parameter.
You can also call the ``apply()`` method on the classes to run the transformation once.

.. autoclass:: lightnet.data.Letterbox
   :members: apply
.. autoclass:: lightnet.data.RandomCrop
   :members: apply
.. autoclass:: lightnet.data.RandomFlip
   :members: apply
.. autoclass:: lightnet.data.HSVShift
   :members: apply
.. autoclass:: lightnet.data.BramboxToTensor
   :members: apply

Postprocessing
--------------
These classes parse the output of your networks to understandable data structures.
They work just like the :mod:`torchvision transforms <pytorch:torchvision>`. |br|
First you create an object and then you call the object with the image object as parameter.
You can also call the ``apply()`` method on the classes to run the transformation once.

.. autoclass:: lightnet.data.GetBoundingBoxes
   :members: apply
.. autoclass:: lightnet.data.TensorToBrambox
   :members: apply
.. autoclass:: lightnet.data.ReverseLetterbox
   :members: apply

Data loading
------------
.. autoclass:: lightnet.data.BramboxData
   :members:
.. autoclass:: lightnet.data.DataLoader
   :members:
.. autofunction:: lightnet.data.list_collate


.. include:: ../links.rst
