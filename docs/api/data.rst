Data
====
.. automodule:: lightnet.data

Preprocessing
-------------
These classes perform data augmentation and conversion on your input.
They work just like the :mod:`torchvision transforms <pytorch:torchvision>`. |br|
First you create an object and then you call the object with the image or annotation object as parameter.
You can also call the ``apply()`` method on the classes to run the transformation once.

.. autoclass:: lightnet.data.transform.Letterbox
   :members: apply
.. autoclass:: lightnet.data.transform.RandomCrop
   :members: apply
.. autoclass:: lightnet.data.transform.RandomFlip
   :members: apply
.. autoclass:: lightnet.data.transform.HSVShift
   :members: apply
.. autoclass:: lightnet.data.transform.BramboxToTensor
   :members: apply

Postprocessing
--------------
These classes parse the output of your networks to understandable data structures.
They work just like the :mod:`torchvision transforms <pytorch:torchvision>`. |br|
First you create an object and then you call the object with the network output as parameter.
You can also call the ``apply()`` method on the classes to run the transformation once.

.. autoclass:: lightnet.data.transform.GetBoundingBoxes
   :members: apply
.. autoclass:: lightnet.data.transform.NonMaxSupression
   :members: apply
.. autoclass:: lightnet.data.transform.TensorToBrambox
   :members: apply
.. autoclass:: lightnet.data.transform.ReverseLetterbox
   :members: apply

Data loading
------------
.. autoclass:: lightnet.data.BramboxData
   :members:
.. autoclass:: lightnet.data.DataLoader
   :members:
.. autofunction:: lightnet.data.list_collate

Utilitary
---------
Some random classes and functions that are used in the data subpackage.

.. autoclass:: lightnet.data.transform.Compose
.. autoclass:: lightnet.data.transform.util.BaseTransform
   :members:
.. autoclass:: lightnet.data.transform.util.BaseMultiTransform
   :members:


.. include:: ../links.rst
