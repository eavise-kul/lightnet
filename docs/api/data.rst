Data
====
.. automodule:: lightnet.data

Every data transformation operator in lightnet is one of two different types:

- :class:`~lightnet.data.transform.util.BaseTransform` : These classes only modify one type of data (usually image).
- :class:`~lightnet.data.transform.util.BaseMultiTransform` : These classes modify both the image and associated annotation data.

Check out the `tutorial <../notes/02-A-basics.html#Pre-processing-pipeline>`_ to learn how to use these operators.

----

Preprocessing
-------------
All pre-processing operators can work with PIL/Pillow images, OpenCV Numpy arrays or PyTorch Tensors (should be normalized float tensors between 0-1).
The pre-processing that works with annotations, expects brambox dataframes.

Fit
~~~
These transformation modify the image and annotation data to fit a certain input dimension and as such are all multi-transforms. |br|
You can pass the required dimensions directly to these classes, or you can pass a dataset object which will be used to get the dimensions from.
The latter only works with `Lightnet Datasets <#lightnet.data.Dataset>`_ and allows to change the required dimensions per batch.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: nomember-template.rst

   lightnet.data.transform.Crop
   lightnet.data.transform.Letterbox
   lightnet.data.transform.Pad

Augmentation
~~~~~~~~~~~~
These transformations allow you to augment your data, allowing you to train on more varied data without needing to fetch more.
Some of these transformations only modify the image data (usually color modifiers), others also need to modify the annotations and are thus multi-transforms.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: nomember-template.rst

   lightnet.data.transform.RandomFlip
   lightnet.data.transform.RandomHSV
   lightnet.data.transform.RandomJitter
   lightnet.data.transform.RandomRotate

Others
~~~~~~
Miscellaneous pre-processing operators that don’t fit in any other category.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: nomember-template.rst

   lightnet.data.transform.BramboxToTensor


----

Postprocessing
--------------

GetBoxes
~~~~~~~~
These operators allow you to convert various network output to a common bounding box tensor format:

.. math::
   Tensor_{<num\_boxes \, x \, 7>} = \begin{bmatrix}
      batch\_num, x_{tl}, y_{tl}, x_{br}, y_{br}, confidence, class\_id \\
      batch\_num, x_{tl}, y_{tl}, x_{br}, y_{br}, confidence, class\_id \\
      batch\_num, x_{tl}, y_{tl}, x_{br}, y_{br}, confidence, class\_id \\
      ...
   \end{bmatrix}

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: nomember-template.rst

   lightnet.data.transform.GetDarknetBoxes
   lightnet.data.transform.GetMultiScaleDarknetBoxes
   lightnet.data.transform.GetCornerBoxes

Filtering
~~~~~~~~~
The following classes allow you to filter output bounding boxes based on some criteria. |br|
They can work on the lightnet common bounding box `tensor format <#getboxes>`_ or on a brambox dataframe. 

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: nomember-template.rst

   lightnet.data.transform.NMS
   lightnet.data.transform.NMSFast
   lightnet.data.transform.NMSSoft
   lightnet.data.transform.NMSSoftFast

Reverse Fit
~~~~~~~~~~~
These operations cancel the `fit pre-processing <#fit>`_ operators and can only work on brambox dataframes.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: nomember-template.rst

   lightnet.data.transform.ReverseCrop
   lightnet.data.transform.ReverseLetterbox
   lightnet.data.transform.ReversePad

Others
~~~~~~
Miscellaneous post-processing operators that don’t fit in any other category.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: nomember-template.rst

   lightnet.data.transform.TensorToBrambox

----

Others
------
Some random classes and functions that are used in the data subpackage.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: member-template.rst

   lightnet.data.Dataset
   lightnet.data.DataLoader
   lightnet.data.brambox_collate
   lightnet.data.transform.Compose
   lightnet.data.transform.util.BaseTransform
   lightnet.data.transform.util.BaseMultiTransform


.. include:: /links.rst
