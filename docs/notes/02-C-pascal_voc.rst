Pascal VOC
===========
This page contains explanation of the scripts located in the `example folder`_ of the repository. |br|
These scripts were build to test the results of lightnet on the Pascal VOC dataset and compare them with darknet,
and as such are a nice example of some real code to learn from.
We perform the same training and testing as explained on the `darknet website`_.

.. rubric:: Get the data

We train YOLO on all of the VOC data from 2007 and 2012.
To get all the data, make a directory to store it all, and execute the following commands:

.. code:: bash

   mkdir data
   cd data
   wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
   wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
   wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
   tar xf VOCtrainval_11-May-2012.tar
   tar xf VOCtrainval_06-Nov-2007.tar
   tar xf VOCtest_06-Nov-2007.tar
   cd ..

There will now be a *VOCdevkit* folder with all the data.

.. rubric:: Generating labels

We need to have the right labels for training and testing the network. |br|
While brambox (and thus lightnet) can work with Pascal VOC annotations,
we still need to group the data in a training and testing set.
Because we are converting this anyway, we take the opportunity to convert the annotations to a pandas format,
which will be faster to parse whilst training/testing.
In the example code below, we save the data as HDF5 files, which requires the pytables_ package.
If you do not wish to install it, you can specify to save it as pandas pickle files instead *(-x .pkl)*,
but this means you will have to change to file names in the config files as well!

.. code:: bash

   # Check out ./bin/labels.py --help for an explanation of the arguments
   ./bin/labels.py -v -x .h5 data/

.. Note::
   There is no validation set.  
   We perform the same training cycle as darknet, and thus have no testing whilst training the network.
   This means there is no need for a separate testing and validation set,
   but also means we have no way to check how well the network performs whilst it is training.

.. rubric:: Get weights

For training, we use weights that are pretrained on ImageNet. |br|
See :ref:`accuracy` for more information on the difference between darknet and lightnet pretrained weights.

========= ===
Framework URL
========= ===
Darknet   https://pjreddie.com/media/files/darknet19_448.conv.23
--------- ---
Lightnet  https://mega.nz/#!ChsBkSQT!8Jpjzzi_tgPtd6gs079g4ea-XOUIr3LspOqAgk97hUA
========= ===

.. rubric:: Train model

Use the **train.py** script to train the model. You can use *train.py --help* for an explanation of the arguments and flags.

.. code:: bash

   # Adapt the model parameters inside of train.py to suite your needs
   ./bin/train.py -c -n cfg/yolo.py -vp <visdom port> <path/to/pretrained/weights>

.. rubric:: Test model

Use the **test.py** script to test the model. You can again use *test.py --help* for an explanation of the arguments and flags.

.. code:: bash

   # We use tqdm for a nice loading bar
   pip install tqdm 
   
   # Adapt the model parameters inside of test.py to suite your needs
   ./bin/test.py -c -n cfg/yolo.py backup/final.pt


.. include:: ../links.rst
.. _example folder: https://gitlab.com/EAVISE/lightnet/tree/master/example
.. _pytables: https://www.pytables.org/
.. _darknet website: https://pjreddie.com/darknet/yolov2/#train-voc
