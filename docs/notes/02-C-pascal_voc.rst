.. _pascalvoc:

Pascal VOC
==========
This page contains explanation of the scripts located in the `example folder`_ of the repository. |br|
These scripts were build to test the results of lightnet on the Pascal VOC dataset and compare them with darknet,
and as such are a nice example of some real code to learn from.
We perform the same training and testing as explained on the `darknet website`_.

Reading scripts
---------------
This example code can be viewed as a final tutorial to learn how to use lightnet,
or even as a starting point to set up your own training routines.
While we will not go over each line in this example, we will give a quick overview of all the files and their use in the codebase.

The Pascal VOC training codebase consists of 2 folders, *cfg* and *bin*. |br|
We use the :class:`~lightnet.engine.HyperParameters` class to store all our hyperparameters in a separate file in the *cfg* folder.
This separation between code and configuration allows to completely modify how you train without having to delve into the code and allows to easily setup eg. grid search pipelines to find the optimal hyperparameters.

The *bin* folder contains the actual scripts that are used to train and evaluate our detection models:

labels.py
   This script parses the Pascal VOC annotations and splits them in train and test sets.
   This is a generic python script which has nothing to do with lightnet, but uses brambox_.

dataset.py
   This file contains a dataset object which loads a Pascal VOC image and it's annotations.

train.py
   This file creates and uses an :class:`~lightnet.engine.Engine` to train a model from the *cfg* directory on the *VOCDataset*.

test.py
   This file creates an object which is similar to the training engine to perform validation of a model.
   It loads a weight-file, runs the model through the testset and computes a PR and mAP metric.


.. Note::
   In this example, the dataset is defined in a separate file and kept the same for each training. |br|
   If you want to be able to also change the data (or pre-processing), you can define it in the HyperParameter config object as well:

   - Define a *params.dataloader* object
   - Define *params.anno*, *params.img_transform* and *params.anno_transform* and build a :class:`~lightnet.models.BramboxDataset` object in the training/testing scripts.
     
   This would make the scripts even more generalizable to any data format that is supported by brambox, but is left as an exercise to the reader.
   

Running scripts
---------------

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
Lightnet  https://mega.nz/#!Xxcx3ISS!okc1vufN288-G5JSAqweF3rPiUeoHi_gG35NzD8VUHA
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
