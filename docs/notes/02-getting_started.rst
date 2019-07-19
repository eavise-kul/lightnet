Getting started
===============
Lightnet was originally built to recreate darknet_ type networks in pytorch_.
It has however evolved towards a library to aid **0phoff** in his PhD research, and as such contains many more building blocks than only the ones needed for darknet.

The library is build in a hierarchical way.
You can choose to only use a small subset of the building blocks from a few subpackages and build up your own network architectures or use the entire library and just use the network models that are provided. |br|
The different subpackages of lightnet are:

lightnet.network
   This submodule contains everything related to building networks.
   This means layers, loss functions, weight-loading, etc.

ligthnet.data
   In here you will find everything related to data-processing.
   This includes data augmentation, post-processing of network output, etc.

lightnet.engine
   This submodule contains blocks related to the automation of training and testing.
   It has an engine that reduces the boilerplate code needed for training,
   functions for visualisation with visdom_, etc.

lightnet.models
   This submodule has some network and dataset implementations that I felt like sharing.
   Feel free to use them or take a lookt at the implementations and learn how to use this library.

The following tutorials will help you to understand how to use these different parts of the library effectively:

.. toctree::
   :maxdepth: 1

   02-A-basics.ipynb
   02-B-engine.ipynb
   02-C-pascal_voc.rst


*Happy Coding!* ♥ |br|
**~0phoff**


.. include:: ../links.rst
