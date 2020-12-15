Prune
=====
.. automodule:: lightnet.prune

.. warning::
   You need the onnx_ library in order to prune networks,
   as it is used to build the dependency map of a network.


Methods
-------
Lightnet contains different methods to prune the channels in a convolution. 

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: callmember-template.rst

   lightnet.prune.L2Pruner
   lightnet.prune.GMPruner
   lightnet.prune.MultiPruner
   lightnet.prune.Pruner


Dependency Map
--------------
Before being able to prune networks, it is important to build up a dependency map,
in order to know which layers to adapt once we prune a certain convolution.
The following functions and classes are used to compute this dependency map
and allow you to add operations which might not be supported by our software for pruning.

.. note::
   In most cases, you should not need to call any of these functions,
   they are merely here for completion sake and more advanced users.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: base-template.rst

   lightnet.prune.dependency.get_onnx_model
   lightnet.prune.dependency.get_dependency_map
   lightnet.prune.dependency.print_dependency_map
   lightnet.prune.dependency.traverse_breadth_first
   lightnet.prune.dependency.traverse_depth_first
   lightnet.prune.dependency.NodeType
   lightnet.prune.dependency.add_nodetype_operation


.. include:: /links.rst
