"""
Lightnet Pruning Module |br|
This module contains classes and functions to prune convolutions in your networks,
in order to make them faster and/or better suited for your application.
"""

from . import dependency

from ._base import *
from ._multi import *
from ._l2 import *
from ._gm import *
