"""
Lightnet Engine Module |br|
This module contains classes and functions to manage the training of your networks.
It has an engine, capable of orchestrating your training and test cycles, and also contains function to easily visualise data with visdom_.
"""

# No __all__ : everything can be passed on here


from .engine import *
from .visual import *
