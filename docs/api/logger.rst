Logger
=======
This package has some basic logging functionality. Here is an example of how to control and use this logger.

.. code:: python

  import lightnet as ln
  ln.log.level = ln.Loglvl.ALL  # Setting loglevel
  ln.log.color = False          # Disable color (eg. for piping output to a file)

  ln.log(Loglvl.DEBUG, 'This is a debug message')
  ln.log(Loglvl.ERROR, 'This is an error message that will raise an error. It will raise this error even if the message is suppressed', ValueError)


.. autoclass:: lightnet.logger.Logger
.. autoclass:: lightnet.logger.Loglvl


.. include:: ../links.rst
