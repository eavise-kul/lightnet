{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}


{% if objtype == 'class' %}

.. autoclass:: {{ objname }}
   :members:

{% else %}

.. auto{{ objtype }}:: {{ objname }}

{% endif %}

.. include:: /links.rst
..
  autogenerated from docs/.templates/base-template.rst
