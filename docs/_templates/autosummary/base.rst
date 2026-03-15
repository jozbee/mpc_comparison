{{ fullname | underline }}

.. currentmodule:: {{ module }}

{% if objtype in ['class', 'exception'] %}
.. auto{{ objtype }}:: {{ objname }}
   :show-inheritance:

{% if methods %}
.. rubric:: Methods

.. autosummary::
   :toctree: .
   :nosignatures:

{% for item in methods %}
{% if item != '__init__' %}
   ~{{ fullname }}.{{ item }}
{% endif %}
{% endfor %}

{% endif %}
{% if attributes %}
.. rubric:: Attributes

.. autosummary::
   :toctree: .
   :nosignatures:

{% for item in attributes %}
   ~{{ fullname }}.{{ item }}
{% endfor %}

{% endif %}
{% else %}
.. auto{{ objtype }}:: {{ objname }}
{% endif %}
