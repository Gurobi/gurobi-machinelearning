{{ objname | escape | underline}}

.. automodule:: {{ fullname }}
   :ignore-module-all:

{% block functions %}
{% if functions %}
.. rubric:: {{ _('Functions') }}

.. autosummary::
   :toctree:
{% for item in functions %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block classes %}
{% if classes %}
.. rubric:: {{ _('Classes') }}

.. autosummary::
   :toctree:

{% for item in classes %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
