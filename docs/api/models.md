# Models

A model is a class whose bases are its components, and a cosmology is an
instance of it. See [Building models](../usage/models.md) and [Creating
cosmologies](../usage/cosmologies.md).

```{eval-rst}
.. autoclass:: discoeb.Model
   :members: from_density_parameters, from_dict, H_0, to_dict, replace, parameter_keys, state_names, setting, sparsity, validate, swap, add, remove, with_settings
```

## Built-in models

```{eval-rst}
.. autoclass:: discoeb.LCDM
   :no-members:

.. autoclass:: discoeb.LCDM_MNU
   :no-members:

.. autoclass:: discoeb.W0WACDM
   :no-members:

.. autoclass:: discoeb.W0WACDM_MNU
   :no-members:
```
