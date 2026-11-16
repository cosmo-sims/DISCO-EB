"""Cosmologies as the solver reads them (``_cosmology``): dictionaries, columns, what is differentiable."""

import pytest
from helpers import COSMOLOGY, cosmologies_of, physical

from discoeb.models import LCDM

# --- A species with a parameter of its own, added without touching the package.


def test_a_dictionary_names_no_model():
    """No model stands in for a bare dictionary: the caller names one."""

    from discoeb import directed_inputs

    with pytest.raises(TypeError, match="LCDM"):
        directed_inputs(COSMOLOGY)
    directed_inputs(cosmologies_of(LCDM, COSMOLOGY))


# --- The columns of a batch, and the keys the solve holds fixed.


def test_columns_fill_defaults_and_require_the_rest():
    import numpy as np
    from helpers import REFERENCE_COSMOLOGY

    from discoeb.models._cosmology import _columns

    keys = LCDM.parameter_keys()
    reference = physical(LCDM, REFERENCE_COSMOLOGY)
    lean = {k: v for k, v in reference.items() if k in ("omega_b", "omega_c")}
    with pytest.raises(KeyError):
        _columns([lean], keys=keys)
    names = ("omega_b", "omega_c", "omega_de", "Tcmb", "YHe", "Neff", "A_s", "n_s")
    required = {k: reference[k] for k in names}
    columns = _columns([required, {**required, "omega_c": 0.13}], keys=keys)
    np.testing.assert_array_equal(columns["omega_c"], [required["omega_c"], 0.13])
    assert columns["omega_k"].shape == (2,) and float(columns["omega_k"][0]) == 0.0
    assert "not_a_parameter" not in _columns(
        [{**required, "not_a_parameter": 1.0}], keys=keys
    )


def test_a_held_key_raises_when_differentiated():
    """``YHe`` is read by the solve but not differentiated: no silent zero."""
    import jax
    from helpers import REFERENCE_COSMOLOGY

    from discoeb.models import LCDM
    from discoeb.models._cosmology import _columns

    reference = physical(LCDM, REFERENCE_COSMOLOGY)

    def column(key):
        def of(value):
            cosmology = {**reference, key: value}
            differentiable = LCDM.spec.differentiable_parameters()
            columns = _columns(
                [cosmology], LCDM.parameter_keys(), differentiable=differentiable
            )
            return columns[key][0]

        return jax.grad(of)(reference[key])

    assert float(column("omega_c")) == 1.0
    with pytest.raises(Exception, match="YHe"):
        column("YHe")
