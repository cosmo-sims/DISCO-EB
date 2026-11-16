"""Cosmologies as the solver reads them (``_cosmology``): dictionaries, columns, what is differentiable."""

import pytest
from helpers import COSMOLOGY, LCDM_X, _split, cosmologies_of, physical

from discoeb.models import LCDM

# --- A species with a parameter of its own, added without touching the package.


def test_a_dictionary_names_no_model():
    """No model stands in for a bare dictionary: the caller names one."""

    from discoeb import (
        LCDM,
        background_history,
        conformal_ages,
        density_coefficients,
        directed_inputs,
    )

    functions = (conformal_ages, density_coefficients, directed_inputs)
    for background_of in (*functions, background_history):
        with pytest.raises(TypeError, match="LCDM"):
            background_of(COSMOLOGY)
    lcdm = cosmologies_of(LCDM, COSMOLOGY)
    assert float(
        conformal_ages(cosmologies_of(LCDM_X, _split(COSMOLOGY)))[0]
    ) == pytest.approx(float(conformal_ages(lcdm)[0]), rel=1e-12)


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
            return _columns([{**reference, key: value}], LCDM.parameter_keys())[key][0]

        return jax.grad(of)(reference[key])

    assert float(column("omega_c")) == 1.0
    with pytest.raises(Exception, match="YHe"):
        column("YHe")


def test_as_cosmology_is_hashable_and_takes_an_instance():
    from helpers import REFERENCE_COSMOLOGY

    from discoeb import as_cosmology

    reference = physical(LCDM, REFERENCE_COSMOLOGY)
    from_dict = as_cosmology(reference)
    from_instance = as_cosmology(LCDM.from_dict(reference))
    assert hash(from_dict) == hash(from_instance) and from_dict == from_instance
    assert from_dict.omega_c == reference["omega_c"]
