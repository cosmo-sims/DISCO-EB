"""Cosmologies as the solver reads them (``_cosmology``): dictionaries, columns, what is differentiable."""

import pytest
from helpers import COSMOLOGY, LCDM_X, _split, cosmologies_of

# --- A species with a parameter of its own, added without touching the package.


def test_a_bare_cosmology_cannot_carry_the_parameter_past_the_default_model():
    """Without a batch the default model stands in, and it has no such species."""

    from discoeb import conformal_ages, density_coefficients, directed_inputs

    for background_of in (conformal_ages, density_coefficients, directed_inputs):
        with pytest.raises(ValueError, match="use a model with SecondDarkMatter"):
            background_of(_split(COSMOLOGY))
        background_of(COSMOLOGY)
    assert float(
        conformal_ages(cosmologies_of(LCDM_X, _split(COSMOLOGY)))[0]
    ) == pytest.approx(float(conformal_ages(COSMOLOGY)[0]), rel=1e-12)


# --- The columns of a batch, and the keys the solve holds fixed.


def test_columns_fill_defaults_and_require_the_rest():
    import numpy as np
    from helpers import REFERENCE_COSMOLOGY

    from discoeb.models import LCDM
    from discoeb.models._cosmology import _columns

    keys = LCDM.parameter_keys()
    lean = {k: v for k, v in REFERENCE_COSMOLOGY.items() if k in ("H0", "Omegam")}
    with pytest.raises(KeyError):
        _columns([lean], keys=keys)
    required = {
        k: REFERENCE_COSMOLOGY[k]
        for k in ("H0", "Omegam", "Omegab", "Tcmb", "YHe", "Neff", "A_s", "n_s")
    }
    columns = _columns([required, {**required, "H0": 70.0}], keys=keys)
    np.testing.assert_array_equal(columns["H0"], [required["H0"], 70.0])
    assert columns["Omegak"].shape == (2,) and float(columns["Omegak"][0]) == 0.0
    assert "not_a_parameter" not in _columns(
        [{**required, "not_a_parameter": 1.0}], keys=keys
    )


def test_a_held_key_raises_when_differentiated():
    """``YHe`` is read by the solve but not differentiated: no silent zero."""
    import jax
    from helpers import REFERENCE_COSMOLOGY

    from discoeb.models._cosmology import _columns

    def column(key):
        def of(value):
            return _columns([{**REFERENCE_COSMOLOGY, key: value}])[key][0]

        return jax.grad(of)(REFERENCE_COSMOLOGY[key])

    assert float(column("H0")) == 1.0
    with pytest.raises(Exception, match="YHe"):
        column("YHe")


def test_as_cosmology_is_hashable_and_takes_an_instance():
    from helpers import REFERENCE_COSMOLOGY

    from discoeb import LCDM, as_cosmology

    from_dict = as_cosmology(REFERENCE_COSMOLOGY)
    from_instance = as_cosmology(LCDM.from_dict(REFERENCE_COSMOLOGY))
    assert hash(from_dict) == hash(from_instance) and from_dict == from_instance
    assert from_dict.H0 == REFERENCE_COSMOLOGY["H0"]
