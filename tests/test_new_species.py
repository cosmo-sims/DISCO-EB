"""A species with a parameter of its own, added without touching the package.

``SecondDarkMatter`` is a second pressureless fluid with a density parameter
``Omega_x``. It declares the key, the kernel input derived from it and its
own check. The model, the cosmology dictionary, the parameter row, the
background, the budget closure and the forward sensitivities pick it up from
the declarations. Splitting the cold dark matter into two such fluids must
leave the solve unchanged.
"""

import numpy as np
import pytest
from conftest import HOST, ConstantThermalHistory, _cuda_available

from discoeb.component import Parameter, Species, State
from discoeb.models import LCDM


class SecondDarkMatter(Species):
    """Pressureless and at rest in synchronous gauge, like CDM: ``clxx' = -k z``."""

    clxx = State(initial=lambda theta: 0.75 * theta[0])

    Omega_x = Parameter(0.0)
    grhox = Parameter(value=lambda grhom, Omega_x: grhom * Omega_x, directed=True)

    @classmethod
    def check(cls, Omega_x):
        if Omega_x < 0.0:
            raise ValueError(f"Omega_x = {Omega_x} must not be negative")

    def grhox_t(self, grhox, inv_a):
        return grhox * inv_a

    def grho(self, grhox_t):
        return grhox_t

    def dgrho(self, grhox_t, clxx):
        return grhox_t * clxx

    def d_clxx(self, z, k):
        return -k * z


LCDM_X = LCDM.add(SecondDarkMatter, name="LCDM+x")

COSMOLOGY = {
    "Omegam": (0.02238280 + 0.1201075) / 0.6732117**2,
    "Omegab": 0.02238280 / 0.6732117**2,
    "A_s": 2.100549e-9,
    "n_s": 0.9660499,
    "H0": 67.32117,
    "Tcmb": 2.7255,
    "YHe": 0.2454006,
    "Neff": 3.046,
}
OMEGA_X = 0.08


def _split(cosmology, omega_x=OMEGA_X):
    """The same universe with ``omega_x`` of its cold dark matter in the second fluid."""

    return {**cosmology, "Omegam": cosmology["Omegam"] - omega_x, "Omega_x": omega_x}


def test_the_parameter_reaches_the_dictionary_the_row_and_the_inputs():
    from discoeb.kernel import DIRECTED_NAMES, parameter_layout

    assert LCDM_X.parameter_keys()["Omega_x"] == 0.0
    assert "Omega_x" not in LCDM.parameter_keys()
    layout = parameter_layout(LCDM_X)
    assert layout.directed == (*DIRECTED_NAMES, "grhox")
    assert {"grhox_nominal", "dir_grhox"} <= set(layout.columns)
    assert layout.columns["dir_grhox"] == layout.n - 1
    assert layout.directed_columns[-1] == layout.columns["grhox_nominal"]
    # The built-in models' rows have no trace of it.
    assert "grhox_nominal" not in parameter_layout(LCDM).columns


def test_the_components_check_the_parameter():
    LCDM_X(_split(COSMOLOGY))
    LCDM(COSMOLOGY)
    LCDM(_split(COSMOLOGY, 0.0))  # at its default, the key is harmless
    with pytest.raises(ValueError, match="must not be negative"):
        LCDM_X(_split(COSMOLOGY, -0.01))
    with pytest.raises(ValueError, match="use a model with SecondDarkMatter"):
        LCDM(_split(COSMOLOGY))


def test_the_background_closes_the_budget_with_the_new_density():
    from discoeb.background import density_parameters
    from discoeb.cosmology import _columns

    def densities(model, cosmology):
        columns = _columns([cosmology], held=False, keys=model.parameter_keys())
        return {
            k: float(v[0]) for k, v in density_parameters(columns, model=model).items()
        }

    lcdm = densities(LCDM, COSMOLOGY)
    split = densities(LCDM_X, _split(COSMOLOGY))
    assert split["grhox"] == pytest.approx(OMEGA_X * split["grhom"], rel=1e-15)
    assert split["grhoc"] + split["grhox"] == pytest.approx(lcdm["grhoc"], rel=1e-14)
    assert split["grhov"] == pytest.approx(lcdm["grhov"], rel=1e-12)
    assert "grhox" not in lcdm


def test_the_right_hand_side_reads_the_generated_provider():
    """``grhox = grhox_nominal + eps * dir_grhox``, and clxx evolves as clxc does."""

    from discoeb.kernel import parameter_layout

    columns = parameter_layout(LCDM_X).columns
    model = LCDM_X.build(ConstantThermalHistory())
    rhs = model.build(columns, **HOST)
    assert (
        "grhox = SecondDarkMatter__grhox(SecondDarkMatter_self, grhox_nominal, eps, dir_grhox)"
        in [line.strip() for line in model.source.splitlines()]
    )
    p = np.zeros(len(columns))
    for name, value in dict(
        grhog=1.5e-7, grhornomass=1.0e-7, grhoc=0.8e-4, grhob=1.9e-5, grhov=2.0e-7
    ).items():
        p[columns[f"{name}_nominal"]] = value
    p[columns["grhox_nominal"]] = 0.3e-4
    p[columns["k"]] = 0.05
    y = np.random.default_rng(0).standard_normal(model.nvar)
    y[model.index["clxx"]] = y[model.index["clxc"]]
    dy = np.asarray(rhs(tuple(y), 100.0, tuple(p)))
    assert np.isfinite(dy).all()
    assert dy[model.index["clxx"]] == dy[model.index["clxc"]]
    # Moving grhox along its direction moves dgrho as moving grhoc does.
    for name in ("grhox", "grhoc"):
        q = p.copy()
        q[columns["eps"]], q[columns[f"dir_{name}"]] = 1.0, 1.0e-5
        moved = np.asarray(rhs(tuple(y), 100.0, tuple(q)))
        if name == "grhox":
            along_x = moved - dy
        else:
            np.testing.assert_allclose(moved - dy, along_x, rtol=1e-6, atol=1e-18)
    pattern = {(int(r), int(c)) for r, c in model.sparsity()}
    assert (model.index["etak"], model.index["clxx"]) in pattern


SOLVE_K = np.geomspace(1.0e-3, 0.3, 8)
SOLVE_TAU = np.geomspace(1.0, 2.0e4, 16)  # the last lies past tau0


@pytest.mark.skipif(not _cuda_available(), reason="needs a CUDA GPU")
def test_splitting_the_dark_matter_leaves_the_solve_unchanged():
    """Two pressureless fluids holding the CDM between them evolve as the CDM did.

    The second fluid's contrast equals the CDM's in the same solve, the
    contrasts and the curvature perturbation agree with the plain LCDM solve
    to the solver's tolerance, and so do the conformal ages. The directed
    input ``grhox`` moves every kept quantity as ``grhoc`` does.
    """

    import jax
    import jax.numpy as jnp

    from discoeb import directed_inputs, solve_eb

    names = ("clxc", "R", "theta_2")
    plain = solve_eb(LCDM(COSMOLOGY), SOLVE_K, SOLVE_TAU, keep=names)
    batch = LCDM_X(_split(COSMOLOGY))
    split = solve_eb(batch, SOLVE_K, SOLVE_TAU, keep=(*names, "clxx"))

    np.testing.assert_allclose(split.tau0, plain.tau0, rtol=1e-12)
    np.testing.assert_array_equal(split["clxx"], split["clxc"])
    for name in names:
        a, b = np.asarray(split[name]), np.asarray(plain[name])
        assert np.abs(a - b).max() < 1e-3 * np.abs(b).max(), name

    x = directed_inputs(batch)
    assert x.shape == (1, 10)

    def kept(inputs):
        return solve_eb(batch, SOLVE_K, SOLVE_TAU, keep=names, inputs=inputs).values

    jac = np.asarray(jax.jacfwd(kept)(jnp.asarray(x)))[..., 0, :]
    assert np.isfinite(jac).all()
    along_c, along_x = jac[..., 2], jac[..., 9]
    scale = np.abs(along_c).max(axis=(-2, -1), keepdims=True)
    assert np.abs(along_x - along_c).max() < 1e-3 * scale.max()
