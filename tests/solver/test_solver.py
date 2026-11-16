"""``solve_eb`` (``_solver``): kept quantities, and solves of models with components added in the tests."""

import numpy as np
import pytest
from helpers import (
    COSMOLOGY,
    LAM,
    LCDM_X,
    QCDM,
    QCDM_COSMOLOGY,
    SOLVE_EB_K,
    X,
    _matter_power,
    _perturbed_cosmologies,
    _split,
    by_name,
    cosmologies_of,
    model_parameters,
    qcdm_parameters,
)

from discoeb.models import LCDM

# --- One component listed several times, each copy under a name of its own.


OMEGA_FLUID = 0.08


@pytest.mark.slow
@pytest.mark.gpu
def test_splitting_the_dark_matter_into_a_named_fluid_leaves_the_solve_unchanged():
    from discoeb import solve_eb

    model = LCDM.add(X, name="LCDM+x")
    k = np.geomspace(1.0e-3, 0.3, 8)
    a = np.geomspace(1.0e-6, 1.0, 16)
    lcdm = cosmologies_of(LCDM, COSMOLOGY)
    with_fluid = cosmologies_of(
        model,
        {
            **COSMOLOGY,
            "Omegam": COSMOLOGY["Omegam"] - OMEGA_FLUID,
            "x_Omega": OMEGA_FLUID,
        },
    )
    names = ("clxc", "R", "tau")
    plain = by_name(solve_eb(lcdm, k, a, keep=names), names)
    keep = (*names, "x_delta")
    split = by_name(solve_eb(with_fluid, k, a, keep=keep), keep)
    # the same expansion: the conformal time, a state, comes out the same
    np.testing.assert_allclose(split["tau"], plain["tau"], rtol=1e-5)
    np.testing.assert_allclose(split["x_delta"], split["clxc"], rtol=1e-5)
    for name in ("clxc", "R"):
        ours, theirs = split[name], plain[name]
        assert np.abs(ours - theirs).max() < 1e-3 * np.abs(theirs).max(), name


# --- A species with a parameter of its own, added without touching the package.


SOLVE_K = np.geomspace(1.0e-3, 0.3, 8)


SOLVE_A = np.geomspace(1.0e-6, 1.0, 16)


@pytest.mark.slow
@pytest.mark.gpu
def test_splitting_the_dark_matter_leaves_the_solve_unchanged():
    """Two pressureless fluids holding the CDM between them evolve as the CDM did.

    The second fluid's contrast equals the CDM's in the same solve, the
    contrast and the curvature perturbation agree with the plain LCDM solve
    to the solver's tolerance, and so does the conformal time. The directed
    input ``grhox`` moves every kept quantity as ``grhoc`` does.
    """

    import jax
    import jax.numpy as jnp

    from discoeb import directed_inputs, solve_eb

    names = ("clxc", "R", "theta_2", "tau")
    lcdm = cosmologies_of(LCDM, COSMOLOGY)
    plain = by_name(solve_eb(lcdm, SOLVE_K, SOLVE_A, keep=names), names)
    batch = cosmologies_of(LCDM_X, _split(COSMOLOGY))
    keep = (*names, "clxx")
    split = by_name(solve_eb(batch, SOLVE_K, SOLVE_A, keep=keep), keep)

    np.testing.assert_allclose(split["tau"], plain["tau"], rtol=1e-5)
    names = names[:-1]
    # The same equations and initial state; the fp32 LU tells the columns apart.
    np.testing.assert_allclose(split["clxx"], split["clxc"], rtol=1e-5)
    # The initial states agree (the growing mode's om sums grho_matter, which
    # the second fluid contributes to), so what is left is the solver's:
    # the smooth quantities to 1e-3, the oscillating photon quadrupole,
    # stepped differently by a system one state larger, to a few times that.
    for name, gate in (("clxc", 1e-3), ("R", 1e-3), ("theta_2", 5e-3)):
        a, b = split[name], plain[name]
        assert np.abs(a - b).max() < gate * np.abs(b).max(), name

    from discoeb.solver._kernel import parameter_layout

    directed = parameter_layout(LCDM_X).directed
    x = directed_inputs(batch)
    assert x.shape == (1, len(directed))

    def kept(inputs):
        return solve_eb(batch, SOLVE_K, SOLVE_A, keep=names, inputs=inputs)

    jac = np.asarray(jax.jacfwd(kept)(jnp.asarray(x)))[..., 0, :]
    assert np.isfinite(jac).all()
    along_c = jac[..., directed.index("grhoc")]
    along_x = jac[..., directed.index("grhox")]
    scale = np.abs(along_c).max(axis=(-2, -1), keepdims=True)
    assert np.abs(along_x - along_c).max() < 1e-3 * scale.max()


# --- A scalar field from outside the package: background states, and no budget to close.


def _power(model, cosmology, k):
    from discoeb import matter_power_spectrum, solve_eb

    cosmology = cosmologies_of(model, cosmology)
    keep = ("clxc", "clxb")
    values = solve_eb(cosmology, k, [1.0], keep=keep, thermo_weight=0.1)
    today = {n: v[0, :, -1] for n, v in by_name(values, keep).items()}
    names = ("omega_c", "omega_b", "A_s", "n_s", "k_p")
    return np.asarray(
        matter_power_spectrum(k, **today, **{n: getattr(cosmology, n) for n in names})
    )


GROWTH_A = np.geomspace(1.0e-5, 1.0, 2000)


def _growth(model, cosmology):
    """``(growth factor, H0)`` in the model's background, whose expansion rate the solve itself gives.

    The matter growth factor today, ``delta'' + aH delta' = 4 pi G a^2 rho_m
    delta``, by Heun steps in ``ln a`` from ``a = 1e-5`` with ``adotoa`` kept
    on that grid. Only the ratio between two models that agree early on is
    meaningful. ``H0`` in km/s/Mpc is the rate today.
    """

    from discoeb import solve_eb
    from discoeb.components.gravity import GRHOM_PER_H0SQ

    cosmology = cosmologies_of(model, cosmology)
    a = GROWTH_A
    rate = np.asarray(solve_eb(cosmology, [1.0e-2], a, keep=("adotoa",)))[0, 0, 0]
    d = model_parameters(model, cosmology)
    source = 0.5 * float(d["grhoc"] + d["grhob"]) / a
    lna = np.log(a)
    delta, slope = 1.0, 0.0
    for i in range(len(a) - 1):
        h = lna[i + 1] - lna[i]
        f0 = (slope / rate[i], (-rate[i] * slope + source[i] * delta) / rate[i])
        d1, s1 = delta + h * f0[0], slope + h * f0[1]
        f1 = (
            s1 / rate[i + 1],
            (-rate[i + 1] * s1 + source[i + 1] * d1) / rate[i + 1],
        )
        delta += 0.5 * h * (f0[0] + f1[0])
        slope += 0.5 * h * (f0[1] + f1[1])
    return delta, float(rate[-1]) * np.sqrt(3.0 / GRHOM_PER_H0SQ)


@pytest.mark.slow
@pytest.mark.gpu
def test_the_solve_matches_lcdm_and_the_growth_in_the_fields_background():
    """A flat potential gives LCDM, and a rolling field changes ``P(k)`` as its expansion history says.

    The field's potential is a parameter and no budget closes, so its Hubble
    constant comes out of the solve: at LCDM's height a rolling field has
    lost density by today and expands more slowly. Inside the horizon a
    field with unit sound speed does not cluster, so the power relative to
    LCDM is the squared ratio of the growth factors in the two backgrounds,
    integrated here on the host from the expansion rate the solve gives.
    """

    k = np.geomspace(1.0e-3, 0.3, 8)
    lcdm = _power(LCDM, QCDM_COSMOLOGY, k)
    flat = _power(QCDM, qcdm_parameters(), k)
    np.testing.assert_allclose(flat, lcdm, rtol=2e-3)

    rolling = qcdm_parameters(lam=LAM)
    field = _power(QCDM, rolling, k)
    growth_lcdm, h0_lcdm = _growth(LCDM, QCDM_COSMOLOGY)
    growth_field, h0_field = _growth(QCDM, rolling)
    assert h0_lcdm == pytest.approx(QCDM_COSMOLOGY["H0"], rel=1e-4)
    assert 0.9 * h0_lcdm < h0_field < h0_lcdm
    # the field is background: what the hook reads of it does not know the
    # wave number (it did until 2026-11-17, when the joint state was not laid
    # out as the solver's two blocks are and the hook read the perturbation)
    from discoeb import solve_eb

    kept = np.asarray(
        solve_eb(cosmologies_of(QCDM, rolling), k, [1.0], keep=("phi", "phidot"))
    )[0, :, :, -1]
    np.testing.assert_allclose(kept, kept[:, :1] * np.ones(len(k)), rtol=1e-6)
    assert np.all(kept > 0.0)
    ratio = field / lcdm
    growth = (growth_field / growth_lcdm) ** 2
    assert abs(growth - 1.0) > 3e-3  # the field's history is not LCDM's
    np.testing.assert_allclose(ratio[k > 0.01], growth, rtol=1e-3)
    # near the horizon the field's own perturbation takes part
    assert abs(ratio[0] - ratio[-1]) > 3e-3


# --- GPU perturbation solves against CLASS: the matter power spectrum and its derivatives.


SOLVE_EB_A = np.geomspace(1.0e-6, 1.0, 48)


@pytest.mark.gpu
def test_solve_eb_keeps_what_the_history_holds():
    """The hooked launch records exactly what the full history evaluates to.

    State elements are copied from the same dense-output states the history
    launch saves; the derived quantities (the comoving curvature perturbation,
    the shear, the expansion rate) are the components' equations compiled into
    the hook, checked here against the same equations evaluated on the host
    over the saved history. The last scale factor is today, so the CDM and
    baryon contrasts there give the batch's ``P(k)``.
    """

    from discoeb.components.recombination import THERMO_ERROR_WEIGHT
    from discoeb.models import LCDM
    from discoeb.solver._kernel import A_START, make_params, parameter_layout
    from discoeb.solver._solver import solve_eb, solve_perturbation_history_batch

    cosmologies = _perturbed_cosmologies(2)
    names = ("R", "clxc", "clxb", "theta_2", "n_15", "sigma", "adotoa")
    values = solve_eb(
        cosmologies_of(LCDM, cosmologies), SOLVE_EB_K, SOLVE_EB_A, keep=names
    )
    n_k, n_a = len(SOLVE_EB_K), len(SOLVE_EB_A)
    assert values.shape == (2, len(names), n_k, n_a)

    hist, k_sorted, prepared = solve_perturbation_history_batch(
        SOLVE_EB_K,
        cosmologies_of(LCDM, cosmologies),
        a_save=np.concatenate(([A_START], SOLVE_EB_A)),
    )
    hist = np.asarray(hist)[:, :, 1:, :]  # (n_k, n_c, n_a, nvar), ascending k
    order = np.argsort(SOLVE_EB_K)
    kept = np.asarray(values)[:, :, order, :]  # ascending k too
    model = (
        prepared.model
    )  # the joint model: the thermal quantities come from the thermal states

    for ci, cosmology in enumerate(prepared.cosmologies):
        for name in ("clxc", "clxb", "theta_2", "n_15"):
            j = names.index(name)
            expected = hist[:, ci, :, model.index[name]]
            np.testing.assert_allclose(kept[ci, j], expected, rtol=1e-9, atol=1e-14)

        # The conformal time is one of the states the history holds.
        row = make_params(cosmology, k_sorted, model=LCDM.spec)
        columns = parameter_layout(LCDM).columns
        base = {name: row[:, col][:, None] for name, col in columns.items()}
        base.update(model.state_values(hist[:, ci]))
        base.update(a=SOLVE_EB_A[None, :], k=k_sorted[:, None])
        R, sigma, adotoa = model.evaluate("R", "sigma", "adotoa", math=np, **base)
        for name, expected in (("R", R), ("sigma", sigma), ("adotoa", adotoa)):
            j = names.index(name)
            np.testing.assert_allclose(kept[ci, j], expected, rtol=2e-6, atol=1e-12)

    # the same thermal error weight as solve_eb's launch, so the two solves agree bit for bit
    pk_batch = _matter_power(
        SOLVE_EB_K, cosmologies_of(LCDM, cosmologies), thermo_weight=THERMO_ERROR_WEIGHT
    )
    for ci, cosmology in enumerate(prepared.cosmologies):
        grhoc, grhob = cosmology["omega_c"], cosmology["omega_b"]
        clxc = np.asarray(values[ci, names.index("clxc"), :, -1])
        clxb = np.asarray(values[ci, names.index("clxb"), :, -1])
        delta_m = (grhoc * clxc + grhob * clxb) / (grhoc + grhob)
        k = np.asarray(SOLVE_EB_K)
        pk = (
            2.0
            * np.pi**2
            / k**3
            * cosmology["A_s"]
            * (k / cosmology["k_p"]) ** (cosmology["n_s"] - 1.0)
            * delta_m**2
        )
        np.testing.assert_allclose(pk, pk_batch[ci], rtol=1e-9)


@pytest.mark.slow
@pytest.mark.gpu
def test_solve_eb_comoving_curvature_is_conserved_outside_the_horizon():
    """``R`` starts at unity and stays there until the mode enters the horizon.

    The adiabatic initial conditions set ``R = 1``; on super-horizon scales it
    is conserved up to ``O((k tau)^2)`` (0.13 (k tau)^2 in radiation
    domination with no neutrinos), and once inside the horizon it decays with
    the potentials. The conformal time is kept beside it, a state of the solve.
    """

    from discoeb.models import LCDM
    from discoeb.solver._solver import solve_eb

    k = np.array([1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1])
    a = np.geomspace(2.0e-6, 1.0, 64)
    values = solve_eb(
        cosmologies_of(LCDM, _perturbed_cosmologies(1)[0]), k, a, keep=("R", "tau")
    )
    assert values.shape == (1, 2, len(k), len(a))
    R, tau = np.asarray(values[0, 0]), np.asarray(values[0, 1])
    assert np.isfinite(R).all()

    assert abs(R[0, 0] - 1.0) < 1.0e-6  # k tau = 1e-4
    super_horizon = tau * k[:, None] < 0.1
    assert np.all(np.abs(R[super_horizon] - 1.0) < 5.0e-3), np.abs(
        R[super_horizon] - 1.0
    ).max()
    # k = 0.1 Mpc^-1 entered the horizon long before today and R has decayed.
    assert abs(R[-1, -1]) < 0.5
