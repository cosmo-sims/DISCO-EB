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
    _qcdm_densities,
    _split,
    by_name,
    cosmologies_of,
)

from discoeb.models import LCDM

# --- One component listed several times, each copy under a name of its own.


OMEGA_FLUID = 0.08


@pytest.mark.slow
@pytest.mark.gpu
def test_splitting_the_dark_matter_into_a_named_fluid_leaves_the_solve_unchanged():
    from discoeb import conformal_ages, solve_eb

    model = LCDM.add(X, name="LCDM+x")
    k = np.geomspace(1.0e-3, 0.3, 8)
    tau = np.geomspace(1.0, 2.0e4, 16)
    lcdm = cosmologies_of(LCDM, COSMOLOGY)
    with_fluid = model.from_dict(
        {
            **COSMOLOGY,
            "Omegam": COSMOLOGY["Omegam"] - OMEGA_FLUID,
            "x_Omega": OMEGA_FLUID,
        }
    )
    plain = by_name(solve_eb(lcdm, k, tau, keep=("clxc", "R")), ("clxc", "R"))
    keep = ("clxc", "R", "x_delta")
    split = by_name(solve_eb(with_fluid, k, tau, keep=keep), keep)
    np.testing.assert_allclose(
        conformal_ages(with_fluid), conformal_ages(lcdm), rtol=1e-12
    )
    np.testing.assert_allclose(split["x_delta"], split["clxc"], rtol=1e-5)
    for name in ("clxc", "R"):
        a, b = split[name], plain[name]
        assert np.abs(a - b).max() < 1e-3 * np.abs(b).max(), name


# --- A species with a parameter of its own, added without touching the package.


SOLVE_K = np.geomspace(1.0e-3, 0.3, 8)


SOLVE_TAU = np.geomspace(1.0, 2.0e4, 16)  # the last lies past tau0


@pytest.mark.slow
@pytest.mark.gpu
def test_splitting_the_dark_matter_leaves_the_solve_unchanged():
    """Two pressureless fluids holding the CDM between them evolve as the CDM did.

    The second fluid's contrast equals the CDM's in the same solve, the
    contrast and the curvature perturbation agree with the plain LCDM solve
    to the solver's tolerance, and so do the conformal ages. The directed
    input ``grhox`` moves every kept quantity as ``grhoc`` does.
    """

    import jax
    import jax.numpy as jnp

    from discoeb import conformal_ages, directed_inputs, solve_eb

    names = ("clxc", "R", "theta_2")
    lcdm = cosmologies_of(LCDM, COSMOLOGY)
    plain = by_name(solve_eb(lcdm, SOLVE_K, SOLVE_TAU, keep=names), names)
    batch = cosmologies_of(LCDM_X, _split(COSMOLOGY))
    keep = (*names, "clxx")
    split = by_name(solve_eb(batch, SOLVE_K, SOLVE_TAU, keep=keep), keep)

    np.testing.assert_allclose(conformal_ages(batch), conformal_ages(lcdm), rtol=1e-12)
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
        return solve_eb(batch, SOLVE_K, SOLVE_TAU, keep=names, inputs=inputs)

    jac = np.asarray(jax.jacfwd(kept)(jnp.asarray(x)))[..., 0, :]
    assert np.isfinite(jac).all()
    along_c = jac[..., directed.index("grhoc")]
    along_x = jac[..., directed.index("grhox")]
    scale = np.abs(along_c).max(axis=(-2, -1), keepdims=True)
    assert np.abs(along_x - along_c).max() < 1e-3 * scale.max()


# --- A scalar field from outside the package: background states, and a budget closed by shooting.


def _power(model, cosmology, k):
    from discoeb import conformal_ages, matter_power_spectrum, solve_eb

    batch = cosmologies_of(model, [cosmology])
    tau0 = conformal_ages(batch)[0]
    keep = ("clxc", "clxb", "z", "vb", "a")
    values = solve_eb(batch, k, tau0.reshape(1), keep=keep, thermo_weight=0.1)
    today = {n: v[0, :, -1] for n, v in by_name(values, keep).items()}
    a_today = today.pop("a")
    pk = matter_power_spectrum(
        k,
        **today,
        tau0=tau0,
        **{n: cosmology[n] for n in ("Omegam", "Omegab", "A_s", "n_s", "k_p")},
    )
    return np.asarray(pk), a_today


def _growth(model, cosmology):
    """The matter growth factor today in the model's background: ``delta'' + aH delta' = 4 pi G a^2 rho_m delta``.

    Heun steps in ``ln a`` on the background's own grid, from ``a = 1e-5``.
    Only the ratio between two models that agree early on is meaningful.
    """

    import jax.numpy as jnp

    from discoeb.solver._background import _background_solution, background_model

    d = _qcdm_densities(model, cosmology)
    a, _, states = _background_solution(d, model)
    rate = np.asarray(
        background_model(model).evaluate("adotoa", math=jnp, a=a, **states, **d)
    )
    a = np.asarray(a)
    source = 0.5 * float(d["grhoc"] + d["grhob"]) / a
    lna = np.log(a)
    delta, slope = 1.0, 0.0
    for i in range(int(np.searchsorted(a, 1.0e-5)), len(a) - 1):
        h = lna[i + 1] - lna[i]
        f0 = (slope / rate[i], (-rate[i] * slope + source[i] * delta) / rate[i])
        d1, s1 = delta + h * f0[0], slope + h * f0[1]
        f1 = (
            s1 / rate[i + 1],
            (-rate[i + 1] * s1 + source[i + 1] * d1) / rate[i + 1],
        )
        delta += 0.5 * h * (f0[0] + f1[0])
        slope += 0.5 * h * (f0[1] + f1[1])
    return delta


@pytest.mark.slow
@pytest.mark.gpu
def test_the_solve_matches_lcdm_and_the_growth_in_the_fields_background():
    """A flat potential gives LCDM, and a rolling field suppresses ``P(k)`` as its expansion history says.

    Inside the horizon a field with unit sound speed does not cluster, so
    the power relative to LCDM is the squared ratio of the growth factors
    in the two backgrounds, which the host integrates on its own. (A CPL
    fluid through the field's equation of state would cross ``w = -1`` in
    the past, which the fluid equations cannot.)
    """

    k = np.geomspace(1.0e-3, 0.3, 8)
    lcdm, _ = _power(LCDM, QCDM_COSMOLOGY, k)
    flat, a_flat = _power(QCDM, QCDM_COSMOLOGY, k)
    np.testing.assert_allclose(a_flat, 1.0, rtol=1e-4)
    np.testing.assert_allclose(flat, lcdm, rtol=2e-3)

    rolling = {**QCDM_COSMOLOGY, "lam": LAM}
    field, a_field = _power(QCDM, rolling, k)
    # the kernel's own integration of the field ends today where the host's did
    np.testing.assert_allclose(a_field, 1.0, rtol=1e-4)
    suppression = field / lcdm
    growth = (_growth(QCDM, rolling) / _growth(LCDM, QCDM_COSMOLOGY)) ** 2
    assert 0.9 < growth < 0.99  # dark energy that sets in earlier: less growth
    np.testing.assert_allclose(suppression[k > 0.01], growth, rtol=1e-3)
    # near the horizon the field's own perturbation takes part
    assert suppression[0] > suppression[-1] + 3e-3


# --- GPU perturbation solves against CLASS: the matter power spectrum and its derivatives.


SOLVE_EB_TAU = np.geomspace(1.0, 2.0e4, 48)  # the last few lie past every tau0


@pytest.mark.gpu
def test_solve_eb_keeps_what_the_history_holds():
    """The hooked launch records exactly what the full history evaluates to.

    State elements are copied from the same dense-output states the history
    launch saves; the derived quantities (the comoving curvature perturbation,
    the shear, the expansion rate) are the components' equations compiled into
    the hook, checked here against the same equations evaluated on the host
    over the saved history. Times past a cosmology's ``tau0`` hold its state
    today, so the CDM and baryon contrasts at the last time give the batch's
    ``P(k)``.
    """

    from discoeb.components.recombination import THERMO_ERROR_WEIGHT
    from discoeb.models import LCDM
    from discoeb.solver._background import TAU_START, density_coefficients
    from discoeb.solver._kernel import make_params, parameter_layout
    from discoeb.solver._solver import (
        conformal_ages,
        solve_eb,
        solve_perturbation_history_batch,
    )

    cosmologies = _perturbed_cosmologies(2)
    names = ("R", "clxc", "clxb", "theta_2", "n_15", "sigma", "adotoa")
    values = solve_eb(
        cosmologies_of(LCDM, cosmologies), SOLVE_EB_K, SOLVE_EB_TAU, keep=names
    )
    n_k, n_tau = len(SOLVE_EB_K), len(SOLVE_EB_TAU)
    assert values.shape == (2, len(names), n_k, n_tau)
    assert np.all(conformal_ages(cosmologies_of(LCDM, cosmologies)) < SOLVE_EB_TAU[-1])

    hist, k_sorted, prepared = solve_perturbation_history_batch(
        SOLVE_EB_K,
        cosmologies_of(LCDM, cosmologies),
        tau_save=np.concatenate(([TAU_START], SOLVE_EB_TAU)),
    )
    hist = np.asarray(hist)[:, :, 1:, :]  # (n_k, n_c, n_tau, nvar), ascending k
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

        # The hook sees the time clamped to the trajectory's own tau0.
        tau = np.minimum(SOLVE_EB_TAU, float(prepared.tau0[ci]))[None, :]
        row = make_params(
            cosmology, k_sorted, float(prepared.tau0[ci]), model=LCDM.spec
        )
        columns = parameter_layout(LCDM).columns
        base = {name: row[:, col][:, None] for name, col in columns.items()}
        base.update(model.state_values(hist[:, ci]))
        base.update(tau=tau, k=k_sorted[:, None])
        R, sigma, adotoa = model.evaluate("R", "sigma", "adotoa", math=np, **base)
        for name, expected in (("R", R), ("sigma", sigma), ("adotoa", adotoa)):
            j = names.index(name)
            np.testing.assert_allclose(kept[ci, j], expected, rtol=2e-6, atol=1e-12)

    # the same thermal error weight as solve_eb's launch, so the two solves agree bit for bit
    pk_batch = _matter_power(
        SOLVE_EB_K, cosmologies_of(LCDM, cosmologies), thermo_weight=THERMO_ERROR_WEIGHT
    )
    for ci, cosmology in enumerate(cosmologies_of(LCDM, cosmologies)):
        grhoc, grhob = density_coefficients(cosmology)[2:4]
        clxc = np.asarray(values[ci, names.index("clxc"), :, -1])
        clxb = np.asarray(values[ci, names.index("clxb"), :, -1])
        delta_m = (grhoc * clxc + grhob * clxb) / (grhoc + grhob)
        k = np.asarray(SOLVE_EB_K)
        pk = (
            2.0
            * np.pi**2
            / k**3
            * cosmology.A_s
            * (k / cosmology.k_pivot) ** (cosmology.n_s - 1.0)
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
    the potentials. A one-quantity solve stores ``n_k * n_tau`` numbers.
    """

    from discoeb.models import LCDM
    from discoeb.solver._solver import solve_eb

    k = np.array([1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1])
    tau = np.geomspace(1.0, 2.0e4, 64)
    values = solve_eb(
        cosmologies_of(LCDM, _perturbed_cosmologies(1)[0]), k, tau, keep=("R",)
    )
    assert values.shape == (1, 1, len(k), len(tau))
    R = np.asarray(values[0, 0])
    assert np.isfinite(R).all()

    assert abs(R[0, 0] - 1.0) < 1.0e-6  # k tau = 1e-4
    super_horizon = tau[None, :] * k[:, None] < 0.1
    assert np.all(np.abs(R[super_horizon] - 1.0) < 5.0e-3), np.abs(
        R[super_horizon] - 1.0
    ).max()
    # k = 0.1 Mpc^-1 entered the horizon long before today and R has decayed.
    assert abs(R[-1, -1]) < 0.5
