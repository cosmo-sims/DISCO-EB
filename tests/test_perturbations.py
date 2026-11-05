"""GPU perturbation solves against CLASS: the matter power spectrum and its derivatives."""

import time

import numpy as np
import pytest


REFERENCE_COSMOLOGY = {
    "Omegam": (0.02238280 + 0.1201075) / 0.6732117**2,
    "Omegab": 0.02238280 / 0.6732117**2,
    "w_DE_0": -1.0,
    "w_DE_a": 0.0,
    "cs2_DE": 1.0,
    "Omegak": 0.0,
    "A_s": 2.100549e-9,
    "n_s": 0.9660499,
    "H0": 67.32117,
    "Tcmb": 2.7255,
    "YHe": 0.2454006,
    "Neff": 3.046,
    "Nmnu": 0.0,
    "mnu": 0.0,
    "k_p": 0.05,
}

MATTER_POWER_K = np.geomspace(1.0e-4, 1.0, 128, dtype=np.float64)
PK_GATE = 5.0e-3
BATCH_SIZE_CASES = [pytest.param(1, id="N1"), pytest.param(128, id="N128")]


def _perturbed_cosmologies(n):
    """Return deterministic nearby flat LCDM parameter dictionaries."""

    if n == 1:
        return [REFERENCE_COSMOLOGY.copy()]
    phase = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    cosmologies = []
    for ph in phase:
        param = REFERENCE_COSMOLOGY.copy()
        omega_b_h2 = 0.02238280 * (1.0 + 0.006 * ph)
        omega_c_h2 = 0.1201075 * (1.0 - 0.008 * ph)
        h = 0.6732117 * (1.0 + 0.004 * np.sin(np.pi * ph))
        param.update(
            {
                "Omegam": (omega_b_h2 + omega_c_h2) / h**2,
                "Omegab": omega_b_h2 / h**2,
                "H0": 100.0 * h,
                "YHe": REFERENCE_COSMOLOGY["YHe"]
                * (1.0 + 0.003 * np.cos(0.5 * np.pi * ph)),
                "Tcmb": REFERENCE_COSMOLOGY["Tcmb"] * (1.0 + 0.0015 * ph),
                "Neff": REFERENCE_COSMOLOGY["Neff"]
                * (1.0 + 0.002 * np.sin(2.0 * np.pi * ph)),
            }
        )
        cosmologies.append(param)
    return cosmologies


def _successive_batches(n_batches, n_cosmologies):
    """``n_batches`` batches of ``n_cosmologies`` distinct nearby cosmologies.

    The ``n_batches * n_cosmologies`` cosmologies of :func:`_perturbed_cosmologies`
    dealt out in turn, so every batch spans the whole parameter range the
    single-batch tests' 128 do and takes comparable solver steps. The pattern
    of an inference loop: batch after batch of fresh parameters, all of one
    shape.
    """

    cosmologies = _perturbed_cosmologies(n_batches * n_cosmologies)
    return [cosmologies[i::n_batches] for i in range(n_batches)]


def _warmup_cosmologies(n):
    """``n`` cosmologies distinct from :func:`_perturbed_cosmologies` of the same ``n``.

    What a timing warms up on so the timed call, on ``_perturbed_cosmologies(n)``
    itself, finds its compiled programs and nothing of its own in the caches:
    it then pays everything a fresh batch pays, the thermodynamics included.
    """

    return _perturbed_cosmologies(n + 1)[1:]


def _wall_time(run):
    """``(seconds, result)`` of one ``run()``."""

    start = time.perf_counter()
    result = run()
    return time.perf_counter() - start, result


def _class_linear_pk(param, k_values):
    """Return the CLASS linear matter spectrum in Mpc cubed at redshift zero."""

    Class = pytest.importorskip("classy").Class
    h = param["H0"] / 100.0
    cosmo = Class()
    cosmo.set(
        {
            "h": h,
            "omega_b": param["Omegab"] * h**2,
            "omega_cdm": (param["Omegam"] - param["Omegab"]) * h**2,
            "T_cmb": param["Tcmb"],
            "YHe": param["YHe"],
            "N_ur": param["Neff"],
            "N_ncdm": 0,
            "Omega_k": 0.0,
            "A_s": param["A_s"],
            "n_s": param["n_s"],
            "k_pivot": 0.05,
            "output": "mPk",
            "P_k_max_1/Mpc": float(k_values[-1]) * 1.1,
            "z_max_pk": 0.0,
        }
    )
    try:
        cosmo.compute()
        return np.array([cosmo.pk_lin(float(k), 0.0) for k in k_values])
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()


@pytest.mark.parametrize("n_cosmologies", BATCH_SIZE_CASES)
def test_matter_power_spectrum_matches_class(n_cosmologies, benchmark):
    """Check and time one GPU launch over nearby flat LCDM cosmologies."""

    from discoeb.perturbations import solve_matter_power_spectrum_batch

    cosmologies = _perturbed_cosmologies(n_cosmologies)
    pk_ours = benchmark.pedantic(
        solve_matter_power_spectrum_batch,
        args=(MATTER_POWER_K, cosmologies),
        rounds=1,
        warmup_rounds=1,
        iterations=1,
    )
    pk_class = np.stack(
        [_class_linear_pk(param, MATTER_POWER_K) for param in cosmologies]
    )
    rel = np.abs(pk_ours / pk_class - 1.0)
    worst = np.unravel_index(np.argmax(rel), rel.shape)
    assert float(rel[worst]) < PK_GATE, (
        f"max relative error {rel[worst]:.6g} for cosmology {worst[0]} "
        f"at k={MATTER_POWER_K[worst[1]]:.6g} Mpc^-1"
    )


DENSITY_NAMES = ("grhog", "grhornomass", "grhoc", "grhob", "grhov")
DERIV_GATE = 1.0e-3


@pytest.mark.parametrize(
    "one_column_per_trajectory",
    [
        pytest.param(True, id="one-column-per-thread"),
        pytest.param(False, id="five-columns-per-thread"),
    ],
)
def test_matter_power_spectrum_derivatives(one_column_per_trajectory, benchmark):
    """Time and check dP(k)/d(background densities) by forward sensitivity.

    The five densities enter both the hierarchy and the matter weighting of
    ``delta_m``, so this is the total derivative. Central differences are the
    noisy side of the comparison: the solve runs at rtol = atol = 1e-4, which
    bounds how well any finite difference of it can agree.

    Both layouts of the sensitivity system are timed: the five columns stacked
    in each mode's thread, and one column per thread over five replicas of
    each mode (see :func:`matter_power_spectrum_jax`).
    """

    import jax
    import jax.numpy as jnp

    from discoeb.perturbations import (
        PHYSICAL_DENSITY_COLUMNS,
        _as_cosmology,
        density_coefficients,
        matter_power_spectrum_jax,
    )

    cosmology = _perturbed_cosmologies(1)[0]
    densities = jnp.asarray(
        np.asarray(density_coefficients(_as_cosmology(cosmology)), dtype=np.float64)
    )
    assert densities.shape == (len(PHYSICAL_DENSITY_COLUMNS),)

    def total_power(d):
        return jnp.sum(
            matter_power_spectrum_jax(
                MATTER_POWER_K,
                cosmology,
                d,
                one_column_per_trajectory=one_column_per_trajectory,
            )
        )

    value_and_grad = jax.value_and_grad(total_power)

    def run():
        value, grad = value_and_grad(densities)
        return float(value), np.asarray(jax.block_until_ready(grad))

    value, grad = benchmark.pedantic(run, rounds=1, warmup_rounds=1, iterations=1)
    assert np.isfinite(grad).all(), f"non-finite gradient {grad}"

    for i, name in enumerate(DENSITY_NAMES):
        step = 1.0e-5 * abs(float(densities[i]))
        plus = float(total_power(densities.at[i].add(step)))
        minus = float(total_power(densities.at[i].add(-step)))
        finite = (plus - minus) / (2.0 * step)
        rel = abs(grad[i] - finite) / max(abs(finite), 1e-300)
        assert rel < DERIV_GATE, (
            f"d(sum P)/d{name}: sensitivity {grad[i]:.8e} against central "
            f"difference {finite:.8e}, relative {rel:.3g}"
        )


def _device_spline_on_host(stacked, ci, channel, tau):
    """The kernel's thermodynamics spline (``ThermalHistory``) evaluated on the host.

    The device reads the tables as float32 and runs the Horner form in float64;
    this does the same from the host copies, so the comparison below is left
    with float32 rounding of the table values only.
    """

    tau_min, inv_dtau, values, seconds = stacked
    values = values.astype(np.float32).astype(np.float64)
    seconds = seconds.astype(np.float32).astype(np.float64)
    n_grid = values.shape[-1]
    i = (np.log(tau) - tau_min[ci]) * inv_dtau[ci]
    idx = np.clip(i.astype(np.int64), 0, n_grid - 2)
    t = i - idx
    y0, y1 = values[ci, channel, idx], values[ci, channel, idx + 1]
    k0, k1 = seconds[ci, channel, idx], seconds[ci, channel, idx + 1]
    c1 = (y1 - y0) - (2.0 * k0 + k1)
    return y0 + t * (c1 + t * (3.0 * k0 + (k1 - k0) * t))


SOLVE_EB_K = MATTER_POWER_K[::16][::-1]  # descending, to exercise the reordering
SOLVE_EB_TAU = np.geomspace(1.0, 2.0e4, 48)  # the last few lie past every tau0


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

    from discoeb.eb_model import perturbation_model
    from discoeb.perturbations import (
        PARAM_COLUMNS,
        TAU_START,
        density_coefficients,
        make_params,
        matter_power_spectrum,
        solve_eb,
        solve_matter_power_spectrum_batch,
        solve_perturbation_history_batch,
        total_matter_density_contrast,
    )

    cosmologies = _perturbed_cosmologies(2)
    names = ("R", "clxc", "clxb", "theta_2", "n_15", "sigma", "adotoa")
    solution = solve_eb(cosmologies, SOLVE_EB_K, SOLVE_EB_TAU, keep=names)
    n_k, n_tau = len(SOLVE_EB_K), len(SOLVE_EB_TAU)
    assert solution.names == names
    assert solution.values.shape == (2, len(names), n_k, n_tau)
    assert solution["R"].shape == (2, n_k, n_tau)
    assert np.all(solution.tau0 < SOLVE_EB_TAU[-1])

    hist, k_sorted, prepared = solve_perturbation_history_batch(
        SOLVE_EB_K, cosmologies, tau_save=np.concatenate(([TAU_START], SOLVE_EB_TAU))
    )
    hist = np.asarray(hist)[:, :, 1:, :]  # (n_k, n_c, n_tau, nvar), ascending k
    order = np.argsort(SOLVE_EB_K)
    kept = np.asarray(solution.values)[:, :, order, :]  # ascending k too
    layout = prepared.layout
    model = perturbation_model(layout)

    for ci, cosmology in enumerate(prepared.cosmologies):
        for name in ("clxc", "clxb", "theta_2", "n_15"):
            j = names.index(name)
            expected = hist[:, ci, :, model.index[name]]
            np.testing.assert_allclose(kept[ci, j], expected, rtol=1e-9, atol=1e-14)

        # The hook sees the time clamped to the trajectory's own tau0.
        tau = np.minimum(SOLVE_EB_TAU, float(prepared.tau0[ci]))[None, :]
        row = make_params(cosmology, k_sorted, float(prepared.tau0[ci]))
        base = {name: row[:, col][:, None] for name, col in PARAM_COLUMNS.items()}
        base.update(model.state_values(hist[:, ci]))
        base.update(tau=tau, k=k_sorted[:, None])
        for channel, quantity in enumerate(("a", "opacity", "cs2", "rhonu", "pnu")):
            base[quantity] = _device_spline_on_host(prepared.stacked, ci, channel, tau)
        R, sigma, adotoa = model.evaluate("R", "sigma", "adotoa", math=np, **base)
        for name, expected in (("R", R), ("sigma", sigma), ("adotoa", adotoa)):
            j = names.index(name)
            np.testing.assert_allclose(kept[ci, j], expected, rtol=2e-6, atol=1e-12)

    pk_batch = solve_matter_power_spectrum_batch(SOLVE_EB_K, cosmologies)
    for ci, cosmology in enumerate(prepared.cosmologies):
        grhoc, grhob = density_coefficients(cosmology)[2:4]
        clxc = np.asarray(solution["clxc"][ci, :, -1])
        clxb = np.asarray(solution["clxb"][ci, :, -1])
        pk = [
            matter_power_spectrum(
                float(k),
                total_matter_density_contrast(grhoc, dc, grhob, db),
                cosmology.A_s,
                cosmology.n_s,
                cosmology.k_pivot,
            )
            for k, dc, db in zip(SOLVE_EB_K, clxc, clxb)
        ]
        np.testing.assert_allclose(pk, pk_batch[ci], rtol=1e-9)


def test_solve_eb_comoving_curvature_is_conserved_outside_the_horizon():
    """``R`` starts at unity and stays there until the mode enters the horizon.

    The adiabatic initial conditions set ``R = 1``; on super-horizon scales it
    is conserved up to ``O((k tau)^2)`` (0.13 (k tau)^2 in radiation
    domination with no neutrinos), and once inside the horizon it decays with
    the potentials. A one-quantity solve stores ``n_k * n_tau`` numbers.
    """

    from discoeb.perturbations import solve_eb

    k = np.array([1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1])
    tau = np.geomspace(1.0, 2.0e4, 64)
    solution = solve_eb(_perturbed_cosmologies(1)[0], k, tau, keep=("R",))
    assert solution.values.shape == (1, 1, len(k), len(tau))
    R = np.asarray(solution["R"][0])
    assert np.isfinite(R).all()

    assert abs(R[0, 0] - 1.0) < 1.0e-6  # k tau = 1e-4
    super_horizon = tau[None, :] * k[:, None] < 0.1
    assert np.all(np.abs(R[super_horizon] - 1.0) < 5.0e-3), np.abs(
        R[super_horizon] - 1.0
    ).max()
    # k = 0.1 Mpc^-1 entered the horizon long before today and R has decayed.
    assert abs(R[-1, -1]) < 0.5
