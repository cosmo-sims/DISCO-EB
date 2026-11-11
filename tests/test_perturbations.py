"""GPU perturbation solves against CLASS: the matter power spectrum and its derivatives."""

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


def _class_base(param):
    """The CLASS settings of a flat, massless-neutrino ``param``, shared by every CLASS reference."""

    h = param["H0"] / 100.0
    return {
        "h": h,
        "omega_b": param["Omegab"] * h**2,
        "omega_cdm": (param["Omegam"] - param["Omegab"]) * h**2,
        "T_cmb": param["Tcmb"],
        "YHe": param["YHe"],
        "N_ur": param["Neff"],
        "N_ncdm": 0,
        "Omega_k": 0.0,
    }


def _class_linear_pk(param, k_values):
    """Return the CLASS linear matter spectrum in Mpc cubed at redshift zero."""

    Class = pytest.importorskip("classy").Class
    cosmo = Class()
    cosmo.set(
        {
            **_class_base(param),
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
DERIV_GATE = 5.0e-3
# Relative step of the central differences. Step-acceptance flips between the
# two shifted solves move P(k) by ~1e-5 of itself whatever the step (both
# blocks of the split solve choose their steps afresh), which at a 1e-5 step
# swamped the derivative; at 1e-2 the noise is 5e-4 of the derivative and the
# truncation error of the difference ~1e-4.
FD_STEP = 1.0e-2


def test_matter_power_spectrum_derivatives(benchmark):
    """Time and check dP(k)/d(background densities) by forward sensitivity.

    The five densities enter the hierarchy, the recombination rates (through
    the expansion rate), the matter weighting of ``delta_m`` and the conformal
    age, so this is the total derivative, with the sensitivities of the
    recombination block carried through modax's split solve. Central
    differences are the noisy side of the comparison: the solve runs at
    rtol = atol = 1e-4, which bounds how well any finite difference of it can
    agree.
    """

    import jax
    import jax.numpy as jnp

    from discoeb.perturbations import (
        _as_cosmology,
        density_coefficients,
        matter_power_spectrum_jax,
    )

    cosmology = _perturbed_cosmologies(1)[0]
    densities = jnp.asarray(
        np.asarray(density_coefficients(_as_cosmology(cosmology)), dtype=np.float64)
    )
    assert densities.shape == (len(DENSITY_NAMES),)

    def total_power(d):
        return jnp.sum(matter_power_spectrum_jax(MATTER_POWER_K, cosmology, d))

    value_and_grad = jax.value_and_grad(total_power)

    def run():
        value, grad = value_and_grad(densities)
        return float(value), np.asarray(jax.block_until_ready(grad))

    value, grad = benchmark.pedantic(run, rounds=1, warmup_rounds=1, iterations=1)
    assert np.isfinite(grad).all(), f"non-finite gradient {grad}"

    for i, name in enumerate(DENSITY_NAMES):
        step = FD_STEP * abs(float(densities[i]))
        plus = float(total_power(densities.at[i].add(step)))
        minus = float(total_power(densities.at[i].add(-step)))
        finite = (plus - minus) / (2.0 * step)
        rel = abs(grad[i] - finite) / max(abs(finite), 1e-300)
        assert rel < DERIV_GATE, (
            f"d(sum P)/d{name}: sensitivity {grad[i]:.8e} against central "
            f"difference {finite:.8e}, relative {rel:.3g}"
        )


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

    from discoeb.perturbations import (
        PARAM_COLUMNS,
        TAU_START,
        THERMO_ERROR_WEIGHT,
        density_coefficients,
        make_params,
        solve_eb,
        solve_matter_power_spectrum_batch,
        solve_perturbation_history_batch,
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
        row = make_params(cosmology, k_sorted, float(prepared.tau0[ci]))
        base = {name: row[:, col][:, None] for name, col in PARAM_COLUMNS.items()}
        base.update(model.state_values(hist[:, ci]))
        base.update(tau=tau, k=k_sorted[:, None])
        R, sigma, adotoa = model.evaluate("R", "sigma", "adotoa", math=np, **base)
        for name, expected in (("R", R), ("sigma", sigma), ("adotoa", adotoa)):
            j = names.index(name)
            np.testing.assert_allclose(kept[ci, j], expected, rtol=2e-6, atol=1e-12)

    # the same thermal error weight as solve_eb's launch, so the two solves agree bit for bit
    pk_batch = solve_matter_power_spectrum_batch(
        SOLVE_EB_K, cosmologies, thermo_weight=THERMO_ERROR_WEIGHT
    )
    for ci, cosmology in enumerate(prepared.cosmologies):
        grhoc, grhob = density_coefficients(cosmology)[2:4]
        clxc = np.asarray(solution["clxc"][ci, :, -1])
        clxb = np.asarray(solution["clxb"][ci, :, -1])
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


def test_solve_eb_derivatives():
    """The kept quantities are differentiable in the cosmological parameters.

    The derivative chains the densities' dependence on ``Omegam``, ``Omegab``,
    ``H0``, ``Tcmb`` and ``Neff`` into the densities-level rule, whose tangent
    hook stores each quantity's derivative along the sensitivity column of its
    replica: ``jax.jvp``, ``jax.jacfwd`` and ``jax.grad`` of the kept values
    all come from one joint launch. The densities-level rule is checked
    against central differences of its own primal at fixed thermal history;
    the differences are the noisy side: the solve runs at rtol = atol = 1e-4,
    a step-acceptance flip between the two shifted solves moves a value by
    ~1e-6 of its scale (5e-5 for the photon quadrupole, whose early-time
    values are tiny), so the Jacobian is checked as the change it predicts
    over a 0.1 percent step against the measured change, in units of each
    quantity's scale, and the gradient of a scale-weighted sum over a 1
    percent step. The parameter-level derivative is then the chain rule
    through ``density_parameters``.
    """

    import jax
    import jax.numpy as jnp

    from discoeb.perturbations import (
        DIFFERENTIABLE_PARAMETERS,
        DIRECTED_COLUMNS,
        DIRECTED_NAMES,
        TAU_START,
        _as_cosmology,
        _input_vector,
        _kept_values,
        _prepare_solve,
        solve_eb,
    )

    cosmology = _perturbed_cosmologies(1)[0]
    # the inputs exactly as solve_eb forms them: a last-bit difference moves a
    # step-acceptance and with it a value by ~1e-6
    theta = jnp.asarray([cosmology[key] for key in DIFFERENTIABLE_PARAMETERS])
    fixed = {k: v for k, v in cosmology.items() if k not in DIFFERENTIABLE_PARAMETERS}
    densities = _input_vector(fixed, theta)[None, :]
    tau = np.geomspace(10.0, 2.0e4, 12)
    names = ("R", "clxc", "theta_2")
    noise = {"R": 5.0e-6, "clxc": 5.0e-6, "theta_2": 1.0e-4}

    plain = solve_eb(cosmology, SOLVE_EB_K, tau, keep=names)
    prepared = _prepare_solve((_as_cosmology(cosmology),))
    kept = _kept_values(
        prepared, SOLVE_EB_K, np.concatenate(([TAU_START], tau)), names, ()
    )
    base = np.asarray(kept(densities))
    # The same launch, from densities formed in JAX rather than as floats.
    np.testing.assert_allclose(np.asarray(plain.values), base, rtol=1e-12)
    scale = np.abs(base).max(axis=(2, 3), keepdims=True)

    # The joint launch's primal: the fp64 LU in place of the plain fp32 one,
    # so the same trajectories to within the solver's tolerance.
    primal, _ = jax.jvp(kept, (densities,), (jnp.zeros_like(densities),))
    assert np.abs(np.asarray(primal) - base).max() < 1.0e-4 * scale.max()

    # The central differences are converged (rtol = atol = 1e-6): at the default
    # tolerance the solve's own error, different on the two sides of a step,
    # is the larger part of a difference.
    kept_converged = _kept_values(
        prepared,
        SOLVE_EB_K,
        np.concatenate(([TAU_START], tau)),
        names,
        (),
        rtol=1.0e-6,
        atol=1.0e-6,
        max_steps=200_000,
    )

    jac = np.asarray(jax.jacfwd(kept)(densities))[..., 0, :]
    assert jac.shape == base.shape + (len(DIRECTED_COLUMNS),)
    assert np.isfinite(jac).all()
    for j, dn in enumerate(DIRECTED_NAMES):
        step = 1.0e-3 * float(densities[0, j])
        unit = jnp.zeros_like(densities).at[0, j].set(1.0)
        finite = (
            np.asarray(kept_converged(densities + step * unit))
            - np.asarray(kept_converged(densities - step * unit))
        ) / (2.0 * step)
        err = np.abs(jac[..., j] - finite) * step / scale
        for q, name in enumerate(names):
            assert err[0, q].max() < noise[name], (
                f"d{name}/d{dn}: {err[0, q].max():.3g} of the scale over a 1e-3 step"
            )

    weights = 1.0 / scale

    def total(d):
        return jnp.sum(kept(d) * weights)

    grad = np.asarray(jax.grad(total)(densities))[0]
    # The transposed rule against the untransposed one: jax.grad of a weighted
    # sum is the Jacobian contracted with the weights (the Jacobian's accuracy
    # is the check above; a central difference of the sum is ill-conditioned,
    # its terms cancelling to 1e-3 of their size for Tcmb).
    summed = np.einsum("cqkt,cqktj->j", np.broadcast_to(weights, base.shape), jac)
    np.testing.assert_allclose(
        grad, summed, rtol=1.0e-6, err_msg="grad vs contracted jacfwd"
    )

    # Through the dictionary: the chain rule with the densities' Jacobian,
    # under jacfwd and grad alike, and a plain number where a derivative is
    # not carried.
    def of_parameters(theta):
        param = {**fixed, **dict(zip(DIFFERENTIABLE_PARAMETERS, theta))}
        return solve_eb(param, SOLVE_EB_K, tau, keep=names).values

    jac_theta = np.asarray(jax.jacfwd(of_parameters)(theta))
    d_densities = np.asarray(jax.jacfwd(lambda t: _input_vector(fixed, t))(theta))
    # Omegab's column is the near-cancelling difference of the CDM and
    # baryon columns, so the contraction itself carries ~1e-9 of roundoff.
    np.testing.assert_allclose(
        jac_theta,
        np.einsum("cqktj,jp->cqktp", jac, d_densities),
        rtol=1e-6,
        atol=1e-9 * np.abs(jac_theta).max(),
    )
    grad_theta = np.asarray(
        jax.grad(lambda t: jnp.sum(of_parameters(t) * weights))(theta)
    )
    np.testing.assert_allclose(
        grad_theta, np.einsum("cqkt,cqktp->p", weights, jac_theta), rtol=1e-10
    )
    assert np.all(grad_theta != 0.0)

    with pytest.raises(ValueError, match="cannot be differentiated"):
        jax.grad(
            lambda m: jnp.sum(
                solve_eb({**cosmology, "mnu": m}, SOLVE_EB_K, tau, keep=names).values
            )
        )(0.0)
