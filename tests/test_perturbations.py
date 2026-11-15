"""GPU perturbation solves against CLASS: the matter power spectrum and its derivatives."""

import numpy as np
import pytest
from conftest import cosmologies_of

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


def _matter_power(k_values, batch, tau_today=None, **options):
    """``P(k)`` of a cosmology or a list of them as a user computes it: ``solve_eb``, then the spectrum.

    ``tau_today`` must lie at or past every conformal age. It defaults to the
    batch's oldest, which needs the parameters as numbers.
    """

    import jax.numpy as jnp

    from discoeb.models import Model
    from discoeb.power_spectrum import (
        matter_power_spectrum,
    )
    from discoeb.solver import conformal_ages, solve_eb

    cosmologies = [batch] if isinstance(batch, Model) else list(batch)
    tau0 = conformal_ages(batch)
    if tau_today is None:
        tau_today = tau0.max()
    options.setdefault("thermo_weight", 0.1)
    solution = solve_eb(
        batch, k_values, [tau_today], keep=("clxc", "clxb", "z", "vb"), **options
    )
    today = {name: solution[name][..., -1] for name in solution.names}
    params = {
        key: jnp.stack([jnp.asarray(getattr(c, key)) for c in cosmologies])
        for key in ("Omegam", "Omegab", "A_s", "n_s", "k_p")
    }
    return matter_power_spectrum(k_values, **today, tau0=tau0, **params)


@pytest.mark.parametrize("n_cosmologies", BATCH_SIZE_CASES)
def test_matter_power_spectrum_matches_class(n_cosmologies, benchmark):
    """Check and time one GPU launch over nearby flat LCDM cosmologies."""

    import jax

    from discoeb.models import LCDM

    cosmologies = _perturbed_cosmologies(n_cosmologies)

    def run(batch):
        return np.asarray(jax.block_until_ready(_matter_power(MATTER_POWER_K, batch)))

    pk_ours = benchmark.pedantic(
        run,
        args=(cosmologies_of(LCDM, cosmologies),),
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


DERIV_GATE = 5.0e-3
# Relative step of the central differences. Step-acceptance flips between the
# two shifted solves move P(k) by ~1e-5 of itself whatever the step (both
# blocks of the split solve choose their steps afresh), which at a 1e-5 step
# swamped the derivative; at 1e-2 the noise is 5e-4 of the derivative and the
# truncation error of the difference ~1e-4.
FD_STEP = 1.0e-2
# Tcmb also reaches P(k) through the recombination rates, and the thermal
# inputs' sensitivities are biased against central differences of the joint
# solve (YHe 2 percent, Tcmb 1.9, Omegab 0.5, with fp32 or fp64 rate math),
# while the densities' agree to 1e-4. Through P(k) that leaves dP/dTcmb 3.7
# percent off. A known gap of the recombination block's sensitivities, gated
# loosely here so it cannot grow unnoticed.
DERIV_GATES = {"Tcmb": 5.0e-2}
# The parameters of the densities, which the derivative tests below have
# always covered. The others have a test each (test_more_parameters_*).
DENSITY_PARAMETERS = ("Omegam", "Omegab", "H0", "Tcmb", "Neff")


def test_matter_power_spectrum_derivatives(benchmark):
    """Time and check dP(k)/d(differentiable parameters) by forward sensitivity.

    The parameters enter the hierarchy, the recombination rates (through the
    expansion rate), the matter weighting of ``delta_m`` and the conformal
    age, so this is the total derivative, with the sensitivities of the
    recombination block carried through modax's split solve. Central
    differences are the noisy side of the comparison: the solve runs at
    rtol = atol = 1e-4, which bounds how well any finite difference of it can
    agree.
    """

    import jax
    import jax.numpy as jnp

    from discoeb.models import LCDM
    from discoeb.solver import conformal_ages

    DIFFERENTIABLE_PARAMETERS = DENSITY_PARAMETERS
    cosmology = _perturbed_cosmologies(1)[0]
    theta = jnp.asarray([cosmology[name] for name in DIFFERENTIABLE_PARAMETERS])
    # past the age of every shifted cosmology; the parameters are traced
    tau_today = 2.0 * conformal_ages(cosmology).max()

    def total_power(t):
        params = {**cosmology, **dict(zip(DIFFERENTIABLE_PARAMETERS, t))}
        return jnp.sum(
            _matter_power(MATTER_POWER_K, cosmologies_of(LCDM, params), tau_today)
        )

    value_and_grad = jax.value_and_grad(total_power)

    def run():
        value, grad = value_and_grad(theta)
        return float(value), np.asarray(jax.block_until_ready(grad))

    value, grad = benchmark.pedantic(run, rounds=1, warmup_rounds=1, iterations=1)
    assert np.isfinite(grad).all(), f"non-finite gradient {grad}"

    for i, name in enumerate(DIFFERENTIABLE_PARAMETERS):
        step = FD_STEP * abs(float(theta[i]))
        plus = float(total_power(theta.at[i].add(step)))
        minus = float(total_power(theta.at[i].add(-step)))
        finite = (plus - minus) / (2.0 * step)
        rel = abs(grad[i] - finite) / max(abs(finite), 1e-300)
        assert rel < DERIV_GATES.get(name, DERIV_GATE), (
            f"d(sum P)/d{name}: sensitivity {grad[i]:.8e} against central "
            f"difference {finite:.8e}, relative {rel:.3g}"
        )


def test_matter_power_spectrum_under_jit():
    """``solve_eb`` and ``matter_power_spectrum`` trace under ``jit``, every key traced.

    One trace serves every cosmology of the same keys, the values agree with
    the eager solve, the gradient under ``jit`` with the eager one, and a key
    outside :data:`DIFFERENTIABLE_PARAMETERS` still refuses a derivative.
    """

    import jax
    import jax.numpy as jnp

    from discoeb.models import LCDM
    from discoeb.power_spectrum import (
        matter_power_spectrum,
    )
    from discoeb.solver import conformal_ages, solve_eb

    DIFFERENTIABLE_PARAMETERS = DENSITY_PARAMETERS
    traces = []

    def power(cosmology):
        traces.append(cosmology)
        batch = cosmologies_of(LCDM, [cosmology])
        tau0 = conformal_ages(batch)[0]
        solution = solve_eb(
            batch,
            SOLVE_EB_K,
            tau0[None],
            keep=("clxc", "clxb", "z", "vb"),
            thermo_weight=0.1,
        )
        today = {name: solution[name][0, :, -1] for name in solution.names}
        params = {
            key: cosmology[key] for key in ("Omegam", "Omegab", "A_s", "n_s", "k_p")
        }
        return matter_power_spectrum(SOLVE_EB_K, **today, tau0=tau0, **params)

    jitted = jax.jit(power)
    for cosmology in _perturbed_cosmologies(2):
        np.testing.assert_allclose(
            np.asarray(jitted(cosmology)), np.asarray(power(cosmology)), rtol=1e-12
        )
    assert sum(isinstance(c["H0"], jax.core.Tracer) for c in traces) == 1

    cosmology = _perturbed_cosmologies(1)[0]
    theta = jnp.asarray([cosmology[n] for n in DIFFERENTIABLE_PARAMETERS])

    def total(t, of):
        return jnp.sum(of({**cosmology, **dict(zip(DIFFERENTIABLE_PARAMETERS, t))}))

    eager = np.asarray(jax.grad(total)(theta, power))
    under_jit = np.asarray(jax.jit(jax.grad(lambda t: total(t, jitted)))(theta))
    np.testing.assert_allclose(under_jit, eager, rtol=1e-12)

    with pytest.raises(ValueError, match="cannot be differentiated"):
        jax.jit(jax.grad(lambda w: jnp.sum(jitted({**cosmology, "w_DE_0": w}))))(-1.0)


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

    from discoeb.background import TAU_START, density_coefficients
    from discoeb.cosmology import as_cosmology
    from discoeb.kernel import PARAM_COLUMNS, make_params
    from discoeb.models import LCDM
    from discoeb.recombination import THERMO_ERROR_WEIGHT
    from discoeb.solver import solve_eb, solve_perturbation_history_batch

    cosmologies = _perturbed_cosmologies(2)
    names = ("R", "clxc", "clxb", "theta_2", "n_15", "sigma", "adotoa")
    solution = solve_eb(
        cosmologies_of(LCDM, cosmologies), SOLVE_EB_K, SOLVE_EB_TAU, keep=names
    )
    n_k, n_tau = len(SOLVE_EB_K), len(SOLVE_EB_TAU)
    assert solution.names == names
    assert solution.values.shape == (2, len(names), n_k, n_tau)
    assert solution["R"].shape == (2, n_k, n_tau)
    assert np.all(solution.tau0 < SOLVE_EB_TAU[-1])

    hist, k_sorted, prepared = solve_perturbation_history_batch(
        SOLVE_EB_K,
        cosmologies_of(LCDM, cosmologies),
        tau_save=np.concatenate(([TAU_START], SOLVE_EB_TAU)),
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
    pk_batch = _matter_power(
        SOLVE_EB_K, cosmologies_of(LCDM, cosmologies), thermo_weight=THERMO_ERROR_WEIGHT
    )
    for ci, cosmology in enumerate(map(as_cosmology, prepared.cosmologies)):
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

    from discoeb.models import LCDM
    from discoeb.solver import solve_eb

    k = np.array([1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1])
    tau = np.geomspace(1.0, 2.0e4, 64)
    solution = solve_eb(
        cosmologies_of(LCDM, _perturbed_cosmologies(1)[0]), k, tau, keep=("R",)
    )
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

    from discoeb.background import TAU_START
    from discoeb.cosmology import _columns
    from discoeb.kernel import _prepare_solve, parameter_layout
    from discoeb.models import LCDM
    from discoeb.sensitivities import _directed_inputs, _kept_values
    from discoeb.solver import solve_eb

    DIFFERENTIABLE_PARAMETERS = DENSITY_PARAMETERS
    cosmology = _perturbed_cosmologies(1)[0]
    # the inputs exactly as solve_eb forms them: a last-bit difference moves a
    # step-acceptance and with it a value by ~1e-6
    theta = jnp.asarray([cosmology[key] for key in DIFFERENTIABLE_PARAMETERS])
    fixed = {k: v for k, v in cosmology.items() if k not in DIFFERENTIABLE_PARAMETERS}

    def inputs_of(theta):
        param = {**fixed, **dict(zip(DIFFERENTIABLE_PARAMETERS, theta))}
        return _directed_inputs(_columns([param]), LCDM)

    densities = inputs_of(theta)
    tau = np.geomspace(10.0, 2.0e4, 12)
    names = ("R", "clxc", "theta_2")
    noise = {"R": 5.0e-6, "clxc": 5.0e-6, "theta_2": 1.0e-4}

    plain = solve_eb(cosmologies_of(LCDM, cosmology), SOLVE_EB_K, tau, keep=names)
    prepared = _prepare_solve(LCDM, (cosmology,), SOLVE_EB_K)
    kept = _kept_values(prepared, np.concatenate(([TAU_START], tau)), names, ())
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
        np.concatenate(([TAU_START], tau)),
        names,
        (),
        rtol=1.0e-6,
        atol=1.0e-6,
        max_steps=200_000,
    )

    jac = np.asarray(jax.jacfwd(kept)(densities))[..., 0, :]
    DIRECTED_NAMES = parameter_layout(LCDM).directed
    assert jac.shape == base.shape + (len(DIRECTED_NAMES),)
    assert np.isfinite(jac).all()
    for j, dn in enumerate(DIRECTED_NAMES):
        if densities[0, j] == 0.0:  # grhok, amnu and z_re of a flat LCDM
            continue
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
        return solve_eb(cosmologies_of(LCDM, param), SOLVE_EB_K, tau, keep=names).values

    jac_theta = np.asarray(jax.jacfwd(of_parameters)(theta))
    d_densities = np.asarray(jax.jacfwd(lambda t: inputs_of(t)[0])(theta))
    # The two are separate launches, one replica per parameter along
    # d inputs / d theta against one per input, so they differ by the
    # solve's noise: checked like the central differences above, as the
    # change each predicts over a 0.1 percent step in units of the
    # quantity's scale.
    chained = np.einsum("cqktj,jp->cqktp", jac, d_densities)
    for p, key in enumerate(DIFFERENTIABLE_PARAMETERS):
        step = 1.0e-3 * float(theta[p])
        err = np.abs(jac_theta[..., p] - chained[..., p]) * step / scale
        for q, name in enumerate(names):
            assert err[0, q].max() < noise[name], (
                f"d{name}/d{key}: {err[0, q].max():.3g} against the chain rule"
            )
    grad_theta = np.asarray(
        jax.grad(lambda t: jnp.sum(of_parameters(t) * weights))(theta)
    )
    np.testing.assert_allclose(
        grad_theta, np.einsum("cqkt,cqktp->p", weights, jac_theta), rtol=1e-10
    )
    assert np.all(grad_theta != 0.0)

    # mnu needs MassiveNeutrinos, cs2_DE DarkEnergy. The sensitivities in
    # YHe, cs2_DE and tau_reio miss central differences of P(k) by 0.8, 1
    # and 1.8 percent (2026-11-13), so they stay held.
    for key, value in (("mnu", 0.0), ("YHe", 0.245), ("tau_reio", 0.054)):
        with pytest.raises(ValueError, match="cannot be differentiated"):
            jax.grad(
                lambda v, key=key: jnp.sum(
                    solve_eb(
                        cosmologies_of(LCDM, {**cosmology, key: v}),
                        SOLVE_EB_K,
                        tau,
                        keep=names,
                    ).values
                )
            )(value)


# The parameters beyond the densities': (model, fiducial changes, P(k) step).
# Both comparisons difference converged solves (rtol = atol = 1e-6): the
# solve is not smooth in the parameters below ~1e-6 of a value, so the
# differences' noise falls as 1 / step (measured for YHe, 2026-11-13).
MORE_PARAMETERS = {
    "w_DE_0": ("W0WACDM", {"w_DE_0": -0.9, "w_DE_a": 0.1}, 9.0e-3),
    "w_DE_a": ("W0WACDM", {"w_DE_0": -0.9, "w_DE_a": 0.1}, 4.0e-3),
    "Omegak": ("LCDM", {"Omegak": 0.02}, 2.0e-3),
    "mnu": ("LCDM_MNU", {"Nmnu": 1.0, "mnu": 0.1, "Neff": 2.0328}, 4.0e-3),
}
# Measured |sensitivity - central difference| / max |central difference|
# (2026-11-13, RTX 4070 SUPER), largest over R and clxb: w_DE_0 1.6e-4,
# w_DE_a 3.4e-4, Omegak 2.7e-4, mnu 9.8e-4 (dR/dmnu). Of sum P(k): w_DE_0
# 6.9e-5, w_DE_a 1.3e-6, Omegak 1.5e-5, mnu 7.2e-5. The gate is 1e-3, 2e-3
# for mnu's solve. YHe, cs2_DE and tau_reio missed P(k) by 0.8, 1 and 1.8
# percent at every step tried and stay held.
MORE_SOLVE_GATES = {"mnu": 2.0e-3}
MORE_PK_GATES: dict = {}
MORE_NAMES = ("R", "clxb", "kappa")


def _more_parameter_case(name):
    import discoeb.models as models

    model_name, changes, step = MORE_PARAMETERS[name]
    cosmology = {**REFERENCE_COSMOLOGY, **changes}
    return getattr(models, model_name), cosmology, step


@pytest.mark.parametrize("name", list(MORE_PARAMETERS))
def test_more_parameters_solve_eb(name):
    """``solve_eb``'s ``jax.jvp`` in each parameter beyond the densities' against central differences.

    The differences come from converged solves (rtol = atol = 1e-6), the
    sensitivity from one at the default tolerance, as in
    :func:`test_solve_eb_derivatives`. ``kappa``, the optical depth, carries
    the thermal parameters and the reionization redshift directly.
    """

    import jax
    import jax.numpy as jnp

    from discoeb.solver import solve_eb

    model, cosmology, step = _more_parameter_case(name)
    tau = np.geomspace(10.0, 1.2e4, 12)  # before every shifted tau0

    def values(p, **options):
        batch = cosmologies_of(model, {**cosmology, name: p})
        return solve_eb(batch, SOLVE_EB_K, tau, keep=MORE_NAMES, **options).values

    p0 = jnp.asarray(float(cosmology[name]))
    _, tangent = jax.jvp(values, (p0,), (jnp.ones_like(p0),))
    tangent = np.asarray(tangent)
    converged = dict(rtol=1.0e-6, atol=1.0e-6, max_steps=200_000)
    finite = (
        np.asarray(values(p0 + step, **converged))
        - np.asarray(values(p0 - step, **converged))
    ) / (2.0 * step)
    assert np.isfinite(tangent).all()
    value = np.abs(np.asarray(values(p0)))
    for q, quantity in enumerate(MORE_NAMES):
        scale = np.abs(value[0, q]).max()
        err = np.abs(tangent[0, q] - finite[0, q]).max()
        signal = np.abs(finite[0, q]).max()
        print(
            f"d{quantity}/d{name}: {err / signal:.3g} of the largest difference, "
            f"which moves the quantity by {signal * step / scale:.3g} of its scale"
        )
        if signal * step < 1.0e-4 * scale:
            # A change the solve's noise swamps: the sensitivity must predict
            # one as small.
            assert err * step < 1.0e-5 * scale, f"d{quantity}/d{name}"
            continue
        assert err < MORE_SOLVE_GATES.get(name, 1.0e-3) * signal, (
            f"d{quantity}/d{name}: {err / signal:.3g} of its largest central difference"
        )


@pytest.mark.parametrize("name", list(MORE_PARAMETERS))
def test_more_parameters_matter_power(name):
    """``jax.grad`` of ``sum P(k)`` in each parameter beyond the densities' against central differences.

    The total derivative: ``compute_matter_power_spectrum`` adds the
    end-time term of the conformal age, which ``w_DE_*``, ``Omegak`` and
    ``mnu`` move. Central differences at the default tolerance, as in
    :func:`test_matter_power_spectrum_derivatives`.
    """

    import jax
    import jax.numpy as jnp

    from discoeb.solver import conformal_ages

    model, cosmology, step = _more_parameter_case(name)
    tau_today = 2.0 * conformal_ages(cosmology).max()

    def total_power(p, **options):
        batch = cosmologies_of(model, {**cosmology, name: p})
        return jnp.sum(_matter_power(MATTER_POWER_K, batch, tau_today, **options))

    p0 = jnp.asarray(float(cosmology[name]))
    grad = float(jax.grad(total_power)(p0))
    # Converged differences: some of these parameters move P(k) by 1e-5 of
    # itself over the step, the noise of a default-tolerance solve.
    converged = dict(rtol=1.0e-6, atol=1.0e-6, max_steps=200_000)
    finite = (
        float(total_power(p0 + step, **converged))
        - float(total_power(p0 - step, **converged))
    ) / (2.0 * step)
    rel = abs(grad - finite) / abs(finite)
    print(f"d(sum P)/d{name}: {grad:.8e} against {finite:.8e}, relative {rel:.3g}")
    assert rel < MORE_PK_GATES.get(name, 1.0e-3), (
        f"d(sum P)/d{name}: sensitivity {grad:.8e} against central "
        f"difference {finite:.8e}, relative {rel:.3g}"
    )
