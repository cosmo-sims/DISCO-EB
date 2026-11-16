"""Forward sensitivities (``_sensitivities``): derivatives of ``solve_eb`` values against central differences."""

import numpy as np
import pytest
from helpers import (
    DENSITY_PARAMETERS,
    MORE_PARAMETERS,
    SOLVE_EB_K,
    _more_parameter_case,
    _perturbed_cosmologies,
    cosmologies_of,
)

# --- GPU perturbation solves against CLASS: the matter power spectrum and its derivatives.


@pytest.mark.slow
@pytest.mark.gpu
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

    from discoeb.models import LCDM
    from discoeb.models._cosmology import _columns
    from discoeb.solver._background import TAU_START
    from discoeb.solver._kernel import _prepare_solve, parameter_layout
    from discoeb.solver._sensitivities import _directed_inputs, _kept_values
    from discoeb.solver._solver import solve_eb

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
    np.testing.assert_allclose(np.asarray(plain), base, rtol=1e-12)
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
        return solve_eb(cosmologies_of(LCDM, param), SOLVE_EB_K, tau, keep=names)

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
                    )
                )
            )(value)


# Measured |sensitivity - central difference| / max |central difference|
# (2026-11-13, RTX 4070 SUPER), largest over R and clxb: w_DE_0 1.6e-4,
# w_DE_a 3.4e-4, Omegak 2.7e-4, mnu 9.8e-4 (dR/dmnu). Of sum P(k): w_DE_0
# 6.9e-5, w_DE_a 1.3e-6, Omegak 1.5e-5, mnu 7.2e-5. The gate is 1e-3, 2e-3
# for mnu's solve. YHe, cs2_DE and tau_reio missed P(k) by 0.8, 1 and 1.8
# percent at every step tried and stay held.
MORE_SOLVE_GATES = {"mnu": 2.0e-3}


MORE_NAMES = ("R", "clxb", "kappa")


@pytest.mark.slow
@pytest.mark.gpu
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

    from discoeb.solver._solver import solve_eb

    model, cosmology, step = _more_parameter_case(name)
    tau = np.geomspace(10.0, 1.2e4, 12)  # before every shifted tau0

    def values(p, **options):
        batch = cosmologies_of(model, {**cosmology, name: p})
        return solve_eb(batch, SOLVE_EB_K, tau, keep=MORE_NAMES, **options)

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
