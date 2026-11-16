"""The matter power spectrum (``power_spectrum``) against CLASS, its derivatives, under ``jit``."""

import numpy as np
import pytest
from helpers import (
    BATCH_SIZE_CASES,
    DENSITY_PARAMETERS,
    MATTER_POWER_K,
    MORE_PARAMETERS,
    SOLVE_EB_K,
    _class_base,
    _matter_power,
    _more_parameter_case,
    _perturbed_cosmologies,
    cosmologies_of,
)

# --- GPU perturbation solves against CLASS: the matter power spectrum and its derivatives.


PK_GATE = 5.0e-3


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


@pytest.mark.gpu
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


@pytest.mark.gpu
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
    from discoeb.solver._solver import conformal_ages

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


@pytest.mark.gpu
def test_matter_power_spectrum_under_jit():
    """``solve_eb`` and ``matter_power_spectrum`` trace under ``jit``, every key traced.

    One trace serves every cosmology of the same keys, the values agree with
    the eager solve, the gradient under ``jit`` with the eager one, and a key
    outside :data:`DIFFERENTIABLE_PARAMETERS` still refuses a derivative.
    """

    import jax
    import jax.numpy as jnp

    from discoeb.models import LCDM
    from discoeb.observables.power_spectrum import (
        matter_power_spectrum,
    )
    from discoeb.solver._solver import conformal_ages, solve_eb

    DIFFERENTIABLE_PARAMETERS = DENSITY_PARAMETERS
    traces = []

    def power(cosmology):
        traces.append(cosmology)
        batch = cosmologies_of(LCDM, [cosmology])
        tau0 = conformal_ages(batch)[0]
        keep = ("clxc", "clxb", "z", "vb")
        values = solve_eb(
            batch, SOLVE_EB_K, tau0.reshape(1), keep=keep, thermo_weight=0.1
        )
        today = dict(zip(keep, jnp.unstack(values[0, :, :, -1])))
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


MORE_PK_GATES: dict = {}


@pytest.mark.slow
@pytest.mark.gpu
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

    from discoeb.solver._solver import conformal_ages

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
