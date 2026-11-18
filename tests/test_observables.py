"""Tests of ``discoeb.observables``."""

import numpy as np
import pytest
from helpers import (
    BATCH_SIZE_CASES,
    DENSITY_PARAMETERS,
    DIAG_A,
    MATTER_POWER_K,
    MORE_PARAMETER_CASES,
    PK_KEEP,
    SOLVE_EB_K,
    _class_base,
    _matter_power,
    _more_parameter_case,
    _perturbed_cosmologies,
    cap_columns,
    class_subset,
    cosmologies_of,
    physical,
)

# ============================================================================
# power_spectrum: The matter power spectrum (``power_spectrum``) against CLASS, its derivatives, under ``jit``
# ============================================================================


def test_matter_power_spectrum_takes_one_cosmology_and_vmaps_over_a_batch():
    """One cosmology per call; ``jax.vmap`` over a batch gives each one's ``P(k)``."""

    import jax
    import jax.numpy as jnp

    from discoeb import matter_power_spectrum

    k = jnp.geomspace(1e-3, 0.1, 5)
    rng = np.random.default_rng(0)
    per_mode = {n: jnp.asarray(rng.normal(size=(3, 5))) for n in ("delta_c", "delta_b")}
    scalars = {
        "omega_c": jnp.asarray([0.11, 0.12, 0.13]),
        "omega_b": jnp.full(3, 0.022),
        "A_s": jnp.full(3, 2.1e-9),
        "n_s": jnp.full(3, 0.966),
        "k_p": jnp.full(3, 0.05),
    }
    batched = jax.vmap(matter_power_spectrum, in_axes=(None,))(k, **per_mode, **scalars)
    assert batched.shape == (3, 5)
    for i in range(3):
        one = matter_power_spectrum(
            k,
            **{n: v[i] for n, v in per_mode.items()},
            **{n: v[i] for n, v in scalars.items()},
        )
        np.testing.assert_allclose(batched[i], one, rtol=1e-14)


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
    """Check and time one GPU launch over nearby flat LCDM cosmologies.

    The launch is timed over the batch. CLASS, a second or so a cosmology on
    the host, is the reference for eight of them, the two ends among them.
    """

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
    assert np.isfinite(pk_ours).all() and np.all(pk_ours > 0.0)
    compared = class_subset(cosmologies)
    pk_class = np.stack(
        [_class_linear_pk(cosmologies[i], MATTER_POWER_K) for i in compared]
    )
    rel = np.abs(pk_ours[compared] / pk_class - 1.0)
    worst = np.unravel_index(np.argmax(rel), rel.shape)
    assert float(rel[worst]) < PK_GATE, (
        f"max relative error {rel[worst]:.6g} for cosmology {compared[worst[0]]} "
        f"at k={MATTER_POWER_K[worst[1]]:.6g} Mpc^-1"
    )


DERIV_GATE = 5.0e-3


# Relative step of the central differences. Step-acceptance flips between the
# two shifted solves move P(k) by ~1e-5 of itself whatever the step (both
# blocks of the split solve choose their steps afresh), which at a 1e-5 step
# swamped the derivative; at 1e-2 the noise is 5e-4 of the derivative and the
# truncation error of the difference ~1e-4.
FD_STEP = 1.0e-2


@pytest.mark.gpu
@pytest.mark.parametrize(
    "columns",
    [
        pytest.param(2, id="two-columns"),
        pytest.param(None, id="five-columns", marks=pytest.mark.slow),
    ],
)
def test_matter_power_spectrum_derivatives(columns, benchmark, monkeypatch):
    """Time and check dP(k)/d(differentiable parameters) by forward sensitivity.

    In launches of two columns, the last one padded, on the kernel the fast
    suite's derivative tests share; and, slow, in one launch of the default
    cap's five-column kernel, which is what a user's gradient in five
    parameters runs on.

    The parameters enter the hierarchy, the recombination rates (through the
    expansion rate) and the matter weighting of ``delta_m``. The solve
    integrates in the scale factor, so this is the derivative of ``P(k)``
    today with nothing to add, with the sensitivities of the
    recombination block carried through modax's split solve. Central
    differences are the noisy side of the comparison: the solve runs at
    rtol = atol = 1e-4, which bounds how well any finite difference of it can
    agree.
    """

    import jax
    import jax.numpy as jnp

    from discoeb.models import LCDM

    if columns is not None:
        cap_columns(monkeypatch, columns)
    DIFFERENTIABLE_PARAMETERS = DENSITY_PARAMETERS
    cosmology = _perturbed_cosmologies(1)[0]
    theta = jnp.asarray([cosmology[name] for name in DIFFERENTIABLE_PARAMETERS])

    def total_power(t):
        params = {**cosmology, **dict(zip(DIFFERENTIABLE_PARAMETERS, t))}
        return jnp.sum(_matter_power(MATTER_POWER_K, cosmologies_of(LCDM, params)))

    # forward mode: the value comes out of the same launch as has_aux
    value_and_grad = jax.jacfwd(lambda t: (total_power(t),) * 2, has_aux=True)

    def run():
        grad, value = value_and_grad(theta)
        return float(value), np.asarray(jax.block_until_ready(grad))

    value, grad = benchmark.pedantic(run, rounds=1, warmup_rounds=1, iterations=1)
    assert np.isfinite(grad).all(), f"non-finite gradient {grad}"

    for i, name in enumerate(DIFFERENTIABLE_PARAMETERS):
        step = FD_STEP * abs(float(theta[i]))
        plus = float(total_power(theta.at[i].add(step)))
        minus = float(total_power(theta.at[i].add(-step)))
        finite = (plus - minus) / (2.0 * step)
        rel = abs(grad[i] - finite) / max(abs(finite), 1e-300)
        assert rel < DERIV_GATE, (
            f"d(sum P)/d{name}: sensitivity {grad[i]:.8e} against central "
            f"difference {finite:.8e}, relative {rel:.3g}"
        )


@pytest.mark.gpu
def test_split_derivatives_match_one_thread(two_columns):
    """A thread per parameter gives the Jacobian the trajectory's own thread gives.

    Two distinct cosmologies and a Jacobian in each one's parameters, so a
    copy in the wrong row would show. At the default tolerance the two
    differ by 1e-3 of a column's scale (3e-3 for ``T_cmb``), which is the
    solve's own error, so the comparison runs at 1e-6, where they agree to
    1e-6. ``T_cmb``, whose sensitivity crosses the hard switch of the matter
    temperature's rate, agrees to 6e-6 or to 1e-4 depending on the kernel's
    last bit, and has a gate of its own. The trajectory's own thread
    carries two columns at a time here (``two_columns``), so the ten
    variables take five launches.
    """

    import jax
    import jax.numpy as jnp

    from discoeb.models import LCDM

    cosmologies = _perturbed_cosmologies(2)
    theta = jnp.asarray([[c[name] for name in DENSITY_PARAMETERS] for c in cosmologies])

    def power(t, split):
        batch = [
            {**c, **dict(zip(DENSITY_PARAMETERS, row))}
            for c, row in zip(cosmologies, t)
        ]
        return _matter_power(
            MATTER_POWER_K,
            cosmologies_of(LCDM, batch),
            split_derivatives=split,
            rtol=1e-6,
            atol=1e-6,
        )

    together = np.asarray(jax.jacfwd(lambda t: power(t, False))(theta))
    split = np.asarray(jax.jacfwd(lambda t: power(t, True))(theta))
    assert np.abs(together).max() > 0.0
    for c in range(2):  # a cosmology's P(k) reads its own parameters only
        assert np.all(split[c, :, 1 - c] == 0.0)
    scale = np.abs(together).max(axis=1, keepdims=True)
    worst = (np.abs(split - together) / np.where(scale > 0.0, scale, 1.0)).max(axis=1)
    gates = np.where(np.asarray(DENSITY_PARAMETERS) == "T_cmb", 1e-3, 1e-5)
    assert np.all(worst <= gates), f"split and one-thread Jacobians differ by {worst}"


@pytest.mark.gpu
def test_matter_power_spectrum_under_jit(two_columns):
    """``solve_eb`` and ``matter_power_spectrum`` trace under ``jit``, every key traced.

    One trace serves every cosmology of the same keys, the values agree with
    the eager solve, the gradient under ``jit`` with the eager one, and a key
    the model does not have has a zero derivative.
    """

    import jax
    import jax.numpy as jnp

    from discoeb.models import LCDM
    from discoeb.observables.power_spectrum import (
        matter_power_spectrum,
    )
    from discoeb.solver._solver import solve_eb

    DIFFERENTIABLE_PARAMETERS = DENSITY_PARAMETERS
    traces = []

    def power(cosmology):
        traces.append(cosmology)
        batch = cosmologies_of(LCDM, [cosmology])
        keep = PK_KEEP
        values = solve_eb(batch, SOLVE_EB_K, [1.0], keep=keep, thermo_weight=0.1)
        today = dict(zip(keep, jnp.unstack(values[0, :, :, -1])))
        params = {
            key: getattr(batch[0], key)
            for key in ("omega_c", "omega_b", "A_s", "n_s", "k_p")
        }
        return matter_power_spectrum(SOLVE_EB_K, **today, **params)

    jitted = jax.jit(power)
    to_physical = jax.jit(lambda c: physical(LCDM, c))
    for cosmology in _perturbed_cosmologies(2):
        # The eager solve of the same physical parameters. Converted under
        # jit as well: compiled, the closure rounds omega_de differently in
        # the last bit, which moves a step-acceptance and P(k) by 1e-5.
        eager = power(to_physical(cosmology))
        np.testing.assert_allclose(
            np.asarray(jitted(cosmology)), np.asarray(eager), rtol=1e-12
        )
    assert sum(isinstance(c.get("H_0"), jax.core.Tracer) for c in traces) == 1

    cosmology = _perturbed_cosmologies(1)[0]
    theta = jnp.asarray([cosmology[n] for n in DIFFERENTIABLE_PARAMETERS])

    def total(t, of):
        return jnp.sum(of({**cosmology, **dict(zip(DIFFERENTIABLE_PARAMETERS, t))}))

    eager = np.asarray(jax.jacfwd(total)(theta, lambda c: power(to_physical(c))))
    under_jit = np.asarray(jax.jit(jax.jacfwd(lambda t: total(t, jitted)))(theta))
    np.testing.assert_allclose(under_jit, eager, rtol=1e-12)

    # LCDM has no dark-energy fluid: the kernel never reads w_DE_0
    slope = jax.jacfwd(lambda w: jnp.sum(jitted({**cosmology, "w_DE_0": w})))(-1.0)
    assert float(slope) == 0.0


MORE_PK_GATES: dict = {"z_re": 1.0e-2}
"""``dP/dz_re`` is gated loosely because its central difference is the noisy
side. ``z_re`` moves ``sum P`` by 1.4e-5 of itself per unit, so rounding
noise of 1e-7 in the converged solves moves the difference by half a
percent: -47.49 in CAMB's convention and -47.75 in the plain one, where the
sensitivity is -47.43 at rtol 1e-6 and -47.44 at the default in both
(2026-11-17)."""


@pytest.mark.gpu
@pytest.mark.parametrize("name", MORE_PARAMETER_CASES)
def test_more_parameters_matter_power(name):
    """``jax.jvp`` of ``sum P(k)`` in each parameter beyond the densities' against central differences.

    The derivative of ``P(k)`` today, at fixed scale factor. Central
    differences from converged solves. One direction, so each model's
    one-column kernel.
    """

    import jax
    import jax.numpy as jnp

    model, cosmology, step, keep = _more_parameter_case(name)
    # a model that keeps more than P(k)'s names keeps them on the shared grid
    a = 1.0 if keep == PK_KEEP else DIAG_A

    def total_power(p, **options):
        batch = cosmologies_of(model, {**cosmology, name: p})
        power = _matter_power(MATTER_POWER_K, batch, a=a, keep=keep, **options)
        return jnp.sum(power)

    p0 = jnp.asarray(float(cosmology[name]))
    grad = float(jax.jvp(total_power, (p0,), (jnp.ones_like(p0),))[1])
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
