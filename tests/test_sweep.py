"""Sweeps of parameter space: stability, accuracy against CLASS and self-convergence."""

import numpy as np
import pytest
from conftest import cosmologies_of
from test_perturbations import REFERENCE_COSMOLOGY, _class_base, _matter_power

SWEEP_SEED = 20261113
N_SWEEP = 32
SWEEP_K = np.geomspace(1.0e-4, 1.0, 64)

BOUNDS = {
    "Omegam": (0.2, 0.45),
    "Omegab": (0.035, 0.065),
    "H0": (55.0, 85.0),
    "Neff": (2.5, 4.0),
    "YHe": (0.2, 0.3),
}
"""The Latin hypercube's box, shared by every model; ``Tcmb``, ``A_s`` and ``n_s`` stay fixed."""

MODEL_BOUNDS = {
    "lcdm": {},
    "w0wacdm": {"w_DE_0": (-1.3, -0.7), "w_DE_a": (-0.5, 0.5)},
    "lcdm-mnu": {"mnu": (0.02, 0.6)},
}
"""Each model's own parameters. ``w_DE_0 + w_DE_a`` stays below -0.2 in the box."""

SWEEP_KEYS = (*BOUNDS, "w_DE_0", "w_DE_a", "mnu")

MATTER_THERMO_WEIGHT = 0.1
"""The thermal error weight of the P(k) solves, as ``_matter_power`` sets it."""


def _model(name):
    from discoeb.models import LCDM, LCDM_MNU, W0WACDM

    return {"lcdm": LCDM, "w0wacdm": W0WACDM, "lcdm-mnu": LCDM_MNU}[name]


def sweep(name, n=N_SWEEP, seed=SWEEP_SEED):
    """``n`` parameter dictionaries of model ``name`` on a Latin hypercube at a fixed seed."""

    from scipy.stats import qmc

    bounds = {**BOUNDS, **MODEL_BOUNDS[name]}
    lo, hi = np.array(list(bounds.values())).T
    points = qmc.scale(qmc.LatinHypercube(d=len(bounds), seed=seed).random(n), lo, hi)
    cosmologies = []
    for row in points:
        param = {**REFERENCE_COSMOLOGY, **dict(zip(bounds, map(float, row)))}
        if "mnu" in param and name == "lcdm-mnu":
            param["Nmnu"] = 1.0
        cosmologies.append(param)
    return cosmologies


def _crosses_phantom_divide(param):
    """Whether CPL's ``w(a)`` passes through -1 for ``0 < a < 1``."""

    today = param["w_DE_0"] + 1.0
    early = param["w_DE_0"] + param["w_DE_a"] + 1.0
    return today * early < 0.0


def _matter_power_and_steps(model, cosmologies, **options):
    """``P(k)`` at :data:`SWEEP_K` through ``solve_eb``, and each trajectory's step count.

    The steps come from a second launch of the same trajectories with the
    same settings, through the history solve, whose modax call returns them:
    ``(n_k, n_cosmologies)`` loop steps, which stop at ``max_steps``.
    """

    import jax

    from discoeb.kernel import _solve_batch

    batch = cosmologies_of(model, cosmologies)
    pk = np.asarray(jax.block_until_ready(_matter_power(SWEEP_K, batch, **options)))
    (_, stats), _, k_sorted = _solve_batch(
        model.spec,
        cosmologies,
        SWEEP_K,
        thermo_weight=MATTER_THERMO_WEIGHT,
        return_stats=True,
        **options,
    )
    steps = np.asarray(stats["loop_steps"]).reshape(len(k_sorted), len(cosmologies))
    return pk, steps


def _class_pk(param, k_values):
    """CLASS's linear ``P(k)`` of the cold dark matter and baryons today, in Mpc^3.

    The settings of the existing comparison, plus the dark-energy fluid
    (without PPF where ``w`` does not cross -1, so the same equations as
    DISCO-EB's) and one massive species at DISCO-EB's neutrino temperature.
    """

    Class = pytest.importorskip("classy").Class
    settings = {
        **_class_base(param),
        "A_s": param["A_s"],
        "n_s": param["n_s"],
        "k_pivot": 0.05,
        "output": "mPk",
        "P_k_max_1/Mpc": float(k_values[-1]) * 1.1,
        "z_max_pk": 0.0,
    }
    if param.get("w_DE_0", -1.0) != -1.0 or param.get("w_DE_a", 0.0) != 0.0:
        settings.update(
            {
                "Omega_Lambda": 0.0,
                "w0_fld": param["w_DE_0"],
                "wa_fld": param["w_DE_a"],
                "cs2_fld": param["cs2_DE"],
                "use_ppf": "yes" if _crosses_phantom_divide(param) else "no",
            }
        )
    if param.get("Nmnu", 0.0):
        settings.update(
            {
                "N_ncdm": 1,
                "deg_ncdm": param["Nmnu"],
                "m_ncdm": param["mnu"],
                "T_ncdm": (4.0 / 11.0) ** (1.0 / 3.0),
            }
        )
    cosmo = Class()
    cosmo.set(settings)
    try:
        cosmo.compute()
        # CLASS has P_cb only beside massive neutrinos; without them it is P_m
        pk = cosmo.pk_cb_lin if settings.get("N_ncdm") else cosmo.pk_lin
        return np.array([pk(float(k), 0.0) for k in k_values])
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()


def _swept_cosmologies(name):
    """The sweep of model ``name`` the tests gate: CPL histories that cross -1 left out."""

    return [c for c in sweep(name) if not _crosses_phantom_divide(c)]


@pytest.fixture(scope="module", params=list(MODEL_BOUNDS))
def swept(request):
    """One model's sweep, solved in one batch: ``(name, cosmologies, P(k), steps)``."""

    name = request.param
    cosmologies = _swept_cosmologies(name)
    pk, steps = _matter_power_and_steps(_model(name), cosmologies)
    return name, cosmologies, pk, steps


def test_sweep_is_stable(swept):
    """Every cosmology of the sweep gives a finite, positive ``P(k)`` well within ``max_steps``.

    Measured 2026-11-13 (64 modes): the most loop steps a trajectory took was
    469 for LCDM, 501 for W0WACDM and 1105 for LCDM_MNU, against a
    ``max_steps`` of 20000.
    """

    from discoeb.kernel import PERTURB_MAX_STEPS

    name, cosmologies, pk, steps = swept
    assert pk.shape == (len(cosmologies), len(SWEEP_K))
    bad = ~(np.isfinite(pk) & (pk > 0.0)).all(axis=1)
    assert not bad.any(), (
        f"{name}: P(k) not finite and positive for {np.flatnonzero(bad)}"
    )
    worst = int(np.argmax(steps.max(axis=0)))
    assert steps.max() < PERTURB_MAX_STEPS // 10, (
        f"{name}: {steps.max()} steps for {cosmologies[worst]}"
    )


SWEEP_PK_GATES = {"lcdm": 6.0e-3, "w0wacdm": 7.0e-3, "lcdm-mnu": 9.0e-3}
"""Largest ``|P / P_CLASS - 1|`` over the sweep's modes and cosmologies.

Measured 2026-11-13 over ``k`` in 1e-4 to 1 Mpc^-1 (median of each
cosmology's maximum, then the sweep's maximum):

- ``lcdm``, 32 cosmologies: 3.7e-3, 5.0e-3, at the highest ``k`` (8.6e-1 to
  1); below ``k = 0.3`` at most 2.5e-3, below 0.01 at most 1.4e-4.
- ``w0wacdm``, the 25 that do not cross -1: 3.5e-3, 5.9e-3, also at the
  highest ``k``.
- ``lcdm-mnu``, 32 with one species of 0.02 to 0.6 eV: 3.7e-3, 7.7e-3, the
  largest at ``k ~ 0.1`` for the heaviest masses (0.49 to 0.58 eV), the
  three-node momentum quadrature (``NQMAX = 3``).
"""


def test_sweep_matches_class(swept):
    """The sweep's ``P(k)`` agrees with CLASS (default precision) to the measured gate."""

    name, cosmologies, pk, _ = swept
    pk_class = np.stack([_class_pk(param, SWEEP_K) for param in cosmologies])
    rel = np.abs(pk / pk_class - 1.0)
    i, j = np.unravel_index(np.argmax(rel), rel.shape)
    assert rel[i, j] < SWEEP_PK_GATES[name], (
        f"{name}: max relative error {rel[i, j]:.4g} at k={SWEEP_K[j]:.4g} Mpc^-1 "
        f"for {cosmologies[i]}"
    )


CROSSING_COSMOLOGY = {
    **REFERENCE_COSMOLOGY,
    "Omegam": 0.2566426945368695,
    "Omegab": 0.03777689274686536,
    "H0": 70.39801665473514,
    "Neff": 3.544575558736806,
    "YHe": 0.27119999546858503,
    "w_DE_0": -1.0190233214019964,
    "w_DE_a": 0.3578979092001213,
}
"""The mildest of the sweep's seven CPL cosmologies whose ``w`` crosses -1
(at ``a = 0.95``): P(k) 5.5 percent off CLASS (with PPF)."""


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "W0WACDM's fluid divides by 1 + w: a w(a) crossing -1 runs out of steps "
        "(20000) and returns a finite, positive P(k) 5.5 to 99 percent off CLASS, "
        f"with no error raised. E.g. {CROSSING_COSMOLOGY}"
    ),
)
def test_dark_energy_crossing_minus_one():
    """CPL histories that cross ``w = -1``, which the model accepts without warning.

    The sweep's seven crossing cosmologies hit ``max_steps`` on every one and
    miss CLASS by 5.5e-2 to 9.9e-1 (measured 2026-11-13).
    """

    from discoeb.kernel import PERTURB_MAX_STEPS

    cosmologies = [c for c in sweep("w0wacdm") if _crosses_phantom_divide(c)]
    assert CROSSING_COSMOLOGY in cosmologies
    pk, steps = _matter_power_and_steps(_model("w0wacdm"), cosmologies)
    assert steps.max() < PERTURB_MAX_STEPS
    pk_class = np.stack([_class_pk(param, SWEEP_K) for param in cosmologies])
    assert np.abs(pk / pk_class - 1.0).max() < SWEEP_PK_GATES["w0wacdm"]


CONVERGENCE_COSMOLOGIES = {
    "lcdm": REFERENCE_COSMOLOGY,
    "w0wacdm": {**REFERENCE_COSMOLOGY, "w_DE_0": -0.9, "w_DE_a": 0.1},
    "lcdm-mnu": {**REFERENCE_COSMOLOGY, "Nmnu": 1.0, "mnu": 0.06},
}
"""One cosmology per model for the self-convergence test: Planck 2018, with
a CPL fluid that stays above -1, or one massive species of 0.06 eV."""

TIGHT = {"rtol": 1.0e-7, "atol": 1.0e-9, "lu_precision": "fp64", "max_steps": 200_000}
"""Solver settings of the converged reference: 1000 times tighter than the defaults."""


def _converged_matter_power(model, cosmology):
    """``P(k)`` at :data:`SWEEP_K` at :data:`TIGHT` settings, from the final state.

    ``solve_eb`` does not take ``lu_precision``, so this reads the CDM and
    baryon contrasts today off the history solve and weights them as
    ``compute_matter_power_spectrum`` does. Also returns the loop steps.
    """

    from discoeb.background import TAU_START, density_coefficients
    from discoeb.cosmology import as_cosmology
    from discoeb.kernel import _solve_batch
    from discoeb.solver import conformal_ages

    tau_today = float(conformal_ages(cosmology).max())
    (hist, stats), prepared, k_sorted = _solve_batch(
        model,
        (cosmology,),
        SWEEP_K,
        np.array([TAU_START, tau_today]),
        return_stats=True,
        **TIGHT,
    )
    # SWEEP_K is ascending, so k_sorted is SWEEP_K
    final = np.asarray(hist).reshape(len(k_sorted), -1, prepared.model.nvar)[:, -1]
    index = prepared.model.index
    c = as_cosmology(cosmology)
    grhoc, grhob = density_coefficients(c)[2:4]
    delta_m = (grhoc * final[:, index["clxc"]] + grhob * final[:, index["clxb"]]) / (
        grhoc + grhob
    )
    k = np.asarray(k_sorted)
    pk = 2.0 * np.pi**2 / k**3 * c.A_s * (k / c.k_pivot) ** (c.n_s - 1.0) * delta_m**2
    return pk, np.asarray(stats["loop_steps"])


SELF_CONVERGENCE_GATE = 3.0e-4
"""Largest ``|P / P_converged - 1|`` of a default solve against :data:`TIGHT`.

Measured 2026-11-13 over 64 modes: 1.9e-4 (LCDM, at k = 0.011), 1.3e-4
(W0WACDM, k = 0.005), 1.8e-4 (LCDM_MNU, k = 0.019), median 1.8e-5, and
1.4e-5 above k = 0.3. The converged solve differs from CLASS by 3.4e-3 to
3.6e-3 like the default one, so the CLASS gap is not the solver's tolerance.
The tight solve takes 3000 (LCDM) to 13000 (LCDM_MNU) steps, and the
three tests 117 s, mostly compiling the fp64-LU kernels: ``pytest -m slow``.
"""


@pytest.mark.slow
@pytest.mark.parametrize("name", list(CONVERGENCE_COSMOLOGIES))
def test_self_convergence(name):
    """A default-tolerance ``P(k)`` agrees with a 1000 times tighter, fp64-LU solve."""

    cosmology = CONVERGENCE_COSMOLOGIES[name]
    pk, _ = _matter_power_and_steps(_model(name), [cosmology])
    converged, steps = _converged_matter_power(_model(name), cosmology)
    assert steps.max() < TIGHT["max_steps"]
    rel = np.abs(pk[0] / converged - 1.0)
    j = int(np.argmax(rel))
    assert rel[j] < SELF_CONVERGENCE_GATE, (
        f"{name}: {rel[j]:.3g} from the converged P(k) at k={SWEEP_K[j]:.4g} Mpc^-1"
    )
