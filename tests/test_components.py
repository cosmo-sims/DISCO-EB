"""Tests of ``discoeb.components``, one section per module."""

import math

import numpy as np
import pytest
from helpers import (
    BATCH_SIZE_CASES,
    COSMOLOGY,
    HOST,
    LCDM_REIO,
    LCDM_X,
    LCDM_XY,
    MODELS,
    N_PARAM,
    PARAM_COLUMNS,
    REFERENCE_COSMOLOGY,
    ConstantThermalHistory,
    X,
    Y,
    _class_base,
    _perturbed_cosmologies,
    _split,
    build_rhs,
    cosmologies_of,
    model_parameters,
    on_the_host,
    state_index,
)
from jax.typing import ArrayLike

from discoeb.components import gravity, species
from discoeb.components._component import (
    Component,
    Equations,
    Gravity,
    Interaction,
    Parameter,
    Setting,
    Species,
    State,
    Sum,
    contributes,
    cuda,
    float64,
    load_setting,
)
from discoeb.components._util import (
    generalized_gauss_laguerre_weights,
    neutrino_momentum_bins,
)
from discoeb.components.gravity import GRHOM_PER_H0SQ, Curvature, Metric
from discoeb.components.species import (
    Fluid,
    MassiveNeutrinos,
    NoRadiationStreaming,
    RadiationStreaming,
)
from discoeb.models import LCDM, W0WACDM_MNU
from discoeb.solver._kernel import make_initial_states

# ============================================================================
# component: The component framework (``_component``): providers, states, parameters, named copies, the traced sparsity
# ============================================================================

# --- The component framework: dependency injection by name, states, settings, errors.


FLOOR = 1e-30  # a module constant a provider reads


# Toy components. They share the names of the package's own, which the
# sections below therefore reach as ``gravity.Background``,
# ``species.ColdDarkMatter`` and ``species.Photons``.


class Background(Component):
    grho = Sum()

    def inv_a(a):
        return 1.0 / a

    def adotoa(grho):
        return math.sqrt(grho / 3.0)


class ColdDarkMatter(Species):
    clxc = State(initial=lambda clxg: 0.75 * clxg)

    def grho(grhoc, inv_a):
        return grhoc * inv_a

    def d_clxc(clxc, tau):
        return clxc / tau


class Baryons(Species):
    clxb = State(initial=lambda clxg: 0.75 * clxg)
    vb = State()

    def grho(grhob, inv_a):
        return grhob * inv_a

    def d_clxb(vb):
        return -vb

    def d_vb(a, adotoa, vb, cs2b, k, clxb):
        return -a * adotoa * vb + cs2b * k * k * clxb


class Photons(Species):
    clxg = State(initial=lambda k, tau: (k * tau) ** 2 / 3.0)
    qg = State()

    def grho(grhog, inv_a):
        return grhog * inv_a * inv_a

    def d_clxg(qg):
        return -4.0 / 3.0 * qg

    def d_qg(k, clxg):
        return 0.25 * k * k * clxg


class ThomsonScattering(Interaction):
    def R(grhog, grhob, a):
        return 4.0 * grhog / (3.0 * grhob * a)

    def kappa(opacity):
        return max(opacity, FLOOR)

    def d_vb(R, kappa, qg, vb):
        return R * kappa * (qg - vb)

    def d_qg(kappa, qg, vb):
        return -kappa * (qg - vb)


PARAMS = {"grhoc": 0, "grhob": 1, "grhog": 2, "cs2b": 3, "k": 4, "opacity": 5, "a": 6}


Y0 = (0.3, 0.2, 0.1, 0.4, 0.05)


TAU = 2.0


P = (1.0, 0.5, 2.0, 0.01, 0.3, 7.0, 0.1)


def reference(y, tau, p):
    clxc, clxb, vb, clxg, qg = y
    grhoc, grhob, grhog, cs2b, k, opacity, a = p
    H = math.sqrt((grhoc / a + grhob / a + grhog / a**2) / 3.0)
    R = 4.0 * grhog / (3.0 * grhob * a)
    return (
        clxc / tau,
        -vb,
        -a * H * vb + cs2b * k * k * clxb + R * opacity * (qg - vb),
        -4.0 / 3.0 * qg,
        0.25 * k * k * clxg - opacity * (qg - vb),
    )


def toy_model(*extra):
    return Equations(
        [
            Background(),
            ColdDarkMatter(),
            Baryons(),
            Photons(),
            ThomsonScattering(),
            *extra,
        ]
    )


def test_the_built_rhs_evaluates_the_equations():
    model = toy_model()
    rhs = model.build(PARAMS, **HOST)
    assert model.state == ["clxc", "clxb", "vb", "clxg", "qg"]
    np.testing.assert_allclose(
        rhs(Y0, TAU, P), reference(Y0, TAU, P), rtol=1e-15, atol=0
    )
    assert rhs._generated_source == model.source


def test_the_driver_injects_by_name_in_dependency_order():
    """Baryons and Thomson scattering both write ``vb``; Photons and Thomson ``qg``."""
    model = toy_model()
    model.build(PARAMS, **HOST)
    lines = [ell.strip() for ell in model.source.splitlines()]
    assert (
        "d_vb = Baryons__d_vb(a, adotoa, vb, cs2b, k, clxb) + ThomsonScattering__d_vb(R, kappa, qg, vb)"
        in lines
    )
    assert (
        "d_qg = Photons__d_qg(k, clxg) + ThomsonScattering__d_qg(kappa, qg, vb)"
        in lines
    )
    total = "grho = ColdDarkMatter__grho(grhoc, inv_a) + Baryons__grho(grhob, inv_a) + Photons__grho(grhog, inv_a)"
    assert total in lines
    assert (
        lines.index("inv_a = Background__inv_a(a)")
        < lines.index(total)
        < lines.index("adotoa = Background__adotoa(grho)")
    )
    # Every y and p access is a literal index.
    for line in lines:
        if "y[" in line or "p[" in line:
            assert line.split("[")[1].split("]")[0].replace(" + i", "").isdigit()


def test_only_what_the_derivatives_need_is_compiled():
    class Unused(Component):
        def never_read(a):
            return a * a

    compiled = []
    model = toy_model(Unused())
    model.build(PARAMS, **{**HOST, "jit": lambda f: compiled.append(f.__name__) or f})
    assert "never_read" not in compiled and "never_read" not in model.source
    assert "inv_a" in compiled


def test_initial_conditions_are_declared_on_the_states():
    """``clxc`` and ``clxb`` start at 3/4 of the photon density, which starts at (k tau)^2 / 3."""
    model = toy_model()
    y0 = model.initial_state([0.1, 0.2], tau=2.0)
    clxg = (np.array([0.1, 0.2]) * 2.0) ** 2 / 3.0
    np.testing.assert_allclose(y0[:, model.index["clxg"]], clxg)
    np.testing.assert_allclose(y0[:, model.index["clxc"]], 0.75 * clxg)
    np.testing.assert_allclose(y0[:, model.index["clxb"]], 0.75 * clxg)
    assert (
        y0[:, [model.index["vb"], model.index["qg"]]] == 0.0
    ).all()  # no initial condition: zero


def stream(kv, x, inv_tau, output):
    """A helper shared between hierarchies, compiled for the device by the framework."""
    lmax = len(x) - 1
    for ell in range(1, lmax):
        output[ell] = (
            kv * (ell * x[ell - 1] - (ell + 1) * x[ell + 1]) * (1.0 / (2 * ell + 1))
        )
    output[lmax] = kv * x[lmax - 1] - (lmax + 1) * x[lmax] * inv_tau


class Hierarchy(Species):
    LMAX = Setting(3)

    def _x_initial(k, output):
        output[0] = k
        output[1] = -k

    x = State(lambda self: self.LMAX + 1, initial=_x_initial)

    def inv_tau(tau):
        return 1.0 / tau

    def d_x(k, x, inv_tau, output):
        output[0] = -k * x[1]
        stream(k, x, inv_tau, output)


def test_vector_states_are_arrays_and_their_output_buffers_are_provided():
    compiled = []
    model = Equations([Hierarchy()])
    rhs = model.build(
        {"k": 0}, **{**HOST, "jit": lambda f: compiled.append(f.__name__) or f}
    )
    assert model.state == ["x_0", "x_1", "x_2", "x_3"]
    k, tau, x = 0.7, 3.0, (1.0, 2.0, 4.0, 8.0)
    np.testing.assert_allclose(
        rhs(x, tau, (k,)),
        (-k * 2.0, k * (1.0 - 8.0) / 3.0, k * (4.0 - 24.0) / 5.0, k * 4.0 - 32.0 / tau),
    )
    # The helper the provider calls is compiled too, before the provider.
    assert compiled.index("stream") < compiled.index("d_x")
    lines = [ell.strip() for ell in model.source.splitlines()]
    assert "c_Hierarchy__d_x = cuda.local.array(4, float64)" in lines
    assert "Hierarchy__d_x(k, x, inv_tau, c_Hierarchy__d_x)" in lines
    assert "d_x[i] += c_Hierarchy__d_x[i]" in lines
    assert "d_x[i] = -0.0" in lines
    # The vector's initial condition fills an output too.
    np.testing.assert_array_equal(model.initial_state([2.0]), [[2.0, -2.0, 0.0, 0.0]])


def pattern(model):
    return {(int(r), int(c)) for r, c in model.sparsity()}


def finite_difference_pattern(f, y, step=1e-6):
    """The nonzeros of a central-difference Jacobian of ``f`` at ``y``."""
    y = np.asarray(y, dtype=float)
    columns = []
    for j in range(len(y)):
        e = np.zeros_like(y)
        e[j] = step
        columns.append((np.asarray(f(y + e)) - np.asarray(f(y - e))) / (2 * step))
    return {(int(r), int(c)) for r, c in zip(*np.nonzero(np.array(columns).T))}


def test_the_jacobian_pattern_is_traced_through_the_providers():
    """``stream`` couples each multipole to its neighbours; nothing declares that."""
    model = Equations([Hierarchy()])
    tridiagonal = {
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
        (2, 3),
        (3, 2),
        (3, 3),
    }  # x_3 damps itself
    assert pattern(model) == tridiagonal
    rhs = model.build({"k": 0}, **HOST)
    assert (
        finite_difference_pattern(
            lambda y: rhs(tuple(y), 3.0, (0.7,)), (1.0, 2.0, 4.0, 8.0)
        )
        == tridiagonal
    )

    class Core(Species):
        v = State(2)

        def d_v(k, v, output):
            output[0] = k * v[1]  # v_1 has no derivative of its own: an empty row

    assert pattern(Equations([Core(), Hierarchy()])) == {(0, 1)} | {
        (r + 2, c + 2) for r, c in tridiagonal
    }

    # The toy model, against the hand-written reference: the pattern threads
    # through the sums and the interaction (vb and qg through the drag).
    model = toy_model()
    expected = finite_difference_pattern(lambda y: reference(y, TAU, P), Y0)
    assert (
        pattern(model)
        == expected
        == {(0, 0), (1, 2), (2, 1), (2, 2), (2, 4), (3, 4), (4, 2), (4, 3), (4, 4)}
    )


def test_providers_of_inputs_alone_are_not_traced():
    """``kappa`` floors ``opacity`` with ``max``, which a traced value refuses: it is never called."""

    class Tables(Component):
        def __init__(self):
            self.calls = 0

            def a(tau):
                self.calls += 1
                return int(tau) + 1.0  # int() of a traced value would raise

            self.a = a

    tables = Tables()
    model = toy_model(tables)
    assert pattern(model) == pattern(toy_model()) and tables.calls == 0
    # ... whereas a numeric evaluation runs it.
    assert model.evaluate("a", tau=2.7) == 3.0 and tables.calls == 1


def test_a_branch_on_the_state_reads_every_input():
    """A provider that branches on the state depends on all it reads: a superset, never a miss."""

    class Switch(Species):
        u = State()
        v = State()

        def d_u(u, v, k):
            return k * u if v > 0.0 else -k * u  # v picks the branch; both read only u

        def d_v(v, u, k):
            return -k * v

    rows = {tuple(e) for e in Equations([Switch()]).sparsity()}
    assert rows == {(0, 0), (0, 1), (1, 1)}

    class Floor(Species):
        u = State()

        def d_u(u, k):
            return (
                max(k, 1e-30) * u
            )  # a branch on an input, inside a provider that reads the state

    assert {tuple(e) for e in Equations([Floor()]).sparsity()} == {(0, 0)}


def test_settings_are_class_attributes_loaded_from_the_environment(monkeypatch):
    monkeypatch.setenv("DISCO_LMAX_T", "7")
    monkeypatch.setenv("DISCO_SCALE", "2.5")
    monkeypatch.setenv("DISCO_FLAG", "yes")

    class Tuned(Component):
        LMAX_T = Setting(3)
        SCALE = Setting(1.0)
        FLAG = Setting(False)
        UNSET = Setting(4)

    assert (Tuned.LMAX_T, Tuned.SCALE, Tuned.FLAG, Tuned.UNSET) == (7, 2.5, True, 4)
    assert Tuned.setting_names() == ("LMAX_T", "SCALE", "FLAG", "UNSET")
    instance = Tuned(LMAX_T=11)
    assert (instance.LMAX_T, instance.SCALE, Tuned.LMAX_T) == (11, 2.5, 7)
    assert instance.constants() == {
        "LMAX_T": 11,
        "SCALE": 2.5,
        "FLAG": True,
        "UNSET": 4,
    }
    with pytest.raises(TypeError, match="has no settings \\['NOPE'\\]"):
        Tuned(NOPE=1)

    class Derived(Tuned):
        EXTRA = Setting("x")

    assert Derived.setting_names() == ("LMAX_T", "SCALE", "FLAG", "UNSET", "EXTRA")
    assert (
        Equations([Hierarchy()]).nvar == 4 and Equations([Hierarchy(LMAX=6)]).nvar == 7
    )

    monkeypatch.setenv("DISCO_LMAX_T", "seven")
    with pytest.raises(ValueError, match="not a int"):
        load_setting("LMAX_T", 3)
    with pytest.raises(ValueError, match="UPPER_CASE"):

        class Lower(Component):
            lmax = Setting(3)

    with pytest.raises(TypeError, match="bool, int, float or str"):
        Setting([1, 2])


def test_device_names_are_replaced_when_a_provider_is_compiled():
    """``cuda`` and ``float64`` are placeholders until then; a local array's size is a literal."""
    with pytest.raises(RuntimeError, match="only available inside a provider"):
        cuda.local.array(3, float64)

    class Scratch(Species):
        N = Setting(3)
        v = State(lambda self: self.N)

        def d_v(self, k, v, output):
            tmp = cuda.local.array(3, float64)  # numba needs a literal size, not self.N
            for i in range(self.N):
                tmp[i] = k * v[i]
            for i in range(self.N):
                output[i] = -tmp[i]

    seen = {}

    def jit(f):
        seen[f.__name__] = f.__globals__
        return f

    rhs = Equations([Scratch()]).build({"k": 0}, **{**HOST, "jit": jit})
    np.testing.assert_allclose(rhs((1.0, 2.0, 3.0), 0.0, (2.0,)), (-2.0, -4.0, -6.0))
    assert seen["d_v"]["cuda"] is HOST["cuda"]
    # The module's own globals are untouched.
    assert Scratch.d_v.__globals__["cuda"] is cuda


def test_partial_writes_of_a_vector_accumulate():
    """Each writer gets a zeroed buffer; entries it leaves alone contribute nothing."""

    class Damping(Interaction):
        def d_x(opacity, x, output):
            for ell in range(2, len(x)):
                output[ell] = -opacity * x[ell]

    model = Equations([Hierarchy(), Damping()])
    rhs = model.build({"k": 0, "opacity": 1}, **HOST)
    k, tau, x, op = 0.7, 3.0, (1.0, 2.0, 4.0, 8.0), 5.0
    undamped = Equations([Hierarchy()]).build({"k": 0}, **HOST)(x, tau, (k,))
    np.testing.assert_allclose(
        rhs(x, tau, (k, op)), np.add(undamped, (0.0, 0.0, -op * 4.0, -op * 8.0))
    )
    lines = [ell.strip() for ell in model.source.splitlines()]
    assert lines.index("Hierarchy__d_x(k, x, inv_tau, c_Hierarchy__d_x)") < lines.index(
        "Damping__d_x(opacity, x, c_Damping__d_x)"
    )


def test_a_contribution_says_what_it_adds_to():
    """A method under a name of its own adds to a sum or to another component's derivative.

    So no two components of a model define one method differently, which a
    type checker would report of the model class that inherits them all.
    """

    class Source(Component):
        x = State()

        def d_x(self, k):
            return k

        @contributes("total")
        def total_source(self, x):
            return 2.0 * x

    class Sink(Interaction):
        total = Sum()

        @contributes("total")
        def total_sink(self):
            return 1.0

        @contributes("d_x")
        def d_x_sink(self, x, total):
            return -x * total

    model = Equations([Source(), Sink()])
    x, k = 3.0, 0.5
    rhs = model.build({"k": 0}, **HOST)
    np.testing.assert_allclose(rhs((x,), 1.0, (k,)), (k - x * (2.0 * x + 1.0),))
    assert Source.provided_names() == ("total",) and Sink.provided_names() == ("total",)

    class Stray(Component):
        @contributes("other")
        def other_stray(self):
            return 1.0

    with pytest.raises(TypeError, match="neither one of the model's sums"):
        Equations([Source(), Sink(), Stray()]).build({"k": 0}, **HOST)


def test_matrix_states_and_tuple_constants():
    """A component written once for all bins; its weights are a tuple constant read through self."""

    class Bins(Species):
        NQ = Setting(2)

        def _psi_initial(self, k, output):
            for i in range(self.NQ):
                output[i, 0] = k * self.weight[i]

        psi = State(lambda self: (self.NQ, 2), initial=_psi_initial)

        def __init__(self, weights, **settings):
            super().__init__(**settings)
            self.weight = tuple(weights)

        def grho(self, a, psi):
            total = 0.0
            for i in range(self.NQ):
                total += self.weight[i] * psi[i, 0] * (a * self.weight[i])
            return total

        def d_psi(self, k, a, psi, output):
            for i in range(self.NQ):
                kv = k * a * self.weight[i]
                output[i, 0] = -kv * psi[i, 1]
                output[i, 1] = kv * psi[i, 0]

    class Reads(Species):
        grho = Sum()
        w = State()

        def d_w(grho):
            return grho

    model = Equations([Bins([0.5, 0.25]), Reads()])
    rhs = model.build({"k": 0, "a": 1}, **HOST)
    assert model.state == ["psi_0_0", "psi_0_1", "psi_1_0", "psi_1_1", "w"]
    k, a, y = 3.0, 2.0, (1.0, 2.0, 4.0, 8.0, 0.0)
    v0, v1 = a * 0.5, a * 0.25
    np.testing.assert_allclose(
        rhs(y, 1.0, (k, a)),
        (
            -k * v0 * 2.0,
            k * v0 * 1.0,
            -k * v1 * 8.0,
            k * v1 * 4.0,
            0.5 * 1.0 * v0 + 0.25 * 4.0 * v1,
        ),
    )
    assert "psi = cuda.local.array((2, 2), float64)" in model.source
    assert "d_psi[i, j] += c_Bins__d_psi[i, j]" in model.source
    assert "Bins__d_psi(Bins_self, k, a, psi, c_Bins__d_psi)" in model.source
    # On the device self is a named tuple of the constants, one field per type.
    device_self = Bins([0.5, 0.25]).device_self()
    assert type(device_self).__name__ == "Bins" and device_self._fields == (
        "NQ",
        "weight",
    )
    assert device_self.NQ == 2 and device_self.weight == (0.5, 0.25)
    # Matrix initial conditions and sparsity work element by element: each bin
    # couples only to itself, and w reads the bins' first elements.
    np.testing.assert_array_equal(
        model.initial_state([2.0]), [[1.0, 0.0, 0.5, 0.0, 0.0]]
    )
    assert pattern(model) == {(0, 1), (1, 0), (2, 3), (3, 2), (4, 0), (4, 2)}


def test_a_device_function_a_component_holds_is_called_through_a_closure():
    class Tables(Component):
        def __init__(self, table):
            def a(i):
                return max(table(i), 1e-30)

            self.a = a

        def root(a):
            return math.sqrt(a**2 + 1.0)

    class Reads(Species):
        w = State()

        def d_w(root):
            return root

    model = Equations([Tables({0: 2.0, 1: -1.0}.get), Reads()])
    rhs = model.build({"i": 0}, **HOST)
    assert rhs((0.0,), 1.0, (0,)) == (math.sqrt(5.0),)
    assert rhs((0.0,), 1.0, (1,)) == (math.sqrt(1e-60 + 1.0),)


def test_every_state_needs_a_derivative():
    class Silent(Species):
        q = State()

    with pytest.raises(ValueError, match="derivatives of \\['q'\\]"):
        toy_model(Silent()).build(PARAMS, **HOST)


def test_an_unprovided_dependency_names_the_provider():
    with pytest.raises(NameError, match="Baryons.d_vb needs 'cs2b'"):
        toy_model().build({k: v for k, v in PARAMS.items() if k != "cs2b"}, **HOST)
    with pytest.raises(NameError, match="'tau' is neither given nor provided"):
        toy_model().initial_state([0.1])


def test_two_providers_of_one_quantity_are_rejected():
    class Other(Component):
        def inv_a(a):
            return 1.0 / a

    with pytest.raises(ValueError, match="both provide 'inv_a'"):
        toy_model(Other()).build(PARAMS, **HOST)


def test_names_that_clash_with_state_elements_are_rejected():
    class Alias(Component):
        def x_2(x):
            return x[0]

    with pytest.raises(ValueError, match="provides 'x_2', which is a state"):
        Equations([Hierarchy(), Alias()]).build({"k": 0}, **HOST)

    class Scalar(Species):
        x_1 = State()

        def d_x_1(x_1):
            return x_1

    with pytest.raises(ValueError, match="clashes"):
        Equations([Hierarchy(), Scalar()])


def test_output_is_only_for_vector_derivatives_and_initial_conditions():
    class Stray(Interaction):
        def d_nothing(k):
            return k

    with pytest.raises(ValueError, match="writes 'nothing'"):
        toy_model(Stray()).build(PARAMS, **HOST)

    class Returns(Interaction):
        def d_x(k, x):
            return k

    with pytest.raises(TypeError, match="of a vector takes 'output' and fills it"):
        Equations([Hierarchy(), Returns()]).build({"k": 0}, **HOST)

    class TakesOutput(Interaction):
        def d_vb(k, output):
            return k

    with pytest.raises(TypeError, match="of a scalar returns the value"):
        toy_model(TakesOutput()).build(PARAMS, **HOST)

    class Quantity(Interaction):
        def inv_tau(tau, output):
            return 1.0 / tau

    with pytest.raises(TypeError, match="only a vector state's derivative fills"):
        Equations([Hierarchy(), Quantity()]).build({"k": 0}, **HOST)

    class BadInitial(Species):
        v = State(2, initial=lambda k: k)

        def d_v(k, v, output):
            output[0] = k

    with pytest.raises(TypeError, match="takes 'output' and fills it"):
        Equations([BadInitial()]).initial_state([1.0])


def test_cycles_are_reported():
    class Loop(Component):
        def u(v):
            return v

        def v(u):
            return u

    class Reads(Species):
        w = State()

        def d_w(u):
            return u

    with pytest.raises(ValueError, match="circular dependency"):
        Equations([Loop(), Reads()]).build({}, **HOST)

    class Chicken(Species):
        c = State(initial=lambda c: c)

        def d_c(c):
            return c

    with pytest.raises(ValueError, match="circular dependency on the host"):
        Equations([Chicken()]).initial_state([1.0])


def test_methods_are_providers_with_or_without_self_or_staticmethod():
    """A method reads its component through self; one that reads nothing may leave it out."""

    class Scaled(Species):
        SCALE = Setting(2.0)
        x = State(initial=lambda self, k: self.SCALE * k)

        def __init__(self, shift, **settings):
            super().__init__(**settings)
            self.shift = float(shift)

        def d_x(self, x, k):
            return self.SCALE * x + self.shift * k

        def k2(k):  # no self: a plain function on the class
            return k * k

        @staticmethod
        def half(k2):
            return 0.5 * k2

    class Reads(Species):
        w = State()

        def d_w(half):
            return half

    model = Equations([Scaled(0.5, SCALE=3.0), Reads()])
    rhs = model.build({"k": 0}, **HOST)
    assert rhs((2.0, 0.0), 1.0, (4.0,)) == (3.0 * 2.0 + 0.5 * 4.0, 8.0)
    assert Baryons.d_clxb(2.0) == -2.0
    np.testing.assert_array_equal(model.initial_state([4.0]), [[12.0, 0.0]])
    assert model.evaluate("half", k=4.0) == 8.0
    assert model.evaluate("x", k=1.5) == 4.5  # the initial condition read self too


def test_interactions_own_no_state():
    with pytest.raises(TypeError, match="cannot own state"):

        class Bad(Interaction):
            x = State()


def test_providers_are_plain_positional_functions():
    class Defaulted(Species):
        x = State()

        def d_x(x, k=1.0):
            return k * x

    with pytest.raises(TypeError, match="plain positional parameters"):
        Equations([Defaulted()]).build({"k": 0}, **HOST)


# --- One component listed several times, each copy under a name of its own.


def test_a_named_copy_prefixes_what_the_component_declares():
    assert Fluid.named("x") is X and X is not Y
    assert issubclass(X, Fluid) and X.__name__ == "Fluid_x"
    assert [n for n, _ in X.declared_states()] == ["x_delta", "x_theta"]
    assert [n for n, _ in X.declared_parameters()] == [
        "x_omega",
        "x_w",
        "x_cs2",
        "x_grho0",
    ]
    # the derived density reads the copy's own omega
    assert dict(X.declared_parameters())["x_grho0"].reads == ("x_omega",)
    with pytest.raises(TypeError, match="already carries a name"):
        X.named("z")
    with pytest.raises(ValueError, match="not a name"):
        Fluid.named("not a name")


def test_each_copy_checks_its_own_keys():
    cosmologies_of(LCDM_XY, {**COSMOLOGY, "x_Omega": 0.01, "y_w": 1.0 / 3.0})
    with pytest.raises(ValueError, match="w != -1"):
        cosmologies_of(LCDM_XY, {**COSMOLOGY, "y_w": -1.0})
    with pytest.raises(ValueError, match="use a model with Fluid_x"):
        cosmologies_of(LCDM, {**COSMOLOGY, "x_Omega": 0.01})


def test_names_resolve_in_initial_conditions_and_sums():
    class Tank(Component):
        level = State(initial=lambda start: 2.0 * start)

        def start(self):
            return 1.5

        def outflow(self, level):
            return 0.1 * level

        def total(self, level):
            return level

        def d_level(self, outflow):
            return -outflow

    class Pool(Component):
        total = Sum()

    model = Equations([Tank.named("a")(), Tank.named("b")(), Pool()])
    assert model.state == ["a_level", "b_level"]
    np.testing.assert_allclose(model.initial_state(np.array([1.0])), [[3.0, 3.0]])
    rhs = model.build({}, **HOST)
    assert rhs((1.0, 2.0), 0.0, ()) == (-0.1, -0.2)
    assert model.evaluate("total", a_level=1.0, b_level=2.0) == 3.0
    assert model.evaluate("d_b_level", a_level=1.0, b_level=2.0) == -0.2


# --- A species with a parameter of its own, added without touching the package.


def test_the_right_hand_side_reads_the_parameter_from_its_column():
    """``grhox`` is a column of the row, and clxx evolves as clxc does."""

    from discoeb.solver._kernel import parameter_layout

    columns = parameter_layout(LCDM_X).columns
    model = LCDM_X.build(ConstantThermalHistory())
    rhs = model.build(columns, **HOST)
    assert f"grhox = p[{columns['grhox']}]" in [
        line.strip() for line in model.source.splitlines()
    ]
    p = np.zeros(len(columns))
    for name, value in dict(
        grhog=1.5e-7, grhornomass=1.0e-7, grhoc=0.8e-4, grhob=1.9e-5, grhov=2.0e-7
    ).items():
        p[columns[f"{name}"]] = value
    p[columns["grhox"]] = 0.3e-4
    p[columns["k"]] = 0.05
    y = np.random.default_rng(0).standard_normal(model.nvar)
    y[model.index["clxx"]] = y[model.index["clxc"]]
    dy = np.asarray(rhs(tuple(y), 100.0, tuple(p)))
    assert np.isfinite(dy).all()
    assert dy[model.index["clxx"]] == dy[model.index["clxc"]]
    # Moving grhox moves dgrho as moving grhoc does.
    for name in ("grhox", "grhoc"):
        q = p.copy()
        q[columns[name]] += 1.0e-5
        moved = np.asarray(rhs(tuple(y), 100.0, tuple(q)))
        if name == "grhox":
            along_x = moved - dy
        else:
            np.testing.assert_allclose(moved - dy, along_x, rtol=1e-6, atol=1e-18)
    pattern = {(int(r), int(c)) for r, c in model.sparsity()}
    assert (model.index["etak"], model.index["clxx"]) in pattern


# ============================================================================
# gravity: Gravity (``gravity``): the Friedmann and Einstein equations, the sums and the growing mode
# ============================================================================

# --- The Einstein-Boltzmann component model, on the host.


def test_the_einstein_sources_sum_over_the_enabled_species():
    model, _ = build_rhs(MODELS["flat-lcdm"])
    lines = [ell.strip() for ell in model.source.splitlines()]
    assert (
        "grho = ColdDarkMatter__grho_c(ColdDarkMatter_self, grhoc_t) + Baryons__grho_b(Baryons_self, grhob_t)"
        " + Photons__grho_g(Photons_self, grhog_t) + MasslessNeutrinos__grho_r(MasslessNeutrinos_self, grhor_t) + CosmologicalConstant__grho_de(CosmologicalConstant_self, grhov_t)"
    ) in lines
    assert (
        "dgq = Baryons__dgq_b(Baryons_self, grhob_t, vb) + Photons__dgq_g(Photons_self, grhog_t, qg) + MasslessNeutrinos__dgq_r(MasslessNeutrinos_self, grhor_t, n, rsa, rsa_q)"
        in lines
    )
    assert "rhonu" not in model.source  # the massive-neutrino table is not read

    model, _ = build_rhs(MODELS["ragged-blocks"])
    assert (
        "DarkEnergy__grho_de(DarkEnergy_self, grhov_t) + MassiveNeutrinos__grho_nu(MassiveNeutrinos_self, nu_norm, rhonu)"
        in model.source
    )
    assert (
        "DarkEnergy__dgq_de(DarkEnergy_self, grhov_t, w_Q, thetaq, k) + MassiveNeutrinos__dgq_nu(MassiveNeutrinos_self, nu_norm, psi)"
        in model.source
    )
    assert "psi = cuda.local.array((2, 10), float64)" in model.source
    assert (
        "MassiveNeutrinos__d_psi(MassiveNeutrinos_self, k, z, sigma, a, amnu, psi, inv_tau, c_MassiveNeutrinos__d_psi)"
        in model.source
    )
    assert "d_psi[i, j] += c_MassiveNeutrinos__d_psi[i, j]" in model.source


def test_the_pressure_sums_over_the_species_and_gives_the_comoving_curvature():
    """``gpres`` is left out of the right-hand side and read by ``R`` alone."""

    for name in ("flat-lcdm", "ragged-blocks"):
        model, _ = build_rhs(MODELS[name])
        assert "gpres" not in model.source
    model = MODELS["ragged-blocks"].build(ConstantThermalHistory())
    fn = model.build_function(
        ("R", "gpres", "grho", "adotoa"),
        inputs=("y", "tau"),
        params=PARAM_COLUMNS,
        **HOST,
    )
    lines = [line.strip() for line in fn._generated_source.splitlines()]
    assert (
        "gpres = Photons__gpres_g(Photons_self, grhog_t) + MasslessNeutrinos__gpres_r(MasslessNeutrinos_self, grhor_t)"
        " + DarkEnergy__gpres_de(DarkEnergy_self, grhov_t, w_Q) + MassiveNeutrinos__gpres_nu(MassiveNeutrinos_self, nu_norm, pnu)"
    ) in lines

    # Flat LCDM with the fluids at rest: R is the curvature eta alone, and
    # the velocity term is the total momentum over rho + p.
    cosmology_model = MODELS["flat-lcdm"]
    model = cosmology_model.build(ConstantThermalHistory(a=1.0))
    fn = model.build_function(
        ("R", "gpres", "grho", "adotoa"),
        inputs=("y", "tau"),
        params=PARAM_COLUMNS,
        **HOST,
    )
    p = np.zeros(N_PARAM)
    grhog, grhor, grhoc, grhob, grhov = 1.5e-7, 1.0e-7, 1.1e-4, 1.9e-5, 2.0e-7
    p[:5] = [grhog, grhor, grhoc, grhob, grhov]
    p[PARAM_COLUMNS["k"]] = 0.05
    p[PARAM_COLUMNS["w_DE_0"]] = -1.0
    y = np.zeros(model.nvar)
    y[state_index(cosmology_model)["etak"]] = -0.05
    y[state_index(cosmology_model)["vb"]] = 0.3
    R, gpres, grho, adotoa = fn(y, 100.0, p)
    assert gpres == pytest.approx((grhog + grhor) / 3.0 - grhov)
    assert grho == pytest.approx(grhog + grhor + grhoc + grhob + grhov)
    dgq = grhob * 0.3
    assert R == pytest.approx(1.0 - adotoa * dgq / (0.05 * (grho + gpres)))


# --- The gravity interface: what the species read, and a theory replacing the one that provides it.


ROUND_COSMOLOGY = {
    "Omegam": 0.3,
    "Omegab": 0.05,
    "A_s": 2.1e-9,
    "n_s": 0.96,
    "H0": 67.0,
    "Tcmb": 2.7255,
    "YHe": 0.245,
    "Neff": 3.046,
}


GRHOM = GRHOM_PER_H0SQ * ROUND_COSMOLOGY["H0"] ** 2
"""The critical density of ``ROUND_COSMOLOGY``, which its densities sum to."""


def _densities(model, cosmology):
    return model_parameters(model, cosmology)


def test_a_model_needs_the_whole_interface():
    assert issubclass(gravity.Background, Gravity) and issubclass(Metric, Gravity)
    for removed in (Metric, gravity.Background):
        model = LCDM.remove(removed, name=f"no {removed.__name__}")
        with pytest.raises(TypeError, match="declares it with"):
            model.build(ConstantThermalHistory()).build(PARAM_COLUMNS, **HOST)


def test_curvature_is_geometry_and_not_a_fluid():
    """It reaches the expansion rate through ``Kcurv`` and none of the sums."""

    assert issubclass(Curvature, Gravity)
    curved = {**ROUND_COSMOLOGY, "Omegak": 0.05}
    adotoa, grho = on_the_host(LCDM, curved, "adotoa", "grho", a=1.0)
    # the fluids fall short of the critical density by the curvature's share
    assert float(grho / GRHOM) == pytest.approx(0.95, rel=1e-12)
    assert float(3.0 * adotoa**2 / GRHOM) == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("a", [1.0e-4, 1.0e-2, 0.5, 1.0])
def test_the_acceleration_equation_is_the_derivative_of_the_friedmann_equation(a):
    """``adotoa_prime = -(grho + 3 gpres) / 6`` for every species, curved or not."""

    import jax
    import jax.numpy as jnp

    cosmology = {
        **ROUND_COSMOLOGY,
        "Omegak": -0.03,
        "w_DE_0": -0.9,
        "w_DE_a": 0.2,
        "Nmnu": 1.0,
        "mnu": 0.3,
    }

    def adotoa(lna):
        return on_the_host(W0WACDM_MNU, cosmology, "adotoa", a=jnp.exp(lna))

    lna = jnp.log(a)
    rate, slope = jax.jvp(adotoa, (lna,), (jnp.ones(()),))
    prime = on_the_host(W0WACDM_MNU, cosmology, "adotoa_prime", a=a)
    # d/dtau = (a'/a) d/dln a
    assert float(prime) == pytest.approx(float(rate * slope), rel=1e-9)


class ScaledGravity(gravity.Background):
    """A Friedmann equation with a rescaled Newton constant: ``H^2 = mu 8 pi G rho / 3 - K``."""

    mu: ArrayLike = Parameter(1.0, column=True)

    def adotoa(self, grho, Kcurv, mu):
        return math.sqrt(mu * grho * (1.0 / 3.0) - Kcurv)


def test_another_friedmann_equation_replaces_the_background():
    """The species are untouched, and the budget closes with the new rate."""

    scaled = LCDM.swap(gravity.Background, ScaledGravity, name="scaled gravity")
    assert scaled.state_names() == LCDM.state_names()
    assert "mu" in scaled.parameter_keys() and "mu" not in LCDM.parameter_keys()
    plain = _densities(LCDM, ROUND_COSMOLOGY)
    d = _densities(scaled, {**ROUND_COSMOLOGY, "mu": 1.1})
    adotoa = on_the_host(scaled, {**ROUND_COSMOLOGY, "mu": 1.1}, "adotoa", a=1.0)
    # the model still expands at H0 today, with less dark energy to do it
    assert float(3.0 * adotoa**2 / GRHOM) == pytest.approx(1.0, rel=1e-12)
    assert float(d["grhov"]) < float(plain["grhov"])
    total = 1.1 * (d["grhov"] + d["grhoc"] + d["grhob"] + d["grhog"] + d["grhornomass"])
    assert float(total / GRHOM) == pytest.approx(1.0, rel=1e-12)
    # and the right-hand side builds with it
    from discoeb.solver._kernel import parameter_layout

    model = scaled.build(ConstantThermalHistory())
    model.build(parameter_layout(scaled).columns, **HOST)
    assert "ScaledGravity__adotoa(ScaledGravity_self, grho, Kcurv, mu)" in model.source


def test_a_rate_is_a_quantity_a_provider_may_read():
    """``d_<state>`` of a scalar state: the sum of what writes it."""

    class Tank(Component):
        level = State()

        def d_level(self, level):
            return -0.5 * level

    class Refill(Component):
        def d_level(self):
            return 2.0

    class Gauge(Component):
        trend = State()

        def falling(self, d_level):
            return -d_level

        def d_trend(self, falling):
            return falling

    model = Equations([Tank(), Refill(), Gauge()])
    rhs = model.build({}, **HOST)
    assert rhs((3.0, 0.0), 0.0, ()) == (0.5, -0.5)
    # computed once, as a quantity, and returned as the state's rate
    assert model.source.count("d_level = ") == 1
    assert model.evaluate("falling", level=3.0) == -0.5
    fn = model.build_function(("d_level", "falling"), inputs=("y",), **HOST)
    assert fn((3.0, 0.0)) == (0.5, -0.5)
    # the trend reads the level through its rate
    assert model.sparsity().tolist() == [[0, 0], [1, 0]]

    class Loop(Component):
        x = State()

        def d_x(self, d_x):
            return d_x

    with pytest.raises(ValueError, match="circular"):
        Equations([Loop()]).build({}, **HOST)


def test_the_metric_rate_is_a_quantity_of_the_model():
    model = LCDM.build(ConstantThermalHistory())
    rhs = model.build(PARAM_COLUMNS, **HOST)
    fn = model.build_function(
        ("d_etak", "adotoa_prime"), inputs=("y", "tau"), params=PARAM_COLUMNS, **HOST
    )
    p = np.zeros(N_PARAM)
    for name, value in dict(
        grhog=1.5e-7, grhornomass=1.0e-7, grhoc=0.8e-4, grhob=1.9e-5, grhov=2.0e-7
    ).items():
        p[PARAM_COLUMNS[f"{name}"]] = value
    p[PARAM_COLUMNS["k"]] = 0.05
    y = np.random.default_rng(1).standard_normal(model.nvar)
    d_etak, adotoa_prime = fn(tuple(y), 100.0, tuple(p))
    assert d_etak == rhs(tuple(y), 100.0, tuple(p))[model.index["etak"]]
    assert adotoa_prime < 0.0  # matter and radiation decelerate


def test_the_gauge_needs_its_species_at_rest():
    assert species.ColdDarkMatter.fixes_gauge and not Metric.fixes_gauge
    with pytest.raises(ValueError, match="fixes the synchronous gauge"):
        LCDM.remove(species.ColdDarkMatter, name="no CDM")
    # a copy under another name fixes it as well
    LCDM.swap(
        species.ColdDarkMatter, species.ColdDarkMatter.named("c"), name="renamed CDM"
    )


def test_the_streaming_approximation_is_a_component_a_model_may_drop():
    """No species names it, and no species is named by it."""

    full = LCDM.swap(RadiationStreaming, NoRadiationStreaming, name="full")
    assert full.state_names() == LCDM.state_names()
    assert "RSA" not in {n for cls in full.spec.components for n in cls.setting_names()}
    with pytest.raises(NameError, match="rsa"):
        LCDM.remove(RadiationStreaming, name="nothing").build(
            ConstantThermalHistory()
        ).build(PARAM_COLUMNS, **HOST)

    def rhs_of(cosmology_model):
        model = cosmology_model.build(ConstantThermalHistory(opacity=1.0e-6))
        return model, model.build(PARAM_COLUMNS, **HOST)

    p = np.zeros(N_PARAM)
    for name, value in dict(
        grhog=1.5e-7, grhornomass=1.0e-7, grhoc=0.8e-4, grhob=1.9e-5, grhov=2.0e-7
    ).items():
        p[PARAM_COLUMNS[f"{name}"]] = value
    p[PARAM_COLUMNS["k"]] = 0.5
    model, rhs = rhs_of(full)
    y = np.random.default_rng(2).standard_normal(model.nvar)
    tau = 5000.0  # deep inside the horizon and decoupled: the approximation is on
    without = np.asarray(rhs(tuple(y), tau, tuple(p)))
    _, off = rhs_of(LCDM.with_settings(RSA=0))
    _, on = rhs_of(LCDM)
    np.testing.assert_array_equal(without, np.asarray(off(tuple(y), tau, tuple(p))))
    assert not np.array_equal(without, np.asarray(on(tuple(y), tau, tuple(p))))

    # each streaming species adds its own density to the sum
    streaming = LCDM.build(ConstantThermalHistory()).build_function(
        ("dgrho_streaming", "tau"), inputs=("y", "tau"), params=PARAM_COLUMNS, **HOST
    )
    assert (
        "dgrho_streaming = Photons__dgrho_streaming_g(Photons_self, rsa, rsa_delta, grhog_t)"
        " + MasslessNeutrinos__dgrho_streaming_r(MasslessNeutrinos_self, rsa, rsa_delta, grhor_t)"
    ) in streaming._generated_source


# --- A species with a parameter of its own, added without touching the package.


def test_the_growing_mode_counts_the_new_species_as_matter():
    """Splitting the dark matter leaves the initial state alone: ``om`` sums ``grho_matter``."""

    from discoeb.solver._kernel import joint_model

    k = np.array([1.0e-2, 1.0e-1])
    plain = make_initial_states(
        cosmologies_of(LCDM, COSMOLOGY),
        k,
        joint_model(LCDM),
        cosmology_model=LCDM,
    )
    split = make_initial_states(
        cosmologies_of(LCDM_X, _split(COSMOLOGY)),
        k,
        joint_model(LCDM_X),
        cosmology_model=LCDM_X,
    )
    n = len(LCDM.state_names())
    np.testing.assert_allclose(split[:, :n], plain[:, :n], rtol=1e-12)
    # the neutrino quadrupole carries the O(om tau) term that reads the sum
    assert (plain[:, joint_model(LCDM).index["n_2"]] != 0.0).all()


# ============================================================================
# species: The species (``species``): their hierarchies, truncations, initial conditions and the named ``Fluid``
# ============================================================================

# --- One component listed several times, each copy under a name of its own.


def test_a_cold_pressureless_copy_moves_as_the_dark_matter():
    """``w = cs2 = 0`` at rest: ``x_delta' = -k z``, the dark matter's equation."""

    from discoeb.solver._kernel import parameter_layout

    columns = parameter_layout(LCDM_XY).columns
    model = LCDM_XY.build(ConstantThermalHistory())
    rhs = model.build(columns, **HOST)
    assert "x_grho_t = Fluid_x__grho_t(Fluid_x_self, x_grho0, a, x_w)" in [
        line.strip() for line in model.source.splitlines()
    ]
    p = np.zeros(len(columns))
    for name, value in dict(
        grhog=1.5e-7, grhornomass=1.0e-7, grhoc=0.8e-4, grhob=1.9e-5, grhov=2.0e-7
    ).items():
        p[columns[f"{name}"]] = value
    p[columns["x_grho0"]] = 0.3e-4
    p[columns["y_grho0"]] = 0.1e-4
    p[columns["y_w"]], p[columns["y_cs2"]] = 1.0 / 3.0, 1.0 / 3.0
    p[columns["k"]] = 0.05
    y = np.random.default_rng(0).standard_normal(model.nvar)
    y[model.index["x_theta"]] = 0.0
    dy = np.asarray(rhs(tuple(y), 100.0, tuple(p)))
    assert np.isfinite(dy).all()
    assert dy[model.index["x_delta"]] == dy[model.index["clxc"]]
    assert dy[model.index["x_theta"]] == 0.0
    # the radiation copy has pressure: its velocity feels its own contrast
    assert dy[model.index["y_theta"]] != 0.0
    pattern = {(int(r), int(c)) for r, c in model.sparsity()}
    assert (model.index["y_theta"], model.index["y_delta"]) in pattern
    assert (model.index["y_theta"], model.index["x_delta"]) not in pattern
    assert (model.index["etak"], model.index["x_delta"]) in pattern


def test_a_built_in_species_takes_a_name_too():
    """A second cold dark matter, with a density ``warm_omega_c`` of its own."""

    second = species.ColdDarkMatter.named("warm")
    parameters = dict(second.declared_parameters())
    assert parameters["warm_grhoc"].reads == ("warm_omega_c",)
    assert LCDM.add(second, name="two").state_names()[-1] == "warm_clxc"


# --- The Einstein-Boltzmann component model, on the host.


def test_a_hierarchy_needs_an_octupole():
    with pytest.raises(ValueError, match="octupole"):
        build_rhs(MODELS["flat-lcdm"].with_settings(LMAX_G=2))


def test_the_truncations_are_the_components_settings():
    """The model's lmax settings reach the components, which size their hierarchies."""
    cosmology_model = MODELS["ragged-blocks"]
    model, _ = build_rhs(cosmology_model)
    photons = next(c for c in model.components if isinstance(c, species.Photons))
    assert (photons.LMAX_G, photons.LMAX_POL) == (11, 13)
    assert (model.states["theta"].shape, model.states["e"].shape) == ((12,), (12,))
    neutrinos = next(c for c in model.components if isinstance(c, MassiveNeutrinos))
    assert (neutrinos.NQMAX, neutrinos.LMAX_NU) == (2, 9)
    assert model.states["psi"].shape == (2, 10) and len(neutrinos.weight) == 2
    # The defaults are the DISCO_LMAX_* environment settings, 15 unless set.
    assert MODELS["flat-lcdm"].setting("LMAX_G") == species.Photons.LMAX_G


def test_the_initial_state_and_sparsity_come_from_the_components():
    """The adiabatic mode: etak ~ -k, matter at 3/4 of the photon density, zero above the octupole."""
    model, _ = build_rhs(MODELS["ragged-blocks"])
    densities = (1.5e-7, 1.0e-7, 1.1e-4, 1.9e-5, 2.0e-7)
    k = np.array([1e-3, 1e-1])
    y0 = model.initial_state(
        k,
        tau=0.1,
        grhog=densities[0],
        grhornomass=densities[1],
        grhoc=densities[2],
        grhob=densities[3],
        grhov=densities[4],
        w_DE_0=-0.9,
        w_DE_a=0.1,
    )
    assert y0.shape == (2, model.nvar) and np.isfinite(y0).all()
    np.testing.assert_allclose(y0[:, model.index["etak"]], -k, rtol=1e-3)
    np.testing.assert_allclose(
        y0[:, model.index["clxc"]], 0.75 * y0[:, model.index["theta_0"]]
    )
    np.testing.assert_allclose(
        y0[:, model.index["clxq"]], (1.0 - 0.9) * y0[:, model.index["clxc"]]
    )
    assert (y0[:, model.index["theta_4"] : model.index["e_0"]] == 0.0).all()
    assert (y0[:, model.index["n_4"] : model.index["n_17"] + 1] == 0.0).all()
    neutrinos = next(c for c in model.components if isinstance(c, MassiveNeutrinos))
    np.testing.assert_allclose(
        y0[:, model.index["psi_1_0"]],
        -0.25 * y0[:, model.index["n_0"]] * neutrinos.dlnf0[1],
    )
    # The sparsity pattern is traced through the same components: the metric
    # reads the dark-energy momentum through dgq, a free-streaming neutrino
    # multipole reads only its neighbours.
    pattern = {(int(r), int(c)) for r, c in model.sparsity()}
    i = model.index
    assert (i["etak"], i["thetaq"]) in pattern and (i["n_5"], i["n_4"]) in pattern
    assert (i["n_5"], i["etak"]) not in pattern and (i["n_5"], i["n_5"]) not in pattern


# ============================================================================
# interactions: Interactions (``interactions``): Thomson scattering
# ============================================================================

# --- The Einstein-Boltzmann component model, on the host.


def test_thomson_scattering_writes_the_photon_and_baryon_derivatives():
    model, _ = build_rhs(MODELS["flat-lcdm"])
    lines = [ell.strip() for ell in model.source.splitlines()]
    assert (
        "d_vb = Baryons__d_vb(Baryons_self, adotoa, vb, k, cs2, clxb) + ThomsonScattering__d_vb_thomson(ThomsonScattering_self, photbar, opacity, slip)"
        in lines
    )
    assert lines.index(
        "Photons__d_theta(Photons_self, k, z, sigma, theta, inv_tau, rsa, c_Photons__d_theta)"
    ) < lines.index(
        "ThomsonScattering__d_theta_thomson(ThomsonScattering_self, opacity, slip, polter, theta, rsa, c_ThomsonScattering__d_theta_thomson)"
    )
    assert (
        "ThomsonScattering__d_e_thomson(ThomsonScattering_self, opacity, polter, e, rsa, c_ThomsonScattering__d_e_thomson)"
        in lines
    )
    # Neutrinos are collisionless: one writer.
    assert sum("__d_n(" in ell for ell in lines) == 1


# ============================================================================
# recombination: Recombination and reionization (``recombination``), read from the joint kernel, against CLASS
# ============================================================================

# --- The recombination history of the perturbation kernel against CLASS, and the time of its solve.


THERMAL_KEEP = ("tau", "xe", "Tmat", "cs2", "opacity", "kappa")


THERMAL_K = np.array([1.0e-3])
"""One wave mode per cosmology: the recombination block does not read the hierarchy."""


N_A = 2000


Z_MIN, Z_MAX = 10.0, 5000.0


XE_GATE = 0.01


TM_GATE = 0.002


CS2_GATE = 0.01


SWITCH_WINDOW = (800.0, 900.0)


CS2_SWITCH_GATE = 0.2


def kernel_thermal_history(cosmologies, model=None, keep=THERMAL_KEEP, n_a: int = N_A):
    """``(a, history)`` of the kernel's recombination block on a logarithmic grid of the scale factor.

    ``a`` runs from :data:`discoeb.A_START` to 1, today for every cosmology.
    ``history[name]`` is ``(n_cosmologies, n_a)`` for each name of ``keep``.
    """

    from discoeb import A_START, LCDM, solve_eb

    batch = cosmologies_of(LCDM if model is None else model, cosmologies)
    a = np.geomspace(A_START, 1.0, n_a)
    values = solve_eb(batch, THERMAL_K, a, keep=keep)
    kept = np.asarray(values)[:, :, 0, :]  # one wave mode
    return a, {name: kept[:, i] for i, name in enumerate(keep)}


def _class_thermodynamics(param):
    """Return ``(z, x_e, T_b, c_b^2)`` from CLASS with recfast and no reionization, ascending in z."""

    Class = pytest.importorskip("classy").Class
    cosmo = Class()
    cosmo.set(
        {
            **_class_base(param),
            "recombination": "recfast",
            "reio_parametrization": "reio_none",
        }
    )
    try:
        cosmo.compute()
        thermo = cosmo.get_thermodynamics()
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()
    order = np.argsort(thermo["z"])
    return tuple(
        np.asarray(thermo[key])[order] for key in ("z", "x_e", "Tb [K]", "c_b^2")
    )


@pytest.mark.gpu
@pytest.mark.parametrize("n_cosmologies", BATCH_SIZE_CASES)
def test_recfast_matches_class(n_cosmologies, benchmark):
    """Check the kernel's recombination history against CLASS and time one warmed batch solve."""

    cosmologies = _perturbed_cosmologies(n_cosmologies)
    a, history = benchmark.pedantic(
        kernel_thermal_history,
        args=(cosmologies,),
        rounds=1,
        warmup_rounds=1,
        iterations=1,
    )

    def worst(ours, theirs, z, mask):
        rel = np.abs(ours[mask] / theirs[mask] - 1.0)
        i = np.argmax(rel)
        return float(rel[i]), float(z[mask][i])

    for i, param in enumerate(cosmologies):
        z = 1.0 / a - 1.0
        inside = (z >= Z_MIN) & (z <= Z_MAX)
        switch = (z > SWITCH_WINDOW[0]) & (z < SWITCH_WINDOW[1])
        z_class, xe_class, tb_class, cb2_class = _class_thermodynamics(param)
        xe_ref = np.interp(z, z_class, xe_class)
        tm_ref = np.interp(z, z_class, tb_class)
        cs2_ref = np.interp(z, z_class, cb2_class)
        assert np.isfinite(history["xe"][i]).all()

        err, at = worst(history["xe"][i], xe_ref, z, inside)
        assert err < XE_GATE, f"cosmology {i}: x_e off by {err:.3g} at z={at:.0f}"
        err, at = worst(history["Tmat"][i], tm_ref, z, inside)
        assert err < TM_GATE, f"cosmology {i}: T_m off by {err:.3g} at z={at:.0f}"
        err, at = worst(history["cs2"][i], cs2_ref, z, inside & ~switch)
        assert err < CS2_GATE, f"cosmology {i}: c_b^2 off by {err:.3g} at z={at:.0f}"
        err, at = worst(history["cs2"][i], cs2_ref, z, inside & switch)
        assert err < CS2_SWITCH_GATE, (
            f"cosmology {i}: c_b^2 off by {err:.3g} at z={at:.0f} in the tight-coupling switch window"
        )


# --- The conformal time, a state of the block.


def test_the_conformal_time_starts_from_the_radiation_eras_closed_form():
    """``tau`` at the start is the integral of ``da / (a^2 H)`` from zero, with no integration."""

    from helpers import on_the_host

    from discoeb import A_START, LCDM, LCDM_MNU
    from discoeb.components.recombination import _tau_initial

    for model, more in ((LCDM, {}), (LCDM_MNU, {"Nmnu": 1.0, "mnu": 0.3})):
        name = model.__name__
        cosmology = {**REFERENCE_COSMOLOGY, **more}
        rate, matter = on_the_host(model, cosmology, "adotoa", "grho_matter", a=A_START)
        closed = _tau_initial(float(rate), float(matter))
        # in a, where the integrand 1 / (a^2 H) is nearly constant; below
        # the grid the universe is radiation alone and tau = 1 / adotoa
        grid = np.linspace(1.0e-3 * A_START, A_START, 4000)
        rates = np.asarray(on_the_host(model, cosmology, "adotoa", a=grid))
        integral = np.trapezoid(1.0 / (grid * rates), grid) + 1.0 / rates[0]
        assert closed == pytest.approx(integral, rel=1e-6), name
        assert 0.05 < closed < 0.2


# --- Reionization in the thermal history.


Z_RE = 7.67
"""Planck 2018's midpoint of CAMB's step, for ``tau_reio`` near 0.054."""


LATE = 1.0 / 31.0
"""Scale factors past this one (``z < 30``) hold the reionization step; the
recombination tail adds ~1e-4 to their optical depth."""


def _class_tau_reio(z_re):
    """CLASS's optical depth of CAMB's step at ``z_re``."""

    Class = pytest.importorskip("classy").Class
    c = Class()
    c.set(
        {
            **_class_base(REFERENCE_COSMOLOGY),
            "recombination": "recfast",
            "reio_parametrization": "reio_camb",
            "z_reio": z_re,
        }
    )
    try:
        c.compute()
        return c.get_current_derived_parameters(["tau_reio"])["tau_reio"]
    finally:
        c.struct_cleanup()
        c.empty()


def _late_history(z_re):
    """``(a, visibility, late optical depth, tau_reio)`` after :data:`LATE`, from the kernel.

    Of LCDM with :class:`~discoeb.components.recombination.ReionizationDepth`,
    whose state ``kappa_reio`` is ``tau_reio`` today.
    """

    a, history = kernel_thermal_history(
        [{**REFERENCE_COSMOLOGY, "z_re": z_re}],
        model=LCDM_REIO,
        keep=(*THERMAL_KEEP, "kappa_reio"),
    )
    late = a > LATE
    kappa = history["kappa"][0]
    visibility = history["opacity"][0] * np.exp(-(kappa[-1] - kappa))
    return (
        a[late],
        visibility[late],
        float(kappa[-1] - kappa[late][0]),
        float(history["kappa_reio"][0][-1]),
    )


@pytest.mark.gpu
def test_reionization_optical_depth():
    """``kappa_reio`` today is CLASS's tau_reio at the same z_re, and the bump sits at z_re."""

    a, g, tau_late, tau_reio = _late_history(Z_RE)
    tau_class = _class_tau_reio(Z_RE)
    assert abs(tau_reio - tau_class) < 1.0e-3, (tau_reio, tau_class)
    # the late optical depth is reionization's plus the recombination tail
    assert 0.0 < tau_late - tau_reio < 5.0e-4, (tau_late, tau_reio)
    z_bump = 1.0 / a[np.argmax(g)] - 1.0
    # the visibility peaks a little below z_re (the step is half way there)
    assert abs(z_bump - Z_RE) < 1.5, z_bump


@pytest.mark.gpu
def test_no_reionization_by_default():
    """z_re = 0 (the default) adds no optical depth after recombination."""

    _, _, tau_late, tau_reio = _late_history(0.0)
    assert tau_late < 5.0e-4 and tau_reio == 0.0


# ============================================================================
# util: The neutrino momentum bins and their Gauss-Laguerre rule (``_util``)
# ============================================================================


@pytest.mark.parametrize("n", [6, 10])
def test_the_rule_is_numpys_for_alpha_zero(n):
    nodes, weights = generalized_gauss_laguerre_weights(n, 0.0)
    expected_nodes, expected_weights = np.polynomial.laguerre.laggauss(n)
    np.testing.assert_allclose(nodes, expected_nodes, rtol=1e-10)
    np.testing.assert_allclose(weights, expected_weights, rtol=1e-8)


def test_the_rule_integrates_polynomials_against_the_weight():
    """An ``n``-point rule is exact to degree ``2n - 1``: the integrals are gamma functions."""
    from math import gamma

    nodes, weights = generalized_gauss_laguerre_weights(6, 1.0)
    for degree in (0, 3, 11):
        integral = float(np.sum(np.asarray(weights) * np.asarray(nodes) ** degree))
        assert integral == pytest.approx(gamma(degree + 2.0), rel=1e-9)


@pytest.mark.parametrize("nqmax", [3, 4, 5, 8])
def test_one_massless_flavour_integrates_to_unit_density(nqmax):
    q, w, dlnf0 = neutrino_momentum_bins(nqmax)
    assert q.shape == w.shape == dlnf0.shape == (nqmax,)
    assert np.sum(w) == pytest.approx(1.0, rel=5e-3)
    np.testing.assert_allclose(dlnf0, -q / (1.0 + np.exp(-q)))
    assert not q.flags.writeable
    assert neutrino_momentum_bins(nqmax)[0] is q  # cached
