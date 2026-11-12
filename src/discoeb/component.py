"""A right-hand side assembled from components whose functions name what they read.

A model is a list of *components*. A component declares its settings and state
variables as class attributes and provides *quantities* through ordinary Python
functions whose **parameter names are the names of the quantities they depend
on**::

    class Baryons(Species):
        clxb = State(initial=lambda clxg: 0.75 * clxg)
        vb = State()

        def grhob_t(self, grhob, inv_a):
            return grhob * inv_a

        def dgrho(self, grhob_t, clxb):
            return grhob_t * clxb

        def d_vb(self, adotoa, vb, k, cs2, clxb):
            return -adotoa * vb + k * cs2 * clxb

    class MasslessNeutrinos(Species):
        LMAX_NR = Setting(15)                               # DISCO_LMAX_NR overrides
        n = State(lambda self: self.LMAX_NR + 1)   # N_0 .. N_LMAX_NR

        def d_n(self, k, z, sigma, n, inv_tau, output):
            output[0] = -k * ((4.0 / 3.0) * z + n[1])
            ...
            for l in range(3, self.LMAX_NR):
                output[l] = k * (l * n[l - 1] - (l + 1) * n[l + 1]) * (1.0 / (2 * l + 1))
            ...

Nothing gets passed explicitly. :meth:`Model.build` reads each provider's
dependencies off its signature (``inspect.signature``), orders the providers
with ``graphlib.TopologicalSorter`` and generates a driver that calls them in
that order, injecting each value by name. This happens once, at build time.
State variables (``y[i]``), parameters (``p[j]``) and the time are the base
quantities. Everything else is derived.

**Settings.** ``LMAX_G = Setting(15)`` declares a setting with its default.
The class loads it at creation from the environment variable ``DISCO_<NAME>``
when set, cast to the default's type. It reads as a plain value on the class
and its instances, and the constructor takes an override as a keyword
(``Photons(LMAX_G=11)``). A provider reads it as ``self.LMAX_G``.

**States.** ``theta = State(shape)`` declares a state variable: a scalar
(``State()``), a vector (``State(16)``) or a matrix (``State((3, 16))``). The
shape may be a function of the component instance when it depends on its
settings. The flat state vector holds the components' states in declaration
order, a vector ``theta`` as the elements ``theta_0, theta_1, ...`` and a
matrix ``psi`` as ``psi_0_0, psi_0_1, ...``.

A state also carries its *initial condition*, ``State(..., initial=fn)``: a
lambda for a scalar, or a ``_``-prefixed function defined just above for a
vector. Its parameters resolve like a provider's, from the quantities
:meth:`Model.initial_state` gets (``k``, ``tau``, the background densities),
from other states' initial values, and from providers the host can evaluate
(``clxg`` for ``theta[0]``). :meth:`Model.evaluate` does the same resolution
for any quantity, on floats or JAX arrays. A scalar's initial condition
returns the value. A vector's takes ``output``, zero-filled, and writes into
it. A state without one starts at zero.

**Sparsity.** Nothing declares which states a derivative depends on.
:meth:`Model.sparsity` reads the Jacobian's pattern off the providers. It
resolves the same dependency graph as a build, but on values that carry the
set of state elements they depend on instead of numbers. A state element
carries its own index, the time and parameters carry nothing, and every
operation carries the union of its operands'. So
``output[l] = kv * (l * x[l - 1] - (l + 1) * x[l + 1])`` traces as the
tridiagonal band it is. The pattern is exact. A data-dependent branch, the
one construct that could hide a dependency, makes the provider count as
reading all of its inputs.

**Providers.** Every method of a component is a provider, named after the
quantity it provides, and so is a function stored on the instance in
``__init__``. ``self`` is the component. A provider reads its settings and
other public numeric attributes (a tuple of quadrature weights the
constructor derived, say) as ``self.NQMAX`` or ``self.weight[i]``. The
remaining parameters are the quantities it depends on. A method or lambda
that reads nothing of its component may leave ``self`` out.

Three kinds of name are special. ``d_<state>`` provides a contribution to
that state's derivative. For a scalar state it returns the contribution. For
a vector state it takes ``output``, which the framework fills with zeros, and
writes its contribution into it. A function named in the model's ``sums``
(``grho``, ``dgrho`` and ``dgq`` for the Einstein-Boltzmann system)
contributes to that sum over components. Names starting with an underscore
are private to the component.

Several components may write one state (the baryon Euler equation and the
Thomson drag both write ``vb``), and the derivative is the sum of their
contributions. A vector state reaches a provider as a 1-D array (a matrix as
2-D), so ``len()``, indexing and loops over multipoles work as in plain
Python. A provider may call a plain module-level helper (``stream``), which
the build compiles for the device alongside it. Wherever a function fills an
array instead of returning, that array is its parameter ``output``.

On the device ``self`` is a named tuple of the component's constants
(:meth:`Component.device_self`). The generated driver passes it to each
provider as a compile-time constant, so ``self.NQMAX`` folds into the loop
bound and ``self.weight[i]`` into the unrolled sum just like a literal. On the
host (:meth:`Model.evaluate`) ``self`` is the component itself.

**Device names.** A provider that needs scratch space may use ``cuda`` and
``float64`` imported from this module (``cuda.local.array(16, float64)``).
The size must be a literal, since numba types it before the optimiser folds
``self.N``. Both are placeholders. Compiling a provider swaps in the real
objects, and a host test passes stand-ins to :meth:`Model.build`. So a model
imports nothing from CUDA and runs as plain Python under test.

Each provider compiles to its own ``cuda.jit(device=True)`` function, and the
generated driver is one more. The driver copies ``y`` into a local array per
state, allocates the contribution buffers, calls the providers and sums the
contributions into local arrays it returns as a tuple. The primal thus has
the fixed-size tuple signature the Enzyme-differentiated numba-CUDA kernel
needs, while the components see arrays. numba-enzyme optimises the primal
with LLVM's ``O3`` before Enzyme differentiates it. That pass inlines the
providers, unrolls the loops and promotes every local array to scalars, so
Enzyme and the kernel see a straight-line right-hand side, as if written by
hand. The driver keeps its text as ``_generated_source`` and registers it
with ``linecache`` so numba can read it.
"""

from __future__ import annotations

import graphlib
import inspect
import itertools
import linecache
import math
import os
import types
from collections import namedtuple
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Generic, Protocol, TypeVar

import numpy as np

SETTINGS_PREFIX = "DISCO_"
"""Environment variables ``DISCO_<NAME>`` override a component's settings."""

OUTPUT = "output"
"""The parameter through which a function fills an array instead of returning."""


# --- device names ---------------------------------------------------------------------


class _Ambient:
    """A device name the build step supplies to providers (see the module docstring)."""

    def __init__(self, name: str):
        self.name = name

    def _unavailable(self, *args, **kwargs):
        raise RuntimeError(
            f"{self.name} is only available inside a provider compiled by Model.build"
        )

    __call__ = __getattr__ = _unavailable

    def __repr__(self):
        return f"<device name {self.name}>"


cuda = _Ambient("cuda")
"""``numba_cuda_mlir.cuda`` inside a compiled provider: ``cuda.local.array(N, float64)``."""

float64 = _Ambient("float64")
"""``numba_cuda_mlir.types.float64`` inside a compiled provider."""

float32 = _Ambient("float32")
"""``numba_cuda_mlir.types.float32`` inside a compiled provider: ``float32(x)`` rounds
``x`` to single precision, so a transcendental of it runs at single rate."""


def _device_names():
    from numba_cuda_mlir import cuda as cuda_module
    from numba_cuda_mlir import types as cuda_types

    return {
        "cuda": cuda_module,
        "float64": cuda_types.float64,
        "float32": cuda_types.float32,
    }


_HOST_NAMES = {
    "cuda": types.SimpleNamespace(
        local=types.SimpleNamespace(array=lambda shape, dtype: np.zeros(shape))
    ),
    "float64": np.float64,
    "float32": np.float32,
}
"""What the device names mean when a provider runs as plain Python on the host."""


def _host_names(math_module) -> dict:
    """The host names for providers evaluated with ``math_module`` as ``math``.

    On arrays (``numpy``, ``jax.numpy``) a provider's ``max`` and ``min``
    become the module's elementwise ``maximum`` and ``minimum``, so the
    recombination block's clamps broadcast like the rest of its algebra.
    """

    names = {**_HOST_NAMES, "math": math_module}
    if hasattr(math_module, "maximum"):
        names.update(
            max=math_module.maximum,
            min=math_module.minimum,
            float32=math_module.float32,
            float64=math_module.float64,
        )
    return names


# --- settings -------------------------------------------------------------------------


def load_setting(name: str, default):
    """``default``, or the environment's ``DISCO_<name>`` cast to the default's type."""

    text = os.environ.get(SETTINGS_PREFIX + name)
    if text is None:
        return default
    if isinstance(default, bool):
        lowered = text.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"{SETTINGS_PREFIX}{name}={text!r} is not a boolean")
    try:
        return type(default)(text)
    except ValueError as e:
        raise ValueError(
            f"{SETTINGS_PREFIX}{name}={text!r} is not a {type(default).__name__}"
        ) from e


T = TypeVar("T", bool, int, float, str)


class Setting(Generic[T]):
    """A component setting, ``LMAX_G = Setting(15)``, loaded from ``DISCO_LMAX_G``.

    On the class it reads as the loaded value, on an instance as the
    constructor's override or the loaded value. The default's type types the
    descriptor, so a type checker sees ``self.LMAX_G`` as an ``int``.
    """

    def __init__(self, default: T):
        if not isinstance(default, (bool, int, float, str)):
            raise TypeError(
                f"a Setting needs a bool, int, float or str default, not {default!r}"
            )
        self.default = default
        self.name = None
        self.value = default

    def __set_name__(self, owner, name):
        if not (name.isidentifier() and name.isupper()):
            raise ValueError(
                f"{owner.__name__}: setting {name!r} must be an UPPER_CASE identifier"
            )
        self.name = name
        self.value = load_setting(name, self.default)

    def __get__(self, instance, owner=None) -> T:
        if instance is None:
            return self.value
        return instance.__dict__.get(self.name, self.value)

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value

    def __repr__(self):
        return f"Setting({self.name} = {self.value!r}, default {self.default!r})"


def _is_constant(value) -> bool:
    if isinstance(value, (bool, int, float)):
        return True
    return (
        isinstance(value, tuple)
        and bool(value)
        and all(isinstance(v, (bool, int, float)) for v in value)
    )


# --- states ---------------------------------------------------------------------------


class State:
    """A state variable declared on a component: ``etak = State()``, ``theta = State(16)``.

    ``shape`` is ``()`` for a scalar, ``n`` for a vector, ``(n, m)`` for a
    matrix, or a function of the component instance returning one of those.
    ``initial`` is the initial condition (see the module docstring).
    """

    def __init__(self, shape=(), initial: Callable | None = None):
        self.shape = shape
        self.initial_fn = initial
        if initial is not None and not callable(initial):
            raise TypeError(
                "initial must be a function of the quantities the initial condition reads"
            )

    def resolve(self, component) -> tuple[int, ...]:
        spec: Any = self.shape(component) if callable(self.shape) else self.shape
        if spec is None or spec == ():
            return ()
        if isinstance(spec, int):
            return (spec,)
        shape = tuple(int(n) for n in spec)
        if not shape or any(n <= 0 for n in shape) or len(shape) > 2:
            raise ValueError(
                f"a state is a scalar, a vector or a matrix, not of shape {spec}"
            )
        return shape


# --- components -----------------------------------------------------------------------


class Component:
    """Provides quantities and may own settings and state variables (see the module docstring)."""

    _setting_names: tuple[str, ...] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        own = [attr for attr, value in vars(cls).items() if isinstance(value, Setting)]
        inherited = [n for n in cls._setting_names if n not in own]
        cls._setting_names = tuple(inherited + own)

    def __init__(self, **overrides):
        names = self.setting_names()
        unknown = sorted(set(overrides) - set(names))
        if unknown:
            raise TypeError(
                f"{self.label} has no settings {unknown}; it has {sorted(names)}"
            )
        for name, value in overrides.items():
            setattr(self, name, value)

    @classmethod
    def setting_names(cls) -> tuple[str, ...]:
        return cls._setting_names

    @classmethod
    def declared_states(cls) -> list[tuple[str, State]]:
        """The ``State`` attributes, in declaration order, base classes first."""

        found, seen = [], set()
        for klass in reversed(cls.__mro__):
            for attr, value in vars(klass).items():
                if isinstance(value, State) and attr not in seen:
                    seen.add(attr)
                    found.append((attr, value))
        return found

    def constants(self) -> dict[str, object]:
        """The settings and every other public numeric attribute: what ``self`` holds on the device."""

        found = {name: getattr(self, name) for name in self.setting_names()}
        for name, value in vars(self).items():
            if not name.startswith("_") and name not in found and _is_constant(value):
                found[name] = value
        return found

    def device_self(self):
        """``self`` as the device sees it: a named tuple of :meth:`constants`.

        numba types a named tuple's fields individually, so an ``int`` setting,
        a ``float`` and a tuple of floats each keep their type, and a global
        instance is a compile-time constant the optimiser folds.
        """

        constants = self.constants()
        key = (self.label, tuple(constants))
        if key not in _SELF_TYPES:
            _SELF_TYPES[key] = namedtuple(self.label, constants)
        return _SELF_TYPES[key](**constants)

    @property
    def label(self) -> str:
        return type(self).__name__


class Species(Component):
    """A component with a density: it contributes to the model's sums."""


class Interaction(Component):
    """A component without state, writing the derivatives of species' state."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        owned = [attr for attr, value in vars(cls).items() if isinstance(value, State)]
        if owned:
            raise TypeError(
                f"{cls.__name__} is an Interaction and cannot own state {owned}"
            )


_FRAMEWORK_CLASSES = (Component, Species, Interaction, object)
_SELF_TYPES: dict[tuple, type] = {}


# --- quantities -----------------------------------------------------------------------


def _elements(name: str, shape: tuple[int, ...]) -> list[str]:
    """The flat names of a state's elements, row-major: ``psi_0_0, psi_0_1, ...``."""

    if not shape:
        return [name]
    return [
        "_".join((name, *map(str, index)))
        for index in itertools.product(*(range(n) for n in shape))
    ]


class _State:
    def __init__(self, name: str, declared: State, component: Component, offset: int):
        self.name, self.declared, self.component, self.offset = (
            name,
            declared,
            component,
            offset,
        )
        self.shape = declared.resolve(component)
        self.elements = _elements(name, self.shape)
        self.size = len(self.elements)


# --- providers ------------------------------------------------------------------------


def _positional_names(fn: Callable, label: str) -> tuple[bool, list[str]]:
    """``(takes self, the names after it)`` of a provider or initial condition."""

    params = list(inspect.signature(fn).parameters.values())
    for p in params:
        if (
            p.kind not in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            or p.default is not p.empty
        ):
            raise TypeError(f"{label}: functions take plain positional parameters only")
    takes_self = bool(params) and params[0].name == "self"
    return takes_self, [p.name for p in params[takes_self:]]


def _check_output(label: str, what: str, shape: tuple, names) -> None:
    """A vector's function fills ``output``, a scalar's returns its value."""

    if shape and OUTPUT not in names:
        raise TypeError(f"{label}: {what} of a vector takes {OUTPUT!r} and fills it")
    if not shape and OUTPUT in names:
        raise TypeError(
            f"{label}: {what} of a scalar returns the value instead of taking {OUTPUT!r}"
        )


class _Provider:
    """One function of a component: what it reads, what it provides."""

    def __init__(
        self,
        component: Component,
        attr: str,
        fn: Callable,
        states: Mapping[str, _State],
    ):
        self.component, self.attr, self.fn = component, attr, fn
        self.label = f"{component.label}.{attr}"
        self.symbol = f"{component.label}__{attr}"
        self.self_symbol = f"{component.label}_self"
        self.takes_self, names = _positional_names(fn, self.label)
        self.written = None
        if attr.startswith("d_"):
            state = states.get(attr[2:])
            if state is None:
                raise ValueError(
                    f"{self.label} writes {attr[2:]!r}, which no component owns"
                )
            self.written = state
            _check_output(self.label, "the derivative", state.shape, names)
        elif OUTPUT in names:
            raise TypeError(
                f"{self.label}: only a vector state's derivative fills an {OUTPUT!r}"
            )
        self.params = tuple(n for n in names if n != OUTPUT)
        self.signature = tuple(names)


def _providers(component: Component, states: Mapping[str, _State]) -> list[_Provider]:
    found, seen = [], set()
    for klass in type(component).__mro__:
        if klass in _FRAMEWORK_CLASSES:
            continue  # the framework's own methods are not providers
        for attr, member in vars(klass).items():
            if attr in seen or attr.startswith("_"):
                continue
            seen.add(attr)
            # Every method not named with a leading underscore is a provider, a
            # staticmethod one without self.
            if isinstance(member, staticmethod):
                found.append(_Provider(component, attr, member.__func__, states))
            elif isinstance(member, types.FunctionType):
                found.append(_Provider(component, attr, member, states))
    for attr, value in vars(component).items():
        if (
            attr not in seen
            and not attr.startswith("_")
            and isinstance(value, types.FunctionType)
        ):
            seen.add(attr)
            found.append(_Provider(component, attr, value, states))
    return found


def _bound_function(
    fn: types.FunctionType, jit: Callable, names: Mapping[str, object], memo: dict
) -> Callable:
    """``fn`` compiled by ``jit`` with its globals resolved.

    numba resolves globals at compile time, so this rebinds ``fn`` to a copy
    of its globals where the device-name placeholders are the real objects
    (``names``), ``math`` is ``names["math"]`` (``math`` itself on the device,
    ``jax.numpy`` on JAX arrays), ``max`` and ``min`` are the array module's
    elementwise ones when ``names`` has them, and the plain helpers it calls
    get the same treatment, recursively. With an identity ``jit`` this is how
    a function runs on the host.
    """

    if fn in memo:
        return memo[fn]
    g = fn.__globals__
    rebound = dict(g)
    for name in fn.__code__.co_names:
        value = g.get(name)
        if isinstance(value, _Ambient):
            rebound[name] = names[value.name]
        elif value is math:
            rebound[name] = names.get("math", math)
        elif isinstance(value, types.FunctionType) and value.__module__ != "builtins":
            rebound[name] = _bound_function(value, jit, names, memo)
        elif value is None and name in names:
            # a builtin the names override, such as max and min on arrays
            rebound[name] = names[name]
    fn_bound = types.FunctionType(
        fn.__code__, rebound, fn.__name__, fn.__defaults__, fn.__closure__
    )
    compiled = jit(fn_bound)
    memo[fn] = compiled
    return compiled


# --- model ----------------------------------------------------------------------------

_SOURCES = itertools.count(1)


class _HostValues:
    """The value of every quantity, resolved on demand on the host.

    A name is a base value, a state (its initial value: zeros, or its
    ``initial`` filler), or a provider evaluated from the same pool. The
    providers run as plain Python with ``names`` in place of the device
    names, so the values may be floats or JAX arrays.
    """

    def __init__(self, model, base, names, memo=None, batch_shape=()):
        self.model, self.values, self.names = model, dict(base), names
        self.memo = {} if memo is None else memo
        self.providers, self.contributions = model._host_providers()
        self.resolving: list[str] = []
        # trailing axes of every state's value (the wave modes of a vectorised
        # initial state), so a vector state's filler writes rows of them
        self.batch_shape = tuple(batch_shape)

    def _call(self, fn, component, takes_self, args):
        bound = _bound_function(fn, lambda f: f, self.names, self.memo)
        return bound(component, *args) if takes_self else bound(*args)

    def __getitem__(self, name):
        if name in self.values:
            return self.values[name]
        if name in self.resolving:
            raise ValueError(
                "circular dependency on the host: "
                + " -> ".join([*self.resolving, name])
            )
        self.resolving.append(name)
        if name in self.model.states:
            s = self.model.states[name]
            if s.shape or self.batch_shape:
                result = np.zeros(s.shape + self.batch_shape)
            else:
                result = 0.0
            fn = s.declared.initial_fn
            if fn is not None:
                takes_self, params = _positional_names(fn, f"{name}.initial")
                _check_output(
                    f"{name}.initial", "the initial condition", s.shape, params
                )
                args = [result if p == OUTPUT else self[p] for p in params]
                filled = self._call(fn, s.component, takes_self, args)
                if not s.shape:
                    result = filled
        elif name in self.model.sums:
            parts = [
                self._call(p.fn, p.component, p.takes_self, [self[d] for d in p.params])
                for p in self.contributions.get(name, ())
            ]
            if not parts:
                raise NameError(f"nothing contributes to the sum {name!r}")
            result = parts[0]
            for part in parts[1:]:
                result = result + part
        elif name in self.providers:
            p = self.providers[name]
            result = self._call(
                p.fn, p.component, p.takes_self, [self[d] for d in p.params]
            )
        else:
            raise NameError(f"{name!r} is neither given nor provided by any component")
        self.resolving.pop()
        self.values[name] = result
        return result


# --- sparsity -------------------------------------------------------------------------


class _Traced:
    """A stand-in for a number while tracing the providers for the Jacobian pattern.

    It carries ``on``, the set of state elements (flat indices) it depends on.
    A state element carries its own index, an input nothing, and every
    operation on traced values returns one carrying the union of its
    operands'. A derivative's value thus carries exactly the columns of its
    row. Indexing is a no-op: a state-free provider does not run, and the one
    value standing in for its result may be indexed like the array it would
    have returned. A branch on a traced value and a conversion to a number are
    the only ways to lose a dependency, so both raise, even on an input, since
    the two branches might read different states.
    :meth:`_TracedValues._call` then takes the provider to read every input
    for everything it writes.
    """

    __slots__ = ("on",)

    def __init__(self, on: frozenset[int] = frozenset()):
        self.on = on

    def _join(self, other):
        if isinstance(other, _Traced):
            return _Traced(self.on | other.on)
        if isinstance(other, np.ndarray):
            return NotImplemented  # numpy applies the operation element by element
        return self  # a number, a bool: nothing to add

    def _same(self, *_):
        return self

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = _join
    __truediv__ = __rtruediv__ = __floordiv__ = __rfloordiv__ = __mod__ = __rmod__ = (
        _join
    )
    __pow__ = __rpow__ = _join
    __lt__ = __le__ = __gt__ = __ge__ = __eq__ = __ne__ = _join
    __neg__ = __pos__ = __abs__ = __getitem__ = __round__ = _same
    __hash__ = object.__hash__

    def __bool__(self):
        raise TypeError(
            "a provider branches on a traced value (an if, a comparison, max or min), "
            "which the sparsity tracing cannot follow: write it branch-free, or compute "
            "the branching quantity in a provider of inputs alone, which is not traced"
        )

    def __float__(self):
        raise TypeError(
            "a provider converts a state-dependent value to a number; it must stay an expression"
        )

    __int__ = __index__ = __complex__ = __float__

    def __repr__(self):
        return f"_Traced({sorted(self.on)})"


def _traced_call(*args):
    """A function of traced values: the union of its arguments' dependencies."""

    return _Traced(frozenset().union(*(_depends(a) for a in args)))


class _TracingMath:
    """``math`` for traced values: every function returns the union of its arguments' dependencies."""

    def __getattr__(self, name):
        attr = getattr(math, name)
        return (
            # constants such as pi, e and inf stay numbers
            attr if not callable(attr) else _traced_call
        )


def _traced_array(shape, first: int | None = None) -> np.ndarray:
    """An object array of traced values: state-free, or the elements ``first, first + 1, ...``."""

    arr = np.empty(shape, dtype=object)
    for n, index in enumerate(np.ndindex(*arr.shape)):
        arr[index] = _Traced() if first is None else _Traced(frozenset({first + n}))
    return arr


def _depends(value) -> frozenset[int]:
    """The state elements a traced value, or an array of them, depends on."""

    if isinstance(value, _Traced):
        return value.on
    if isinstance(value, np.ndarray) and value.dtype == object:
        return frozenset().union(*(v.on for v in value.flat if isinstance(v, _Traced)))
    return frozenset()


_TRACING_NAMES = {
    "cuda": types.SimpleNamespace(
        local=types.SimpleNamespace(array=lambda shape, dtype: _traced_array(shape))
    ),
    "float64": lambda x: x,  # a cast carries its operand's dependencies
    "float32": lambda x: x,
    "math": _TracingMath(),
    "max": _traced_call,
    "min": _traced_call,
}


class _TracedValues(_HostValues):
    """The pool of :meth:`Model.sparsity`: every quantity as the state elements it depends on.

    The states are the base values, each element carrying its own index. A
    name nothing provides (the time, a parameter, a thermodynamic table) is an
    input and carries nothing. Providers resolve as usual, with one difference
    from a numeric evaluation: a provider with no state-dependent input does
    not run, since its result cannot depend on the state either. So a table
    lookup, a ``max`` against a floor or an ``int()`` of a tag never sees a
    traced value. A provider that does branch on the state (RECFAST's rates
    switch between regimes) depends on all of its inputs.
    """

    def __init__(self, model):
        base = {}
        for s in model.states.values():
            base[s.name] = (
                _Traced(frozenset({s.offset}))
                if not s.shape
                else _traced_array(s.shape, s.offset)
            )
        super().__init__(model, base, _TRACING_NAMES)

    def __getitem__(self, name):
        if (
            name not in self.values
            and name not in self.model.states
            and name not in self.model.sums
            and name not in self.providers
        ):
            self.values[name] = _Traced()  # an input
        return super().__getitem__(name)

    def _call(self, fn, component, takes_self, args):
        if not any(_depends(a) for a in args):
            return _Traced()
        try:
            return super()._call(fn, component, takes_self, args)
        except TypeError:
            # The provider branches on a traced value (an if, max or min, a
            # conversion to a number), so the trace can't tell which inputs
            # each result reads. Make everything it writes depend on
            # everything it reads: a superset of the pattern, never a miss.
            on = frozenset().union(*(_depends(a) for a in args))
            for a in args:
                if isinstance(a, np.ndarray) and a.dtype == object and not _depends(a):
                    a[...] = _Traced(on)  # the output buffer it fills
            return _Traced(on)


class Model:
    """Components, their state in declaration order, and the functions built from them."""

    def __init__(self, components: Iterable[Component], sums: Sequence[str] = ()):
        self.components = list(components)
        self.sums = tuple(sums)
        labels = [c.label for c in self.components]
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        if duplicates:
            raise ValueError(f"two components of one class: {duplicates}")
        self.states: dict[str, _State] = {}
        offset = 0
        for c in self.components:
            for name, declared in type(c).declared_states():
                if name in self.states:
                    raise ValueError(
                        f"{c.label} declares {name!r}, already owned by "
                        f"{self.states[name].component.label}"
                    )
                if not name.isidentifier() or name.startswith("d_") or name == OUTPUT:
                    raise ValueError(f"{c.label}: {name!r} is not a valid state name")
                self.states[name] = _State(name, declared, c, offset)
                offset += self.states[name].size
        self.state: list[str] = [e for s in self.states.values() for e in s.elements]
        taken: dict[str, str] = {}
        for s in self.states.values():
            for e in s.elements:
                if e in taken or (e != s.name and e in self.states):
                    raise ValueError(
                        f"state element {e!r} of {s.name!r} clashes with {taken.get(e, e)!r}"
                    )
                taken[e] = s.name
        self.index = {name: i for i, name in enumerate(self.state)}
        self.source: str | None = None

    @property
    def nvar(self) -> int:
        return len(self.state)

    def _all_providers(self) -> list[_Provider]:
        """Every component's providers, read off their signatures once per model."""

        if not hasattr(self, "_provider_list"):
            self._provider_list = [
                p for c in self.components for p in _providers(c, self.states)
            ]
        return self._provider_list

    def _host_providers(self):
        """``(providers by quantity, contributions by sum)`` of a host evaluation, once per model."""

        if not hasattr(self, "_host_provider_table"):
            self._host_provider_table = self._collect({}, set(), shadow=True)[:2]
        return self._host_provider_table

    # --- the Jacobian's sparsity pattern ---------------------------------------------

    def sparsity(self) -> np.ndarray:
        """The ``(nnz, 2)`` index array of the Jacobian's nonzeros, traced through the providers.

        Runs the same resolution as :meth:`evaluate`, but on values that carry
        the set of state elements they depend on instead of numbers
        (:class:`_TracedValues`). Each contribution to ``d_<state>`` comes
        back carrying the columns of that row, a vector's element by element
        through the ``output`` it fills. The pattern is exact except at
        data-dependent branches, where a provider counts as reading all of its
        inputs. That gives the superset a colouring needs and no more. The
        diagonal the Rosenbrock matrix ``I/(h gamma) - J`` adds belongs to the
        solver, not the pattern. Rows come sorted.
        """

        values = _TracedValues(self)
        entries: set[tuple[int, int]] = set()
        for written in self._collect({}, set(), shadow=True)[2].values():
            for p in written:
                s = p.written
                if s.shape:
                    buffer = _traced_array(s.shape)
                    args = [buffer if n == OUTPUT else values[n] for n in p.signature]
                    values._call(p.fn, p.component, p.takes_self, args)
                    # row-major, the order of the elements
                    rows = enumerate(buffer.reshape(-1))
                else:
                    rows = [
                        (
                            0,
                            values._call(
                                p.fn,
                                p.component,
                                p.takes_self,
                                [values[d] for d in p.params],
                            ),
                        )
                    ]
                for i, traced in rows:
                    entries.update((s.offset + i, column) for column in traced.on)
        return np.array(sorted(entries), dtype=np.int64).reshape(-1, 2)

    def evaluate(self, *quantities, math=math, **base):
        """Evaluate ``quantities`` on the host from the ``base`` values.

        A name resolves as a base value, a state's initial value, a sum over
        the components, or a provider, whose dependencies resolve the same
        way. ``math`` is the module the providers' ``math.sqrt`` and friends
        use: the standard library for floats, ``jax.numpy`` for JAX arrays.
        That makes the background expansion rate the model's own ``grho`` and
        ``adotoa``, traced under ``jit``. Returns one value or a tuple.
        """

        memo = self.__dict__.setdefault("_host_memo", {}).setdefault(id(math), {})
        values = _HostValues(self, base, _host_names(math), memo)
        results = tuple(values[q] for q in quantities)
        return results[0] if len(quantities) == 1 else results

    def initial_state(self, k_values, **base) -> np.ndarray:
        """The initial state, ``(n_k, nvar)``, from the states' initial conditions.

        ``base`` holds the quantities the initial conditions may name besides
        ``k``: the time and the background parameters. A name resolves as a
        base quantity, a state (its own initial value), or a provider
        evaluated on the host from the same pool.

        One resolution covers all wave modes: ``k`` is the whole array,
        ``numpy`` stands in for ``math`` so the initial conditions' algebra
        broadcasts over it, and a vector state's filler writes rows of ``n_k``
        values (``batch_shape`` of :class:`_HostValues`). Resolving once per
        mode cost 6 ms per cosmology at 144 modes, mostly in inspecting every
        initial condition's signature at every mode: 0.85 s of a batch of 128
        cosmologies.
        """

        k_values = np.asarray(k_values, dtype=np.float64)
        n_k = len(k_values)
        memo = self.__dict__.setdefault("_host_memo", {}).setdefault(id(np), {})
        values = _HostValues(
            self, dict(base, k=k_values), _host_names(np), memo, batch_shape=(n_k,)
        )
        y0 = np.zeros((n_k, self.nvar), dtype=np.float64)
        for s in self.states.values():
            value = np.broadcast_to(
                np.asarray(values[s.name], dtype=np.float64), s.shape + (n_k,)
            )
            y0[:, s.offset : s.offset + s.size] = value.reshape(s.size, n_k).T
        return y0

    # --- compiled functions ----------------------------------------------------------

    def _collect(self, params, base, shadow=False):
        """Providers by quantity, contributions by sum, derivative providers by state.

        A provider of a quantity in ``base`` is an error for the right-hand
        side, whose only base quantity is the time. With ``shadow`` (a
        :meth:`build_function` input) the given value wins and the provider
        drops out. That is how a kernel supplies a density directly.
        """

        producers: dict[str, _Provider] = {}
        contributions: dict[str, list[_Provider]] = {s: [] for s in self.sums}
        derivatives: dict[str, list[_Provider]] = {s: [] for s in self.states}
        for provider in self._all_providers():
            q = provider.attr
            if provider.written is not None:
                derivatives[provider.written.name].append(provider)
            elif q in self.sums:
                contributions[q].append(provider)
            elif q in producers:
                raise ValueError(
                    f"{provider.label} and {producers[q].label} both provide {q!r}"
                )
            elif q in self.states or q in self.index:
                raise ValueError(
                    f"{provider.label} provides {q!r}, which is a state variable"
                )
            elif q in params:
                raise ValueError(
                    f"{provider.label} provides {q!r}, which is a parameter"
                )
            elif q in base and not shadow:
                raise ValueError(f"{provider.label} provides {q!r}, which is given")
            elif q not in base:
                producers[q] = provider
        return producers, contributions, derivatives

    @staticmethod
    def _order(wanted, producers, contributions, given):
        """The quantities ``wanted`` (``(label, name)`` pairs) need, in dependency order."""

        def deps_of(quantity):
            if quantity in producers:
                return producers[quantity].params
            return [d for p in contributions[quantity] for d in p.params]

        needed: set[str] = set()
        stack = list(wanted)
        while stack:
            label, dep = stack.pop()
            if dep in given or dep in needed:
                continue
            if dep not in producers and dep not in contributions:
                raise NameError(f"{label} needs {dep!r}, which nothing provides")
            needed.add(dep)
            owner = producers[dep].label if dep in producers else f"sum {dep}"
            stack.extend((owner, d) for d in deps_of(dep))
        graph = {
            q: [d for d in deps_of(q) if d in needed]
            for q in itertools.chain(producers, contributions)
            if q in needed
        }
        try:
            return list(graphlib.TopologicalSorter(graph).static_order())
        except graphlib.CycleError as e:
            raise ValueError(f"circular dependency: {' -> '.join(e.args[1])}") from None

    def _device(self, jit, cuda, float64):
        """The compiler and the device names a build uses, or the stand-ins."""

        if jit is None:
            import sys

            names = _device_names()
            jit = names["cuda"].jit(device=True)
            # numba's SSA pass recurses once per basic block, and the driver
            # has a few blocks per provider call under the Python error model.
            sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
        elif cuda is None or float64 is None:
            raise TypeError("a custom jit needs cuda and float64 stand-ins too")
        else:
            names = {"cuda": cuda, "float64": float64, "float32": float64}
        return jit, names

    def _subset(self, states) -> list:
        """The :class:`_State` records of ``states`` (``None`` for all), in model order."""

        if states is None:
            return list(self.states.values())
        unknown = [n for n in states if n not in self.states]
        if unknown:
            raise ValueError(f"no such states: {unknown}")
        return [s for s in self.states.values() if s.name in states]

    def _state_reads(self, states=None):
        """Lines reading the states from the flat ``y``: the model's, or the subset ``states`` packed in model order."""

        offset = 0
        for s in self._subset(states):
            if not s.shape:
                yield f"    {s.name} = y[{offset}]"
            else:
                yield _local_array(s.name, s.shape)
                yield from _each(
                    s.shape,
                    lambda ix, flat, s=s, o=offset: f"{s.name}[{ix}] = y[{o} + {flat}]",
                )
            offset += s.size

    @staticmethod
    def _codegen(jit, names):
        """A namespace for generated source and ``call(provider, buffer)`` emitting a call into it."""

        memo: dict = {}
        namespace: dict[str, object] = {
            "cuda": names["cuda"],
            "float64": names["float64"],
        }

        def call(provider, buffer=None):
            namespace[provider.symbol] = _bound_function(provider.fn, jit, names, memo)
            args = [
                buffer if n == OUTPUT and buffer is not None else n
                for n in provider.signature
            ]
            if provider.takes_self:
                namespace[provider.self_symbol] = provider.component.device_self()
                args.insert(0, provider.self_symbol)
            return f"{provider.symbol}({', '.join(args)})"

        return namespace, call

    @staticmethod
    def _quantity_lines(order, producers, contributions, params, call):
        """The parameter reads and the quantities of ``order``, each its provider or its sum."""

        for pname, column in params.items():
            yield f"    {pname} = p[{int(column)}]"
        for quantity in order:
            if quantity in producers:
                yield f"    {quantity} = {call(producers[quantity])}"
            else:
                yield f"    {quantity} = " + " + ".join(
                    call(p) for p in contributions[quantity]
                )

    def state_indices(self, states) -> list[int]:
        """The flat indices of the elements of ``states``, in model order."""

        return [s.offset + i for s in self._subset(states) for i in range(s.size)]

    def state_values(self, y, xp=np) -> dict:
        """The states of a flat state array ``y`` (last axis), by name.

        A vector state comes back with its components as the leading axis (a
        matrix with two), so a provider's ``theta[l]`` and ``psi[i, l]`` index
        the component and broadcast over the rest. That lets :meth:`evaluate`
        run the model over a whole history on the host. ``xp`` is ``numpy``
        or ``jax.numpy``, matching ``y``.
        """

        values = {}
        for s in self.states.values():
            block = y[..., s.offset : s.offset + s.size]
            if not s.shape:
                values[s.name] = block[..., 0]
            else:
                values[s.name] = xp.moveaxis(block, -1, 0).reshape(
                    s.shape + block.shape[:-1]
                )
        return values

    def build(
        self,
        params: Mapping[str, int],
        *,
        time: str = "tau",
        name: str = "rhs",
        states: Sequence[str] | None = None,
        quantities: Sequence[str] = (),
        jit: Callable | None = None,
        cuda=None,
        float64=None,
    ) -> Callable:
        """Generate ``rhs(y, tau, p) -> tuple`` and return it as a plain Python function.

        ``params`` maps parameter names to their column in the row ``p``. By
        default the build compiles the providers with
        ``cuda.jit(device=True)`` against the CUDA ``cuda`` and ``float64``.
        A test may pass ``jit`` (an identity for plain Python) and stand-ins
        for the two device names to run the model on the host.

        ``states`` restricts the function to a block of the model: ``y`` then
        holds those states alone, packed in model order, and the function
        returns their rates. ``quantities`` are names the function takes as
        given, in a tuple ``q`` after ``p`` (``rhs(y, tau, p, q)``), in place
        of their providers. A block of a modax hierarchy reads these from the
        blocks upstream of it. A block whose rates need a state outside it
        fails to compile, which checks that the coupling is one-way.
        """

        jit, names = self._device(jit, cuda, float64)
        if time in self.states or time in params or time in self.index:
            raise ValueError(f"the time {time!r} clashes with a state or parameter")
        for pname in params:
            if pname in self.states or pname in self.index:
                raise ValueError(f"parameter {pname!r} clashes with a state variable")
        quantities = tuple(quantities)
        block = self._subset(states)
        producers, contributions, derivatives = self._collect(
            params, {time, *quantities}, shadow=bool(quantities)
        )
        missing = [s.name for s in block if not derivatives[s.name]]
        if missing:
            raise ValueError(f"no component writes the derivatives of {missing}")
        given = {time, *(s.name for s in block), *params, *quantities}
        wanted = [
            (p.label, d) for s in block for p in derivatives[s.name] for d in p.params
        ]
        order = self._order(wanted, producers, contributions, given)

        namespace, call = self._codegen(jit, names)

        signature = f"y, {time}, p" + (", q" if quantities else "")
        lines = [
            f"def {name}({signature}):",
            "    # states",
            *self._state_reads(states),
            "    # parameters",
            *self._quantity_lines((), {}, {}, params, call),
            *(f"    {quantity} = q[{k}]" for k, quantity in enumerate(quantities)),
            "    # quantities",
            *self._quantity_lines(order, producers, contributions, {}, call),
            "    # derivatives",
        ]
        for s in block:
            d = f"d_{s.name}"
            if not s.shape:
                lines.append(
                    f"    {d} = " + " + ".join(call(p) for p in derivatives[s.name])
                )
                continue
            lines.extend(_zeros(d, s.shape))
            for p in derivatives[s.name]:
                buffer = f"c_{p.symbol}"
                lines.extend(_zeros(buffer, s.shape))
                lines.append(f"    {call(p, buffer)}")
                lines.extend(
                    _each(
                        s.shape, lambda ix, _, d=d, b=buffer: f"{d}[{ix}] += {b}[{ix}]"
                    )
                )
        lines.append("    return (")
        for s in block:
            if not s.shape:
                lines.append(f"        d_{s.name},")
            elif len(s.shape) == 1:
                lines.extend(f"        d_{s.name}[{i}]," for i in range(s.shape[0]))
            else:
                lines.extend(
                    f"        d_{s.name}[{i}, {j}],"
                    for i in range(s.shape[0])
                    for j in range(s.shape[1])
                )
        lines.append("    )")
        self.source = "\n".join(lines) + "\n"
        return _compile(name, self.source, namespace)

    def build_function(
        self,
        outputs: Sequence[str],
        *,
        inputs: Sequence[str] = (),
        params: Mapping[str, int] | None = None,
        name: str = "quantities",
        states: Sequence[str] | None = None,
        jit: Callable | None = None,
        cuda=None,
        float64=None,
    ) -> Callable:
        """Generate a device function returning the quantities ``outputs``.

        ``inputs`` are the base quantities it takes as positional arguments, in
        that order. The name ``"y"`` among them stands for the flat state,
        read into the states as :meth:`build` does (only the block ``states``
        when given, packed in model order). With ``params`` the function takes
        the parameter row ``p`` last. The components provide everything else
        the outputs need, so a kernel that needs the expansion rate or the
        line-of-sight sources computes them from the same equations as the
        right-hand side. Returns a scalar for one output, a tuple otherwise.
        """

        jit, names = self._device(jit, cuda, float64)
        params = dict(params or {})
        inputs = tuple(inputs)
        with_state = "y" in inputs
        given = {*inputs, *params}
        if with_state:
            given |= {s.name for s in self._subset(states)}
        producers, contributions, _ = self._collect(params, given, shadow=True)
        order = self._order(
            [(name, o) for o in outputs], producers, contributions, given
        )

        namespace, call = self._codegen(jit, names)

        signature = list(inputs) + (["p"] if params else [])
        lines = [f"def {name}({', '.join(signature)}):"]
        if with_state:
            lines.extend(self._state_reads(states))
        lines.extend(
            self._quantity_lines(order, producers, contributions, params, call)
        )
        lines.append(
            "    return "
            + (outputs[0] if len(outputs) == 1 else "(" + ", ".join(outputs) + ")")
        )
        source = "\n".join(lines) + "\n"
        fn = _compile(name, source, namespace)
        return jit(fn)


def _local_array(name: str, shape: tuple) -> str:
    """A ``cuda.local.array`` of ``shape``, its size a literal as numba needs."""

    size = shape[0] if len(shape) == 1 else tuple(shape)
    return f"    {name} = cuda.local.array({size}, float64)"


def _each(shape: tuple, body: Callable[[str, str], str]) -> list[str]:
    """Loop lines over the elements of a vector or matrix ``shape``.

    ``body(index, flat)`` gives the statement at element ``index`` (``i`` or
    ``i, j``), whose offset in the row-major element order is ``flat``.
    """

    if len(shape) == 1:
        return [f"    for i in range({shape[0]}):", f"        {body('i', 'i')}"]
    n, m = shape
    return [
        f"    for i in range({n}):",
        f"        for j in range({m}):",
        f"            {body('i, j', f'i * {m} + j')}",
    ]


def _zeros(target: str, shape: tuple) -> list[str]:
    """A local array of ``shape`` at -0.0, the additive identity, so the first += folds away."""

    return [
        _local_array(target, shape),
        *_each(shape, lambda ix, _: f"{target}[{ix}] = -0.0"),
    ]


class _Generated(Protocol):
    """A generated function, carrying the source it was compiled from."""

    _generated_source: str

    def __call__(self, *args: Any) -> Any: ...

    def __hash__(self) -> int: ...


def _compile(name: str, source: str, namespace: dict[str, object]) -> _Generated:
    """Exec ``source`` under a filename ``linecache`` knows, so numba can read it back."""

    filename = f"<discoeb generated {name} {next(_SOURCES)}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    scope: dict[str, object] = {}
    exec(compile(source, filename, "exec"), namespace, scope)  # noqa: S102
    fn: Any = scope[name]
    fn._generated_source = source
    return fn
