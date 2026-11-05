"""A right-hand side assembled from components whose functions name what they read.

A model is a list of *components*. A component declares its settings and its
state variables as class attributes and provides *quantities* through ordinary
Python functions whose **parameter names are the names of the quantities they
depend on**::

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

Nothing is passed explicitly. :meth:`Model.build` reads each provider's
dependencies off its signature, orders the providers with a topological sort
and generates a driver that calls them in that order, injecting each value by
name -- the dependency-injection pattern of ``inspect.signature`` +
``graphlib.TopologicalSorter``, resolved once, at build time. State variables
(``y[i]``), parameters (``p[j]``) and the time are the base quantities;
everything else is derived.

**Settings.** ``LMAX_G = Setting(15)`` declares a setting with its default. It
is loaded when the class is created, from the environment variable
``DISCO_<NAME>`` when set (cast to the default's type), and reads as a plain
value on the class and on its instances; the constructor takes an override as
a keyword argument (``Photons(LMAX_G=11)``). A provider reads it as
``self.LMAX_G``, like any other attribute of its component.

**States.** ``theta = State(shape)`` declares a state variable: a scalar
(``State()``), a vector (``State(16)``) or a matrix (``State((3, 16))``); the
shape may be a function of the component instance when it depends on its
settings. The flat state vector holds the components' states in declaration
order, a vector ``theta`` as the elements ``theta_0, theta_1, ...``
(``psi_0_0, psi_0_1, ...``). A state also carries its *initial condition*,
``State(..., initial=fn)``, a function (a lambda for a scalar, a
``_``-prefixed function defined just above for a vector) whose parameters are
injected like a provider's (:meth:`Model.evaluate` is the same resolution for
any quantity, on floats or JAX arrays), from the quantities
:meth:`Model.initial_state` is given (``k``, ``tau``, the background
densities), from other states' initial values, and from providers that can be
evaluated on the host (``clxg`` for ``theta[0]``). A scalar's returns the
value; a vector's takes ``output``, zero-filled, and writes into it. A state
without one starts at zero.

**Sparsity.** Nothing declares which state a derivative depends on: the
Jacobian's sparsity pattern is read off the providers. :meth:`Model.sparsity`
resolves the same dependency graph as a build, but on values that carry the
set of state elements they depend on instead of numbers (a state element its
own index, the time and the parameters nothing, every operation the union of
its operands'), so ``output[l] = kv * (l * x[l - 1] - (l + 1) * x[l + 1])``
is traced as the tridiagonal band it is, with no one saying so. The pattern
is exact; a data-dependent branch, the one construct that could hide a
dependency, is refused.

**Providers.** Every method of a component is a provider, named after the
quantity it provides, and so is a function stored on the instance in
``__init__``. It is an ordinary method: ``self`` is the component, whose
settings and other public numeric attributes (a tuple of quadrature weights
the constructor derived) it reads as ``self.NQMAX`` or ``self.weight[i]``; the
remaining parameters are the quantities it depends on. A method or a lambda
that reads nothing of its component may leave ``self`` out. Three kinds of
name are special. ``d_<state>`` provides a contribution to that state's
derivative: for a scalar state it returns the contribution; for a vector state
it takes ``output``, which the framework hands it filled with zeros, and
writes its contribution into it. A function named in the model's ``sums`` (for
the Einstein-Boltzmann system, ``grho``, ``dgrho`` and ``dgq``) contributes to
that sum over components. Names starting with an underscore are the
component's own business. Several components may write one state (the baryon
Euler equation and the Thomson drag both write ``vb``); the derivative is the
sum of their contributions. A vector state reaches a provider as a 1-D array
(a matrix as a 2-D one), so ``len()``, indexing and loops over multipoles are
the ordinary Python ones, and a provider may call a plain module-level helper
(``stream``), which is compiled for the device alongside it. Wherever a
function fills an array instead of returning, that array is its parameter
``output``.

On the device ``self`` is a named tuple of the component's constants
(:meth:`Component.device_self`), passed to each provider by the generated
driver as a compile-time constant, so ``self.NQMAX`` folds into the loop bound
and ``self.weight[i]`` into the unrolled sum exactly as a literal would. On
the host (:meth:`Model.evaluate`) it is the component itself.

**Device names.** A provider that needs scratch space may use ``cuda`` and
``float64`` imported from this module (``cuda.local.array(16, float64)``; the
size must be a literal, since numba types it before the optimiser folds
``self.N``); they are placeholders, replaced by the real objects when a
provider is compiled, or by the stand-ins a host test passes to
:meth:`Model.build`, so a model imports nothing from CUDA and runs as plain
Python under test.

Each provider is compiled as its own ``cuda.jit(device=True)`` function, and
the generated driver is one more. The driver copies ``y`` into a local array
per state, allocates the contribution buffers, calls the providers and sums
the contributions into local arrays it returns as a tuple, so the primal has
the fixed-size tuple signature the Enzyme-differentiated numba-CUDA kernel
needs while the components see arrays. numba-enzyme optimises the primal with
LLVM's ``O3`` before Enzyme differentiates it, which inlines the providers,
unrolls the loops and promotes every local array to scalars; what Enzyme and
the kernel see is the hand-written straight-line right-hand side. The
driver's text is kept on the function as ``_generated_source`` and registered
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
from typing import Generic, TypeVar

import numpy as np

SETTINGS_PREFIX = "DISCO_"
"""Environment variables ``DISCO_<NAME>`` override a component's settings."""

OUTPUT = "output"
"""The parameter through which a function fills an array instead of returning."""


# --- device names ---------------------------------------------------------------------


class _Ambient:
    """A name a provider uses that the build step supplies: see the module docstring."""

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
``x`` to single precision, so that a transcendental of it runs at single rate."""


def _device_names():
    from numba_cuda_mlir import cuda as cuda_module, types as cuda_types

    return {"cuda": cuda_module, "float64": cuda_types.float64, "float32": cuda_types.float32}


_HOST_NAMES = {
    "cuda": types.SimpleNamespace(local=types.SimpleNamespace(array=lambda shape, dtype: np.zeros(shape))),
    "float64": np.float64,
    "float32": np.float32,
}
"""What the device names mean when a provider runs as plain Python on the host."""


def _host_names(math_module) -> dict:
    """The host names for providers evaluated with ``math_module`` as ``math``.

    On arrays (``numpy``, ``jax.numpy``) a provider's ``max`` and ``min`` are
    the module's elementwise ``maximum`` and ``minimum``, so the clamps of the
    recombination block broadcast like the rest of its algebra.
    """

    names = {**_HOST_NAMES, "math": math_module}
    if hasattr(math_module, "maximum"):
        names.update(
            max=math_module.maximum, min=math_module.minimum,
            float32=math_module.float32, float64=math_module.float64,
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
        raise ValueError(f"{SETTINGS_PREFIX}{name}={text!r} is not a {type(default).__name__}") from e


T = TypeVar("T", bool, int, float, str)


class Setting(Generic[T]):
    """A component setting, ``LMAX_G = Setting(15)``, loaded from ``DISCO_LMAX_G``.

    On the class it reads as the loaded value; on an instance, as the
    constructor's override or the loaded value. The descriptor is typed by its
    default, so ``self.LMAX_G`` is an ``int`` to a type checker.
    """

    def __init__(self, default: T):
        if not isinstance(default, (bool, int, float, str)):
            raise TypeError(f"a Setting needs a bool, int, float or str default, not {default!r}")
        self.default = default
        self.name = None
        self.value = default

    def __set_name__(self, owner, name):
        if not (name.isidentifier() and name.isupper()):
            raise ValueError(f"{owner.__name__}: setting {name!r} must be an UPPER_CASE identifier")
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
    return isinstance(value, tuple) and bool(value) and all(
        isinstance(v, (bool, int, float)) for v in value
    )


# --- states ---------------------------------------------------------------------------


class State:
    """A state variable declared on a component: ``etak = State()``, ``theta = State(16)``.

    ``shape`` is ``()`` for a scalar, ``n`` for a vector, ``(n, m)`` for a
    matrix, or a function of the component instance returning one of those.
    ``initial`` is the initial condition; see the module docstring.
    """

    def __init__(self, shape=(), initial: Callable | None = None):
        self.shape = shape
        self.initial_fn = initial
        if initial is not None and not callable(initial):
            raise TypeError("initial must be a function of the quantities the initial condition reads")

    def resolve(self, component) -> tuple[int, ...]:
        spec = self.shape(component) if callable(self.shape) else self.shape
        if spec is None or spec == ():
            return ()
        if isinstance(spec, int):
            return (spec,)
        shape = tuple(int(n) for n in spec)
        if not shape or any(n <= 0 for n in shape) or len(shape) > 2:
            raise ValueError(f"a state is a scalar, a vector or a matrix, not of shape {spec}")
        return shape


# --- components -----------------------------------------------------------------------


class Component:
    """Provides quantities, and may own settings and state variables; see the module docstring."""

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
            raise TypeError(f"{self.label} has no settings {unknown}; it has {sorted(names)}")
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
            raise TypeError(f"{cls.__name__} is an Interaction and cannot own state {owned}")


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
        self.name, self.declared, self.component, self.offset = name, declared, component, offset
        self.shape = declared.resolve(component)
        self.elements = _elements(name, self.shape)
        self.size = len(self.elements)


# --- providers ------------------------------------------------------------------------


def _positional_names(fn: Callable, label: str) -> tuple[bool, list[str]]:
    """``(takes self, the names after it)`` of a provider or initial condition."""

    params = list(inspect.signature(fn).parameters.values())
    for p in params:
        if p.kind not in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) or p.default is not p.empty:
            raise TypeError(f"{label}: functions take plain positional parameters only")
    takes_self = bool(params) and params[0].name == "self"
    return takes_self, [p.name for p in params[takes_self:]]


class _Provider:
    """One function of a component: what it reads, what it provides."""

    def __init__(self, component: Component, attr: str, fn: Callable, states: Mapping[str, _State]):
        self.component, self.attr, self.fn = component, attr, fn
        self.owner = component.label
        self.label = f"{self.owner}.{attr}"
        self.symbol = f"{self.owner}__{attr}"
        self.self_symbol = f"{self.owner}_self"
        self.takes_self, names = _positional_names(fn, self.label)
        self.written = None
        if attr.startswith("d_"):
            state = states.get(attr[2:])
            if state is None:
                raise ValueError(f"{self.label} writes {attr[2:]!r}, which no component owns")
            self.written = state
            if state.shape and OUTPUT not in names:
                raise TypeError(
                    f"{self.label} writes the vector {state.name!r}: take {OUTPUT!r} and fill it"
                )
            if not state.shape and OUTPUT in names:
                raise TypeError(
                    f"{self.label} writes the scalar {state.name!r}: return the contribution "
                    f"instead of taking {OUTPUT!r}"
                )
        elif OUTPUT in names:
            raise TypeError(f"{self.label}: only a vector state's derivative fills an {OUTPUT!r}")
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
            # Every non-magic method is a provider, a staticmethod one without self.
            if isinstance(member, staticmethod):
                found.append(_Provider(component, attr, member.__func__, states))
            elif isinstance(member, types.FunctionType):
                found.append(_Provider(component, attr, member, states))
    for attr, value in vars(component).items():
        if attr not in seen and not attr.startswith("_") and isinstance(value, types.FunctionType):
            seen.add(attr)
            found.append(_Provider(component, attr, value, states))
    return found


def _bound_function(fn: Callable, jit: Callable, names: Mapping[str, object], memo: dict) -> Callable:
    """``fn`` compiled by ``jit`` with its globals resolved.

    numba resolves a global at compile time, so the function is rebound to a
    copy of its globals in which the device-name placeholders are the real
    objects (``names``), ``math`` is whatever ``names["math"]`` says (``math``
    itself on the device, ``jax.numpy`` when a quantity is evaluated on JAX
    arrays), ``max`` and ``min`` are the array module's elementwise ones when
    ``names`` says so, and the plain helpers it calls are bound the same way,
    recursively. With an identity ``jit`` this is how a function runs on the
    host.
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
            rebound[name] = names[name]  # a builtin the names override: max and min on arrays
    fn_bound = types.FunctionType(fn.__code__, rebound, fn.__name__, fn.__defaults__, fn.__closure__)
    compiled = jit(fn_bound)
    memo[fn] = compiled
    return compiled


# --- model ----------------------------------------------------------------------------

_SOURCES = itertools.count(1)


class _HostValues:
    """The value of every quantity, resolved on demand on the host.

    A name is a base value, a state (its initial value: zeros, or its
    ``initial`` filler), or a provider evaluated from the same pool. The
    providers run as plain Python with ``names`` standing in for the device
    names, so the values may be floats or JAX arrays alike.
    """

    def __init__(self, model, base, names, memo=None, batch_shape=()):
        self.model, self.values, self.names = model, dict(base), names
        self.memo = {} if memo is None else memo
        self.providers, self.contributions = model._host_providers()
        self.resolving: list[str] = []
        # trailing axes of every state's value (the wave modes of a vectorised
        # initial state), so that a vector state's filler writes rows of them
        self.batch_shape = tuple(batch_shape)

    def _call(self, fn, component, takes_self, args):
        bound = _bound_function(fn, lambda f: f, self.names, self.memo)
        return bound(component, *args) if takes_self else bound(*args)

    def __getitem__(self, name):
        if name in self.values:
            return self.values[name]
        if name in self.resolving:
            raise ValueError("circular dependency on the host: " + " -> ".join([*self.resolving, name]))
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
                if s.shape and OUTPUT not in params:
                    raise TypeError(f"{name}.initial: a vector's initial condition takes {OUTPUT!r} and fills it")
                if not s.shape and OUTPUT in params:
                    raise TypeError(f"{name}.initial: a scalar's initial condition returns the value")
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
            result = self._call(p.fn, p.component, p.takes_self, [self[d] for d in p.params])
        else:
            raise NameError(f"{name!r} is neither given nor provided by any component")
        self.resolving.pop()
        self.values[name] = result
        return result


# --- sparsity -------------------------------------------------------------------------


class _Traced:
    """A value standing in for a number while the providers are traced for the Jacobian pattern.

    It carries ``on``, the set of state elements (flat indices) it depends on:
    a state element carries its own index, an input nothing, and every
    operation on traced values gives one carrying the union of its operands',
    so a derivative's value ends up carrying exactly the columns of its row.
    Indexing is a no-op (a state-free provider is not run, and the one value
    standing in for its result may be indexed as the array it would have
    been). A branch on a traced value and a conversion to a number are the
    only ways a dependency could be lost, so both raise -- even on an input,
    since the branches taken might read different states -- and the provider
    is then taken to read every input for everything it writes
    (:meth:`_TracedValues._call`).
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
    __truediv__ = __rtruediv__ = __floordiv__ = __rfloordiv__ = __mod__ = __rmod__ = _join
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
        raise TypeError("a provider converts a state-dependent value to a number; it must stay an expression")

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
        return attr if not callable(attr) else _traced_call  # pi, e, inf, ... stay numbers


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


def _reads_state(value) -> bool:
    return bool(_depends(value))


_TRACING_NAMES = {
    "cuda": types.SimpleNamespace(local=types.SimpleNamespace(array=lambda shape, dtype: _traced_array(shape))),
    "float64": lambda x: x,  # a cast carries its operand's dependencies
    "float32": lambda x: x,
    "math": _TracingMath(),
    "max": _traced_call,
    "min": _traced_call,
}


class _TracedValues(_HostValues):
    """The pool of :meth:`Model.sparsity`: every quantity as the state elements it depends on.

    The states are the base values, each element carrying its own index; a
    name nothing provides (the time, a parameter, a thermodynamic table) is an
    input and carries nothing; a provider is resolved like any other. One
    thing differs from a numeric evaluation: a provider none of whose inputs
    depends on the state is not called, since its result cannot either -- so
    a table lookup, a ``max`` against a floor or an ``int()`` of a tag never
    sees a traced value. A provider that does branch on the state (RECFAST's
    rates switch between regimes) depends on all of its inputs.
    """

    def __init__(self, model):
        base = {}
        for s in model.states.values():
            base[s.name] = _Traced(frozenset({s.offset})) if not s.shape else _traced_array(s.shape, s.offset)
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
        if not any(_reads_state(a) for a in args):
            return _Traced()
        try:
            return super()._call(fn, component, takes_self, args)
        except TypeError:
            # The provider branches on a traced value (an if, max or min, a
            # conversion to a number), so the trace cannot tell which inputs
            # each result reads: everything it writes depends on everything
            # it reads. A superset of the pattern, never a miss.
            on = frozenset().union(*(_depends(a) for a in args))
            for a in args:
                if isinstance(a, np.ndarray) and a.dtype == object and not _reads_state(a):
                    a[...] = _Traced(on)  # the output buffer it fills
            return _Traced(on)


class Model:
    """Components, their state in declaration order, and what is built from them."""

    def __init__(self, components: Iterable[Component], sums: Sequence[str] = ()):
        self.components = list(components)
        self.sums = tuple(sums)
        labels = [c.label for c in self.components]
        duplicates = sorted({l for l in labels if labels.count(l) > 1})
        if duplicates:
            raise ValueError(f"two components of one class: {duplicates}")
        self.states: dict[str, _State] = {}
        self.owner: dict[str, Component] = {}
        offset = 0
        for c in self.components:
            for name, declared in type(c).declared_states():
                if name in self.owner:
                    raise ValueError(
                        f"{c.label} declares {name!r}, already owned by {self.owner[name].label}"
                    )
                if not name.isidentifier() or name.startswith("d_") or name == OUTPUT:
                    raise ValueError(f"{c.label}: {name!r} is not a valid state name")
                self.owner[name] = c
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

    def _host_providers(self):
        """``(providers by quantity, contributions by sum)``, read once per model."""

        if not hasattr(self, "_host_provider_table"):
            providers, contributions = {}, {s: [] for s in self.sums}
            for c in self.components:
                for p in _providers(c, self.states):
                    if p.written is not None:
                        continue
                    if p.attr in self.sums:
                        contributions[p.attr].append(p)
                    else:
                        providers.setdefault(p.attr, p)
            self._host_provider_table = (providers, contributions)
        return self._host_provider_table

    # -- the Jacobian's sparsity pattern ----------------------------------------------

    def sparsity(self) -> np.ndarray:
        """The ``(nnz, 2)`` index array of the Jacobian's nonzeros, traced through the providers.

        The same resolution as :meth:`evaluate`, on values that carry the set
        of state elements they depend on instead of numbers
        (:class:`_TracedValues`): each contribution to ``d_<state>`` comes back
        carrying the columns of that row, a vector's element by element through
        the ``output`` it fills. The pattern is exact up to data-dependent
        branches, where a provider counts as reading all of its inputs, so it
        is the superset a colouring needs and no more; the diagonal the Rosenbrock matrix ``I/(h gamma) - J``
        adds is the solver's business, not the pattern's. Rows are sorted.
        """

        values = _TracedValues(self)
        entries: set[tuple[int, int]] = set()
        for c in self.components:
            for p in _providers(c, self.states):
                s = p.written
                if s is None:
                    continue
                if s.shape:
                    buffer = _traced_array(s.shape)
                    args = [buffer if n == OUTPUT else values[n] for n in p.signature]
                    values._call(p.fn, p.component, p.takes_self, args)
                    rows = enumerate(buffer.reshape(-1))  # row-major, the order of the elements
                else:
                    rows = [(0, values._call(p.fn, p.component, p.takes_self, [values[d] for d in p.params]))]
                for i, traced in rows:
                    entries.update((s.offset + i, column) for column in traced.on)
        return np.array(sorted(entries), dtype=np.int64).reshape(-1, 2)

    def evaluate(self, *quantities, math=math, **base):
        """Evaluate ``quantities`` on the host from the ``base`` values.

        A name resolves as a base value, a state's initial value, a sum over
        the components, or a provider, with its dependencies resolved the same
        way. ``math`` is the module the providers' ``math.sqrt`` and friends
        mean here: the standard library for floats, ``jax.numpy`` for JAX
        arrays (so the background expansion rate is the model's own ``grho``
        and ``adotoa``, traced under ``jit``). Returns one value or a tuple.
        """

        memo = self.__dict__.setdefault("_host_memo", {}).setdefault(id(math), {})
        values = _HostValues(self, base, _host_names(math), memo)
        results = tuple(values[q] for q in quantities)
        return results[0] if len(quantities) == 1 else results

    def initial_state(self, k_values, **base) -> np.ndarray:
        """The initial state, ``(n_k, nvar)``, from the states' initial conditions.

        ``base`` holds the quantities the initial conditions may name besides
        ``k``: the time and the background parameters. A name is resolved as a
        base quantity, a state (its own initial value), or a provider evaluated
        on the host from the same pool.

        The pool is resolved once for all wave modes: ``k`` is the whole array,
        ``numpy`` stands in for ``math`` so that the initial conditions' algebra
        broadcasts over it, and a vector state's filler writes rows of
        ``n_k`` values (``batch_shape`` of :class:`_HostValues`). Resolving the
        pool once per mode instead cost 6 ms per cosmology at 144 modes (most
        of it in the signature inspection of every initial condition at every
        mode), 0.85 s of a batch of 128 cosmologies.
        """

        k_values = np.asarray(k_values, dtype=np.float64)
        n_k = len(k_values)
        memo = self.__dict__.setdefault("_host_memo", {}).setdefault(id(np), {})
        values = _HostValues(self, dict(base, k=k_values), _host_names(np), memo, batch_shape=(n_k,))
        y0 = np.zeros((n_k, self.nvar), dtype=np.float64)
        for s in self.states.values():
            value = np.broadcast_to(np.asarray(values[s.name], dtype=np.float64), s.shape + (n_k,))
            y0[:, s.offset:s.offset + s.size] = value.reshape(s.size, n_k).T
        return y0

    # -- compiled functions -----------------------------------------------------------

    def _collect(self, params, base, shadow=False):
        """Providers by quantity, contributions by sum, derivative providers by state.

        A provider of a quantity in ``base`` is an error for the right-hand
        side, whose only base quantity is the time; with ``shadow`` (a
        :meth:`build_function` input) the given value wins and the provider is
        left out, which is how a kernel supplies a density directly.
        """

        producers: dict[str, _Provider] = {}
        contributions: dict[str, list[_Provider]] = {s: [] for s in self.sums}
        derivatives: dict[str, list[_Provider]] = {s: [] for s in self.states}
        for c in self.components:
            for provider in _providers(c, self.states):
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
                    raise ValueError(f"{provider.label} provides {q!r}, which is a state variable")
                elif q in params:
                    raise ValueError(f"{provider.label} provides {q!r}, which is a parameter")
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

    def _state_reads(self):
        for s in self.states.values():
            if not s.shape:
                yield f"    {s.name} = y[{s.offset}]"
            elif len(s.shape) == 1:
                yield f"    {s.name} = cuda.local.array({s.shape[0]}, float64)"
                yield f"    for i in range({s.shape[0]}):"
                yield f"        {s.name}[i] = y[{s.offset} + i]"
            else:
                n, m = s.shape
                yield f"    {s.name} = cuda.local.array(({n}, {m}), float64)"
                yield f"    for i in range({n}):"
                yield f"        for j in range({m}):"
                yield f"            {s.name}[i, j] = y[{s.offset} + i * {m} + j]"

    def state_values(self, y, xp=np) -> dict:
        """The states of a flat state array ``y`` (last axis), by name.

        A vector state comes back with its components as the leading axis (a
        matrix with two), so a provider's ``theta[l]`` and ``psi[i, l]`` index
        the component and broadcast over the rest, which is how
        :meth:`evaluate` runs the model over a whole history on the host.
        ``xp`` is ``numpy`` or ``jax.numpy``, whichever ``y`` is.
        """

        values = {}
        for s in self.states.values():
            block = y[..., s.offset:s.offset + s.size]
            if not s.shape:
                values[s.name] = block[..., 0]
            else:
                values[s.name] = xp.moveaxis(block, -1, 0).reshape(s.shape + block.shape[:-1])
        return values

    def build(
        self,
        params: Mapping[str, int],
        *,
        time: str = "tau",
        name: str = "rhs",
        jit: Callable | None = None,
        cuda=None,
        float64=None,
    ) -> Callable:
        """Generate ``rhs(y, tau, p) -> tuple`` and return it as a plain Python function.

        ``params`` maps parameter names to their column in the row ``p``. By
        default the providers are compiled with ``cuda.jit(device=True)`` and
        see the CUDA ``cuda`` and ``float64``; a test may pass ``jit`` (an
        identity for plain Python) and stand-ins for the two device names to
        run the model on the host.
        """

        jit, names = self._device(jit, cuda, float64)
        if time in self.states or time in params or time in self.index:
            raise ValueError(f"the time {time!r} clashes with a state or parameter")
        for pname in params:
            if pname in self.states or pname in self.index:
                raise ValueError(f"parameter {pname!r} clashes with a state variable")
        producers, contributions, derivatives = self._collect(params, {time})
        missing = [s for s, ps in derivatives.items() if not ps]
        if missing:
            raise ValueError(f"no component writes the derivatives of {missing}")
        given = {time, *self.states, *params}
        wanted = [(p.label, d) for ps in derivatives.values() for p in ps for d in p.params]
        order = self._order(wanted, producers, contributions, given)

        memo: dict = {}
        namespace: dict[str, object] = {"cuda": names["cuda"], "float64": names["float64"]}

        def call(provider, buffer=None):
            namespace[provider.symbol] = _bound_function(provider.fn, jit, names, memo)
            args = [buffer if n == OUTPUT else n for n in provider.signature]
            if provider.takes_self:
                namespace[provider.self_symbol] = provider.component.device_self()
                args.insert(0, provider.self_symbol)
            return f"{provider.symbol}({', '.join(args)})"

        def zeros(target, shape):
            # -0.0 is the additive identity, so the first += folds away.
            if len(shape) == 1:
                yield f"    {target} = cuda.local.array({shape[0]}, float64)"
                yield f"    for i in range({shape[0]}):"
                yield f"        {target}[i] = -0.0"
            else:
                n, m = shape
                yield f"    {target} = cuda.local.array(({n}, {m}), float64)"
                yield f"    for i in range({n}):"
                yield f"        for j in range({m}):"
                yield f"            {target}[i, j] = -0.0"

        def accumulate(target, source, shape):
            if len(shape) == 1:
                yield f"    for i in range({shape[0]}):"
                yield f"        {target}[i] += {source}[i]"
            else:
                n, m = shape
                yield f"    for i in range({n}):"
                yield f"        for j in range({m}):"
                yield f"            {target}[i, j] += {source}[i, j]"

        lines = [f"def {name}(y, {time}, p):", "    # states", *self._state_reads(), "    # parameters"]
        for pname, column in params.items():
            lines.append(f"    {pname} = p[{int(column)}]")
        lines.append("    # quantities")
        for quantity in order:
            if quantity in producers:
                lines.append(f"    {quantity} = {call(producers[quantity])}")
            else:
                lines.append(f"    {quantity} = " + " + ".join(call(p) for p in contributions[quantity]))
        lines.append("    # derivatives")
        for s in self.states.values():
            d = f"d_{s.name}"
            if not s.shape:
                lines.append(f"    {d} = " + " + ".join(call(p) for p in derivatives[s.name]))
                continue
            lines.extend(zeros(d, s.shape))
            for p in derivatives[s.name]:
                buffer = f"c_{p.symbol}"
                lines.extend(zeros(buffer, s.shape))
                lines.append(f"    {call(p, buffer)}")
                lines.extend(accumulate(d, buffer, s.shape))
        lines.append("    return (")
        for s in self.states.values():
            if not s.shape:
                lines.append(f"        d_{s.name},")
            elif len(s.shape) == 1:
                lines.extend(f"        d_{s.name}[{i}]," for i in range(s.shape[0]))
            else:
                lines.extend(
                    f"        d_{s.name}[{i}, {j}]," for i in range(s.shape[0]) for j in range(s.shape[1])
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
        jit: Callable | None = None,
        cuda=None,
        float64=None,
    ) -> Callable:
        """Generate a device function returning the quantities ``outputs``.

        ``inputs`` are the base quantities it takes as positional arguments, in
        that order; the name ``"y"`` among them stands for the flat state, read
        into the states as :meth:`build` does. With ``params`` the function
        takes the parameter row ``p`` last. Everything else the outputs need is
        provided by the components, so a kernel that needs the expansion rate
        or the line-of-sight sources computes them from the same equations as
        the right-hand side. Returns a scalar for one output, a tuple otherwise.
        """

        jit, names = self._device(jit, cuda, float64)
        params = dict(params or {})
        inputs = tuple(inputs)
        with_state = "y" in inputs
        given = {*inputs, *params}
        if with_state:
            given |= set(self.states)
        producers, contributions, _ = self._collect(params, given, shadow=True)
        order = self._order([(name, o) for o in outputs], producers, contributions, given)

        memo: dict = {}
        namespace: dict[str, object] = {"cuda": names["cuda"], "float64": names["float64"]}

        def call(provider):
            namespace[provider.symbol] = _bound_function(provider.fn, jit, names, memo)
            args = list(provider.signature)
            if provider.takes_self:
                namespace[provider.self_symbol] = provider.component.device_self()
                args.insert(0, provider.self_symbol)
            return f"{provider.symbol}({', '.join(args)})"

        signature = list(inputs) + (["p"] if params else [])
        lines = [f"def {name}({', '.join(signature)}):"]
        if with_state:
            lines.extend(self._state_reads())
        for pname, column in params.items():
            lines.append(f"    {pname} = p[{int(column)}]")
        for quantity in order:
            if quantity in producers:
                lines.append(f"    {quantity} = {call(producers[quantity])}")
            else:
                lines.append(f"    {quantity} = " + " + ".join(call(p) for p in contributions[quantity]))
        for o in outputs:
            if o not in given and o not in producers and o not in contributions:
                raise NameError(f"{name}: {o!r} is neither an input nor provided")
        lines.append("    return " + (outputs[0] if len(outputs) == 1 else "(" + ", ".join(outputs) + ")"))
        source = "\n".join(lines) + "\n"
        fn = _compile(name, source, namespace)
        return jit(fn)


def _compile(name: str, source: str, namespace: dict[str, object]) -> Callable:
    """Exec ``source`` under a filename ``linecache`` knows, so numba can read it back."""

    filename = f"<discoeb generated {name} {next(_SOURCES)}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    scope: dict[str, object] = {}
    exec(compile(source, filename, "exec"), namespace, scope)  # noqa: S102
    fn = scope[name]
    fn._generated_source = source
    return fn
