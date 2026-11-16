"""
A cosmological model in DISCO-EB is comprised of a list of *components*. A component declares its settings and state
variables as class attributes and provides *quantities* through ordinary Python functions we call *providers*:

    class Baryons(Species):
        clxb = State(initial=lambda clxg: 0.75 * clxg)
        vb = State()

        def grhob_t(self, grhob, inv_a):
            return grhob * inv_a

        @contributes("dgrho")
        def dgrho_b(self, grhob_t, clxb):
            return grhob_t * clxb

        def d_vb(self, adotoa, vb, k, cs2, clxb):
            return -adotoa * vb + k * cs2 * clxb

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
matrix ``psi`` as ``psi_0_0, psi_0_1, ...``. A state also carries its
*initial condition*, ``State(..., initial=fn)``. Its parameters resolve
like a provider's.

**Providers.** Components provide quantities by defining them in methods. Each
method can take as inputs any other defined quantity. The topological sorting
algorithm that builds the right-hand side of the Einstein-Boltzmann system will
call the providers in the order that respects their dependencies. Providers
of the form ``d_<state>`` define a contribution to that state's derivative.
A provider may also be decorated with ``@contributes`` to contribute to
a sum over components, such as ``grho``, ``dgrho`` and ``dgq``. The
component that reads a sum declares it: ``grho = Sum()``.

**Parameters.** ``Omegam: ArrayLike = Parameter()`` declares a cosmological
parameter.

**Sparsity.** The sparsity pattern of the E-B system is automatically
determined by tracing the dependencies of the various provider methods.
"""

from __future__ import annotations

import functools
import graphlib
import inspect
import itertools
import linecache
import math
import os
import sys
import types
from collections import namedtuple
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, ClassVar, Generic, Literal, Protocol, TypeVar

import numpy as np

if sys.version_info >= (3, 11):
    from typing import dataclass_transform
else:
    from typing_extensions import dataclass_transform

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
            f"{self.name} is only available inside a provider compiled by Equations.build"
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
F = TypeVar("F")


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


# --- parameters -----------------------------------------------------------------------


class _Required:
    def __repr__(self):
        return "REQUIRED"


REQUIRED: Any = _Required()
"""The default of a cosmology key every cosmology must give."""

_DECLARED: dict[str, list[tuple[type, ParameterDeclaration]]] = {}
"""Every parameter any component class declares, by name: its owners and declarations."""


class ParameterDeclaration:
    """A cosmological parameter a component declares, as :func:`Parameter` and :func:`Derived` make it.

    Without ``value`` it is a key of the cosmology dictionary, which may leave
    it out when it has a ``default``. With ``value``, a function whose
    parameter names are other parameters, it is derived from them on the host
    in JAX. ``column`` puts it in the kernel's parameter row under its name,
    for the providers to read. ``directed`` puts it there as ``x_nominal`` and
    ``dir_x`` instead and generates the provider ``x = x_nominal + eps *
    dir_x``, so the forward sensitivities may move it. ``closes_budget`` marks
    the parameter the background solves for so that the model expands at
    ``H0`` today: a density, or anything else the expansion rate depends on,
    such as the height of a scalar field's potential.
    """

    def __init__(
        self,
        default=REQUIRED,
        *,
        value: Callable | None = None,
        column: bool = False,
        directed: bool = False,
        closes_budget: bool = False,
    ):
        if value is not None and not callable(value):
            raise TypeError("value must be a function of other parameters")
        if (value is not None or closes_budget) and default is not REQUIRED:
            raise TypeError("a derived parameter has no default")
        if value is not None and closes_budget:
            raise TypeError("the budget closure derives the parameter itself")
        self.default = default
        self.value: Any = value
        self.directed = directed
        self.column = column or directed
        self.closes_budget = closes_budget
        self.name: str | None = None
        self.reads: tuple[str, ...] = ()
        if value is not None:
            takes_self, reads = _positional_names(value, "a parameter's value")
            self.reads = tuple(reads)
            if takes_self:
                raise TypeError("a parameter's value reads other parameters, not self")

    def __set_name__(self, owner, name):
        if not name.isidentifier() or name.startswith("_"):
            raise ValueError(f"{owner.__name__}: {name!r} is not a parameter name")
        if name not in _annotated_names(owner):
            # Unannotated, a type checker would not see it in a model's
            # constructor (the dataclass_transform on Component).
            raise TypeError(
                f"{owner.__name__}.{name} needs an annotation: "
                f"{name}: ArrayLike = {'Derived' if not self.is_key else 'Parameter'}(...)"
            )
        if self.name is not None and self.name != name:
            raise ValueError(f"one parameter declared as {self.name!r} and {name!r}")
        self.name = name
        _DECLARED.setdefault(name, []).append((owner, self))

    @property
    def is_key(self) -> bool:
        """Whether a cosmology dictionary gives it (else it is derived)."""
        return self.value is None and not self.closes_budget

    @property
    def column_name(self) -> str:
        """The row column the kernel reads it from: ``x``, or ``x_nominal`` if directed."""
        return f"{self.name}_nominal" if self.directed else str(self.name)

    @functools.cached_property
    def provider(self) -> Callable:
        """``x(self, x_nominal, eps, dir_x)`` of a directed parameter, a provider of its component."""
        n = str(self.name)
        source = f"def {n}(self, {n}_nominal, eps, dir_{n}):\n    return {n}_nominal + eps * dir_{n}\n"
        scope: dict[str, Any] = {}
        exec(compile(source, f"<discoeb parameter {n}>", "exec"), {}, scope)  # noqa: S102
        return scope[n]

    def renamed(self, renames: Mapping[str, str]) -> ParameterDeclaration:
        """A copy for a named component: it reads its component's own parameters under their new names."""

        copy = ParameterDeclaration(
            self.default if self.value is None and not self.closes_budget else REQUIRED,
            column=self.column,
            directed=self.directed,
            closes_budget=self.closes_budget,
        )
        copy.value = self.value
        copy.reads = tuple(renames.get(r, r) for r in self.reads)
        return copy

    def __repr__(self):
        return f"Parameter({self.name})"


def _annotated_names(cls) -> set[str]:
    """The names a class body annotates, without evaluating the annotations."""

    if sys.version_info >= (3, 14):
        import annotationlib

        return set(
            annotationlib.get_annotations(cls, format=annotationlib.Format.FORWARDREF)
        )
    return set(vars(cls).get("__annotations__", {}))


def Parameter(
    default: Any = REQUIRED, *, column: bool = False, directed: bool = False
) -> Any:
    """Declare a cosmology key: ``Omegam: ArrayLike = Parameter()``.

    It is an argument of every model whose components include the class:
    required without a ``default``, optional with one. ``column`` puts it in
    the kernel's parameter row under its name, for the providers to read, and
    ``directed`` as ``x_nominal`` and ``dir_x``, so the forward sensitivities
    may move it (:class:`ParameterDeclaration`). The annotation is required:
    it is what lets a type checker see the argument (the
    ``dataclass_transform`` on :class:`Component`).
    """

    return ParameterDeclaration(default, column=column, directed=directed)


def Derived(
    value: Callable | None = None,
    *,
    column: bool = False,
    directed: bool = False,
    closes_budget: bool = False,
    init: Literal[False] = False,
) -> Any:
    """Declare a parameter derived from others: ``grhom: ArrayLike = Derived(lambda H0: ...)``.

    ``value`` is a function whose parameter names are other parameters. Or
    ``closes_budget`` marks the one parameter the background solves for, so
    that the model expands at ``H0`` today. Either way it is not a
    constructor argument: ``init`` is there for type checkers, which read
    its default (``False``) to leave it out of the model's signature.
    """

    if value is None and not closes_budget:
        raise TypeError("Derived needs a value function, or closes_budget=True")
    return ParameterDeclaration(
        value=value, column=column, directed=directed, closes_budget=closes_budget
    )


def declared_parameters(classes) -> dict[str, ParameterDeclaration]:
    """The parameters the component classes declare, by name, in class and declaration order.

    Raises ``ValueError`` if two of them declare one name by different
    :class:`ParameterDeclaration` objects.
    """

    found: dict[str, ParameterDeclaration] = {}
    owner: dict[str, str] = {}
    for cls in classes:
        for name, parameter in cls.declared_parameters():
            if name in found and found[name] is not parameter:
                raise ValueError(
                    f"{owner[name]} and {cls.__name__} both declare the parameter {name!r}"
                )
            if name not in found:
                found[name], owner[name] = parameter, cls.__name__
    closing = [n for n, p in found.items() if p.closes_budget]
    if len(closing) > 1:
        raise ValueError(f"only one parameter may close the budget, not {closing}")
    return found


def declarations_of(name: str) -> list[tuple[type, ParameterDeclaration]]:
    """The component classes declaring the parameter ``name``, with their declarations."""

    return list(_DECLARED.get(name, ()))


def derive_parameters(
    parameters: Mapping[str, ParameterDeclaration],
    given: Mapping,
    *,
    math=math,
    close=None,
) -> dict:
    """Return the value of every parameter in ``parameters``.

    A key's value is ``given[key]``, or its default. A derived parameter's is
    its ``value`` function of the others, evaluated with ``math`` as the
    module its ``math.sqrt`` and friends use (``jax.numpy`` for traced
    values). The parameter that closes the budget is ``close(name, values)``,
    where ``values`` holds every parameter that does not depend on it.
    """

    names = _host_names(math)
    memo: dict = {}
    values: dict = {}

    def reaches(name, target, seen=()):
        p = parameters.get(name)
        if p is None or name in seen:
            return False
        return any(r == target or reaches(r, target, (*seen, name)) for r in p.reads)

    def get(name, stack=()):
        if name in values:
            return values[name]
        if name in stack:
            raise ValueError("circular parameters: " + " -> ".join((*stack, name)))
        p = parameters.get(name)
        if p is None:
            raise NameError(
                f"{stack[-1] if stack else 'a parameter'} reads {name!r}, "
                "which no component of the model declares"
            )
        if p.closes_budget:
            if close is None:
                raise ValueError(f"nothing closes the budget for {name!r}")
            others = {
                m: get(m, (*stack, name))
                for m in parameters
                if m != name and not reaches(m, name)
            }
            value = close(name, others)
        elif p.value is None:
            if name in given:
                value = given[name]
            elif p.default is REQUIRED:
                raise KeyError(name)
            else:
                value = p.default
        else:
            fn = _bound_function(p.value, lambda f: f, names, memo)
            value = fn(*(get(r, (*stack, name)) for r in p.reads))
        values[name] = value
        return value

    for name in parameters:
        get(name)
    return values


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


class Sum:
    """A quantity summed over the components that contribute to it: ``grho = Sum()``.

    The component that reads a sum declares it, and any component of the
    model adds to it with :func:`contributes`. A model's sums are those its
    components declare. A sum keeps its name in a named component
    (:meth:`Component.named`).
    """

    def __repr__(self):
        return "Sum()"


# --- components -----------------------------------------------------------------------


@dataclass_transform(
    kw_only_default=True, eq_default=False, field_specifiers=(Parameter, Derived)
)
class Component:
    """Provides quantities and may own settings, parameters and state variables (see the module docstring).

    A component may define a ``check`` classmethod whose parameters name the
    cosmology keys it validates (``check(cls, w_DE_0, w_DE_a)``), raising
    ``ValueError`` for values it cannot represent.
    :meth:`ModelSpec.check <discoeb.models.ModelSpec.check>` calls
    it with their values, unless one of them is traced. ``check`` is never a
    provider.
    """

    fixes_gauge: ClassVar[bool] = False
    """Whether the component's rest frame fixes the gauge the model's equations are written in."""
    _setting_names: ClassVar[tuple[str, ...]] = ()
    _renames: ClassVar[Mapping[str, str]] = {}
    """Of a :meth:`named` component: the names it declares to those it takes in a model."""
    _named_parameters: ClassVar[tuple[tuple[str, ParameterDeclaration], ...] | None] = (
        None
    )

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
    def provided_names(cls) -> tuple[str, ...]:
        """The quantities the class's methods provide, as written (derivatives left out)."""

        found = []
        for klass in cls.__mro__:
            if klass in _FRAMEWORK_CLASSES:
                continue
            for attr, member in vars(klass).items():
                if attr.startswith("_") or attr == "check":
                    continue
                if isinstance(member, (staticmethod, types.FunctionType)):
                    quantity = _contributed(member, attr)
                    if not quantity.startswith("d_"):
                        found.append(quantity)
        return tuple(dict.fromkeys(found))

    @classmethod
    def named(cls, name: str):
        """Return this component under a name of its own: ``Fluid.named("dr")``.

        The copy is a subclass whose states, parameters and quantities
        carry the prefix ``name_``: the state ``delta`` becomes
        ``dr_delta``, the key ``Omega`` ``dr_Omega``, the quantity
        ``grho_t`` ``dr_grho_t``. Its functions are the class's own, and
        each name in their signatures that the component itself declares
        resolves to the prefixed one. Every other name, and a contribution
        to one of the model's sums, stays as written. So one model may list
        a component several times, each copy with its own state and
        parameters: two dark fluids, two scalar fields. Settings are shared.
        """

        if not name.isidentifier() or name.startswith("_"):
            raise ValueError(f"{name!r} is not a name for a component")
        if cls._renames:
            raise TypeError(f"{cls.__name__} already carries a name")
        key = (cls, name)
        if key not in _NAMED:
            own = [n for n, _ in cls.declared_states()]
            own += [n for n, _ in cls.declared_parameters()]
            own += cls.provided_names()
            renames = {n: f"{name}_{n}" for n in dict.fromkeys(own)}
            parameters = {
                renames[n]: p.renamed(renames) for n, p in cls.declared_parameters()
            }
            _NAMED[key] = type(
                f"{cls.__name__}_{name}",
                (cls,),
                {
                    "__doc__": cls.__doc__,
                    "__module__": cls.__module__,
                    "_renames": types.MappingProxyType(renames),
                    "_named_parameters": tuple(parameters.items()),
                    "__annotations__": dict.fromkeys(parameters, "ArrayLike"),
                    **parameters,
                },
            )
        return _NAMED[key]

    @classmethod
    def declared_states(cls) -> list[tuple[str, State]]:
        """The ``State`` attributes, in declaration order, base classes first."""

        found, seen = [], set()
        for klass in reversed(cls.__mro__):
            for attr, value in vars(klass).items():
                if isinstance(value, State) and attr not in seen:
                    seen.add(attr)
                    found.append((cls._renames.get(attr, attr), value))
        return found

    @classmethod
    def declared_sums(cls) -> tuple[str, ...]:
        """The names of the ``Sum`` attributes, base classes first."""

        return tuple(
            dict.fromkeys(
                attr
                for klass in reversed(cls.__mro__)
                for attr, value in vars(klass).items()
                if isinstance(value, Sum)
            )
        )

    @classmethod
    def declared_parameters(cls) -> list[tuple[str, ParameterDeclaration]]:
        """The parameter attributes (:func:`Parameter`, :func:`Derived`), in declaration order, base classes first."""

        if cls._named_parameters is not None:
            return list(cls._named_parameters)
        found, seen = [], set()
        for klass in reversed(cls.__mro__):
            for attr, value in vars(klass).items():
                if isinstance(value, ParameterDeclaration) and attr not in seen:
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


class Gravity(Component):
    """A component of the theory of gravity: the expansion rate, the metric and its constraints.

    The species read what these provide, the conformal Hubble rate ``adotoa``
    and the metric sources ``z`` and ``sigma``, and never each other, so a
    model changes its theory of gravity by replacing them.
    """


class Interaction(Component):
    """A component without state, writing the derivatives of species' state."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        owned = [attr for attr, value in vars(cls).items() if isinstance(value, State)]
        if owned:
            raise TypeError(
                f"{cls.__name__} is an Interaction and cannot own state {owned}"
            )


_FRAMEWORK_CLASSES = (Component, Species, Gravity, Interaction, object)
_SELF_TYPES: dict[tuple, type] = {}
_NAMED: dict[tuple, type] = {}
"""The named copies of component classes (:meth:`Component.named`), one per class and name."""


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
        sums: Sequence[str] = (),
        method: str | None = None,
    ):
        self.component, self.fn = component, fn
        method = method or attr
        self.label = f"{component.label}.{method}"
        self.symbol = f"{component.label}__{method}"
        if method != attr and not (attr.startswith("d_") or attr in sums):
            raise TypeError(
                f"{self.label} contributes to {attr!r}, which is neither one of "
                f"the model's sums {tuple(sums)} nor a derivative d_<state>; "
                f"the component that reads it declares it with {attr} = Sum()"
            )
        self.self_symbol = f"{component.label}_self"
        self.takes_self, names = _positional_names(fn, self.label)
        # the names as the model knows them: a named component's own carry its prefix
        names = [_model_name(component, n, sums) for n in names]
        self.written = None
        if attr.startswith("d_"):
            written = _model_name(component, attr[2:], sums)
            state = states.get(written)
            if state is None:
                raise ValueError(
                    f"{self.label} writes {written!r}, which no component owns"
                )
            self.attr = f"d_{written}"
            self.written = state
            _check_output(self.label, "the derivative", state.shape, names)
        else:
            self.attr = _model_name(component, attr, sums)
            if OUTPUT in names:
                raise TypeError(
                    f"{self.label}: only a vector state's derivative fills an {OUTPUT!r}"
                )
        self.params = tuple(n for n in names if n != OUTPUT)
        self.signature = tuple(names)


def contributes(quantity: str) -> Callable[[F], F]:
    """Mark a method as a contribution to a sum or a derivative: ``@contributes("dgrho")``.

    A method otherwise provides the quantity it is named after, and a sum
    (``grho``, ``dgrho``, ...) or a state's derivative ``d_x`` takes a
    contribution from several components. Under their shared name, those
    methods would clash in a model class that inherits them all, which type
    checkers report. So each contribution gets a name of its own and says
    what it contributes to::

        @contributes("dgrho")
        def dgrho_b(self, grhob_t, clxb):
            return grhob_t * clxb

    A method named after the sum still contributes to it at run time.
    """

    def mark(method: F) -> F:
        fn = method.__func__ if isinstance(method, staticmethod) else method
        setattr(fn, "_contributes", quantity)
        return method

    return mark


def _contributed(member, attr: str) -> str:
    """The quantity a class member provides: what it ``@contributes`` to, else its name."""

    fn = member.__func__ if isinstance(member, staticmethod) else member
    return getattr(fn, "_contributes", attr)


def _model_name(component: Component, name: str, sums: Sequence[str] = ()) -> str:
    """The name a model knows ``name`` by when ``component`` writes it: prefixed if a named component declares it."""

    if name in sums:
        return name
    return type(component)._renames.get(name, name)


def _providers(
    component: Component, states: Mapping[str, _State], sums: Sequence[str] = ()
) -> list[_Provider]:
    found, seen = [], set()
    for klass in type(component).__mro__:
        if klass in _FRAMEWORK_CLASSES:
            continue  # the framework's own methods are not providers
        for attr, member in vars(klass).items():
            if attr in seen or attr.startswith("_") or attr == "check":
                continue
            seen.add(attr)
            # Every method not named with a leading underscore is a provider, a
            # staticmethod one without self. It provides the quantity it is
            # named after, or the one it @contributes to.
            if isinstance(member, (staticmethod, types.FunctionType)):
                fn = member.__func__ if isinstance(member, staticmethod) else member
                quantity = _contributed(member, attr)
                found.append(_Provider(component, quantity, fn, states, sums, attr))
    provided = {p.attr for p in found}
    for attr, parameter in type(component).declared_parameters():
        # A directed parameter's provider, unless a subclass overrides it.
        if parameter.directed and attr not in provided:
            found.append(_Provider(component, attr, parameter.provider, states))
    for attr, value in vars(component).items():
        if (
            attr not in seen
            and not attr.startswith("_")
            and isinstance(value, types.FunctionType)
        ):
            seen.add(attr)
            found.append(_Provider(component, attr, value, states, sums))
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


class _Filled:
    """An initial condition's ``output``: records ``output[i] = value``, then stacks them.

    A filler writes and reads single elements. Nothing is written in place, so
    the values may be JAX arrays, traced under ``jit``, as well as floats or
    NumPy arrays: :meth:`array` stacks them with ``xp``, zeros where nothing
    was written, each broadcast to ``batch_shape``.
    """

    def __init__(self, shape, batch_shape, xp):
        self.shape, self.batch_shape, self.xp = tuple(shape), tuple(batch_shape), xp
        self.values: dict[tuple[int, ...], Any] = {}

    def _key(self, index) -> tuple[int, ...]:
        index = index if isinstance(index, tuple) else (index,)
        if len(index) != len(self.shape):
            raise IndexError(f"an initial condition indexes {self.shape} by {index}")
        return tuple(int(i) % n for i, n in zip(index, self.shape))

    def __setitem__(self, index, value):
        self.values[self._key(index)] = value

    def __getitem__(self, index):
        return self.values.get(self._key(index), 0.0)

    def array(self):
        xp = self.xp
        elements = [
            xp.broadcast_to(
                xp.asarray(self.values.get(i, 0.0), dtype=xp.float64), self.batch_shape
            )
            for i in np.ndindex(*self.shape)
        ]
        if not self.shape:
            return elements[0]
        return xp.stack(elements).reshape(self.shape + self.batch_shape)


class _HostValues:
    """The value of every quantity, resolved on demand on the host.

    A name is a base value, a state (its initial value: zeros, or its
    ``initial`` filler), or a provider evaluated from the same pool. The
    providers run as plain Python with ``names`` in place of the device
    names, so the values may be floats or JAX arrays.
    """

    def __init__(self, model, base, names, memo=None, batch_shape=(), xp=np):
        self.model, self.values, self.names = model, dict(base), names
        self.xp = xp
        self.memo = {} if memo is None else memo
        self.providers, self.contributions, self.derivatives = model._host_providers()
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
            output = _Filled(s.shape, self.batch_shape, self.xp)
            result = output.array() if s.shape or self.batch_shape else 0.0
            fn = s.declared.initial_fn
            if fn is not None:
                takes_self, params = _positional_names(fn, f"{name}.initial")
                _check_output(
                    f"{name}.initial", "the initial condition", s.shape, params
                )
                args = [
                    output
                    if p == OUTPUT
                    else self[_model_name(s.component, p, self.model.sums)]
                    for p in params
                ]
                filled = self._call(fn, s.component, takes_self, args)
                result = output.array() if s.shape else filled
        elif name in self.contributions:
            parts = [
                self._call(p.fn, p.component, p.takes_self, [self[d] for d in p.params])
                for p in self.contributions[name]
            ]
            if not parts:
                raise NameError(f"nothing contributes to the sum {name!r}")
            result = parts[0]
            for part in parts[1:]:
                result = result + part
        elif name.startswith("d_") and name[2:] in self.model.states:
            result = self._rate(self.model.states[name[2:]])
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

    def _rate(self, s):
        """A state's derivative: the sum of the contributions the components write."""

        written = self.derivatives[s.name]
        if not written:
            raise NameError(f"no component writes the derivative of {s.name!r}")
        total = None
        for p in written:
            if s.shape:
                output = _Filled(s.shape, self.batch_shape, self.xp)
                args = [output if n == OUTPUT else self[n] for n in p.signature]
                self._call(p.fn, p.component, p.takes_self, args)
                part = output.array()
            else:
                part = self._call(
                    p.fn, p.component, p.takes_self, [self[d] for d in p.params]
                )
            total = part if total is None else total + part
        return total


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


MARKED = -1
"""The pseudo-element a marked input carries through a trace (:meth:`Equations.stages`)."""


class _TracedValues(_HostValues):
    """The pool of :meth:`Equations.sparsity`: every quantity as the state elements it depends on.

    The states are the base values, each element carrying its own index. A
    name nothing provides (the time, a parameter, a thermodynamic table) is an
    input and carries nothing. Providers resolve as usual, with one difference
    from a numeric evaluation: a provider with no state-dependent input does
    not run, since its result cannot depend on the state either. So a table
    lookup, a ``max`` against a floor or an ``int()`` of a tag never sees a
    traced value. A provider that does branch on the state (RECFAST's rates
    switch between regimes) depends on all of its inputs.
    """

    def __init__(self, model, marked: Sequence[str] = ()):
        base = {}
        for s in model.states.values():
            base[s.name] = (
                _Traced(frozenset({s.offset}))
                if not s.shape
                else _traced_array(s.shape, s.offset)
            )
        super().__init__(model, base, _TRACING_NAMES)
        # inputs to follow like a state: each carries the pseudo-element MARKED
        self.marked = frozenset(marked)

    def __getitem__(self, name):
        if (
            name not in self.values
            and name not in self.model.states
            and name not in self.contributions
            and name not in self.providers
        ):
            # an input
            self.values[name] = (
                _Traced(frozenset({MARKED})) if name in self.marked else _Traced()
            )
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


def _minimum_vertex_cut(weight, edges, sources, sinks) -> set:
    """The lightest set of vertices separating ``sources`` from ``sinks``, nearest the sinks.

    A vertex cut is an edge cut of the graph with every vertex split into an
    arc of its weight, so this runs Edmonds-Karp on that graph. A source or a
    sink may itself be cut. Of the minimum cuts it returns the one whose
    sink side is smallest.
    """

    INF = float("inf")
    capacity: dict = {}
    arcs: dict = {}

    def arc(u, v, c):
        capacity[u, v] = capacity.get((u, v), 0.0) + c
        capacity.setdefault((v, u), 0.0)
        arcs.setdefault(u, set()).add(v)
        arcs.setdefault(v, set()).add(u)

    for n, w in weight.items():
        arc(("in", n), ("out", n), float(w))
    for u, v in edges:
        arc(("out", u), ("in", v), INF)
    for n in sources:
        arc("source", ("in", n), INF)
    for n in sinks:
        arc(("out", n), "sink", INF)

    def search(start, forward):
        parent = {start: None}
        queue = [start]
        for u in queue:
            for v in arcs.get(u, ()):
                residual = capacity[u, v] if forward else capacity[v, u]
                if v not in parent and residual > 0.0:
                    parent[v] = u
                    queue.append(v)
        return parent

    while True:
        parent = search("source", True)
        if "sink" not in parent:
            break
        path, v = [], "sink"
        while parent[v] is not None:
            path.append((parent[v], v))
            v = parent[v]
        flow = min(capacity[e] for e in path)
        if flow == INF:
            raise ValueError("no finite cut separates the sources from the sinks")
        for u, v in path:
            capacity[u, v] -= flow
            capacity[v, u] += flow
    reaches_sink = search("sink", False)
    return {
        n
        for n in weight
        if ("out", n) in reaches_sink and ("in", n) not in reaches_sink
    }


class Equations:
    """Components, their state in declaration order, and the functions built from them."""

    def __init__(self, components: Iterable[Component]):
        self.components = list(components)
        self.sums = tuple(
            dict.fromkeys(s for c in self.components for s in type(c).declared_sums())
        )
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
                p
                for c in self.components
                for p in _providers(c, self.states, self.sums)
            ]
        return self._provider_list

    def _host_providers(self):
        """``(providers by quantity, contributions by sum, derivative providers by state)`` of a host evaluation, once per model."""

        if not hasattr(self, "_host_provider_table"):
            self._host_provider_table = self._collect({}, set(), shadow=True)
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

        entries: set[tuple[int, int]] = set()
        for s, i, on in self._rate_dependencies(_TracedValues(self)):
            entries.update((s.offset + i, column) for column in on)
        return np.array(sorted(entries), dtype=np.int64).reshape(-1, 2)

    def _rate_dependencies(self, values):
        """``(state, element, what it depends on)`` of every contribution to a derivative, traced on ``values``."""

        for written in values.derivatives.values():
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
                    yield s, i, traced.on

    # --- stages ------------------------------------------------------------------------

    def stages(self, mode: Sequence[str] = ("k",)) -> tuple[tuple[str, ...], ...]:
        """``(background states, perturbation states)``, read off the equations.

        A state belongs to the background if its derivative depends neither
        on a ``mode`` input (the wave number) nor on a state that does, so
        the background is the largest set of states that evolves on its own.
        Nothing declares the split. The same trace as :meth:`sparsity` finds
        it, with the mode inputs followed like states: the scale factor, the
        recombination history and a scalar field's homogeneous value come out
        as the background of whatever model lists them.
        """

        key = tuple(mode)
        cache = self.__dict__.setdefault("_stages_cache", {})
        if key not in cache:
            owner = {
                s.offset + i: s.name
                for s in self.states.values()
                for i in range(s.size)
            }
            reads: dict[str, set] = {name: set() for name in self.states}
            for s, _, on in self._rate_dependencies(_TracedValues(self, marked=key)):
                reads[s.name] |= {MARKED if c == MARKED else owner[c] for c in on}
            perturbed = {name for name, on in reads.items() if MARKED in on}
            while True:
                more = {
                    n
                    for n, on in reads.items()
                    if n not in perturbed and on & perturbed
                }
                if not more:
                    break
                perturbed |= more
            cache[key] = (
                tuple(n for n in self.states if n not in perturbed),
                tuple(n for n in self.states if n in perturbed),
            )
        return cache[key]

    def published(self, mode: Sequence[str] = ("k",)) -> tuple[str, ...]:
        """The background quantities the perturbation equations take as given, as few as will do.

        The background block of a solve steps ahead of the perturbations and
        hands them what they read of it. Any set of quantities that
        separates the background states from the perturbation equations
        would do. This returns the smallest, counted in numbers handed over
        (a minimum vertex cut of the dependency graph), and among equals the
        one nearest the perturbations, so that work a quantity hides (the
        recombination rates behind the opacity) stays in the background
        block. A background state the perturbations read directly is
        published itself.
        """

        key = tuple(mode)
        cache = self.__dict__.setdefault("_published_cache", {})
        if key not in cache:
            cache[key] = self._published(key)
        return cache[key]

    def _published(self, mode) -> tuple[str, ...]:
        background, perturbations = self.stages(mode)
        if not background or not perturbations:
            return ()
        producers, contributions, derivatives = self._host_providers()
        values = _TracedValues(self, marked=mode)
        elements = set(self.state_indices(background))

        def reads(q):
            if q in producers:
                return producers[q].params
            return [d for p in contributions.get(q, ()) for d in p.params]

        def is_background(q):
            """Whether ``q`` depends on background states and on nothing else that varies."""
            on = _depends(values[q])
            return bool(on) and on <= elements

        # what the perturbation equations need, directly and through quantities
        direct = {
            d for name in perturbations for p in derivatives[name] for d in p.params
        }
        needed: set[str] = set()
        stack = list(direct)
        while stack:
            q = stack.pop()
            if q in needed or (q not in producers and q not in contributions):
                continue
            needed.add(q)
            stack.extend(reads(q))
        # the graph between the background states and the first readers that
        # are not background themselves
        inside = set(background) | {q for q in needed if is_background(q)}
        sinks = (direct & inside) | {
            d for q in needed - inside for d in reads(q) if d in inside
        }
        edges = [
            (d, q)
            for q in inside
            if q not in self.states
            for d in reads(q)
            if d in inside
        ]
        weight = {n: (self.states[n].size if n in self.states else 1) for n in inside}
        cut = _minimum_vertex_cut(
            weight, edges, sources=background, sinks=sorted(sinks)
        )
        vectors = [n for n in cut if n in self.states and self.states[n].shape]
        if vectors:
            raise NotImplementedError(
                f"the perturbations read the vector background states {vectors} "
                "directly; provide the scalars they need as quantities"
            )
        rank = {n: i for i, n in enumerate([*self.states, *producers, *contributions])}
        return tuple(sorted(cut, key=rank.__getitem__))

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

    def initial_state(self, k_values, xp=np, **base):
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

        ``xp`` is the array module, ``numpy`` or ``jax.numpy``. With
        ``jax.numpy`` the base values may be traced: nothing is written in
        place, so the initial state traces under ``jit`` and stays on the
        device.
        """

        k_values = xp.asarray(k_values, dtype=xp.float64)
        n_k = k_values.shape[0]
        memo = self.__dict__.setdefault("_host_memo", {}).setdefault(id(xp), {})
        values = _HostValues(
            self,
            dict(base, k=k_values),
            _host_names(xp),
            memo,
            batch_shape=(n_k,),
            xp=xp,
        )
        blocks = []
        for s in sorted(self.states.values(), key=lambda s: s.offset):
            if s.offset != sum(b.shape[1] for b in blocks):
                raise ValueError(f"state {s.name!r} does not follow its predecessor")
            value = xp.broadcast_to(
                xp.asarray(values[s.name], dtype=xp.float64), s.shape + (n_k,)
            )
            blocks.append(value.reshape(s.size, n_k).T)
        return xp.concatenate(blocks, axis=1)

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
            elif q in producers and producers[q].fn is provider.fn:
                continue  # one directed parameter two components declare
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
        # A scalar state's rate is a quantity too: the sum of what writes it,
        # under the name d_<state>, for a provider that needs it (a
        # constraint that reads eta', a field equation that reads h'').
        for name, written in derivatives.items():
            if written and not self.states[name].shape and f"d_{name}" not in base:
                contributions[f"d_{name}"] = written
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
            if d in order:
                continue  # a provider read the rate: it is a quantity above
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
