"""Cosmological models, built from the species, interactions and gravity components.

A model is a class whose bases are its components, and an instance of it is
a cosmology: the model with values for its parameters. The components declare
the parameters (:func:`~discoeb.component.Parameter`), so the constructor
takes exactly the model's, and a type checker sees them::

    class LCDM_MNU(LCDM, MassiveNeutrinos):
        \"\"\"LCDM with massive neutrinos.\"\"\"

    planck = LCDM(H0=67.32, Omegam=0.3158, Omegab=0.0494, A_s=2.1e-9,
                  n_s=0.966, Tcmb=2.7255, YHe=0.2454, Neff=3.046)
    solve_eb([planck, planck.replace(H0=70.0)], k, tau, keep=("R",))

A model's bases are its components in state order, a model among them
standing for its own components: ``LCDM_MNU`` is LCDM's components, then
:class:`~discoeb.species.MassiveNeutrinos`. :meth:`Model.swap`,
:meth:`Model.add`, :meth:`Model.remove` and :meth:`Model.with_settings`
derive models at run time; a type checker sees only those written as class
statements.
"""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass, field
from typing import Any, ClassVar

import jax

from .component import (
    REQUIRED,
    Component,
    Equations,
    ParameterDeclaration,
    declarations_of,
    declared_parameters,
)
from .cosmology import DIFFERENTIABLE_PARAMETERS, _cosmology_dictionary, _is_concrete
from .gravity import (
    GRAVITY_INTERFACE,
    SUMS,
    AdiabaticMode,
    Background,
    Curvature,
    Kinematics,
    Metric,
)
from .interactions import ThomsonScattering
from .recombination import Recombination
from .species import (
    Baryons,
    ColdDarkMatter,
    CosmologicalConstant,
    DarkEnergy,
    MassiveNeutrinos,
    MasslessNeutrinos,
    Photons,
    RadiationStreaming,
)

__all__ = [
    "DEFAULT_MODEL",
    "LCDM",
    "LCDM_MNU",
    "W0WACDM",
    "W0WACDM_MNU",
    "Model",
    "ModelSpec",
]


@dataclass(frozen=True)
class ModelSpec:
    """What the solver reads of a :class:`Model`: its components in state order and their settings.

    Every model class carries one (``LCDM.spec``). It is a value, hashable
    and compared by its fields, so the solver caches its compiled kernel on
    it, and :meth:`resolved` makes every setting explicit for that. A
    cosmology's parameters, as a dictionary, are a point in its parameter
    space, and :meth:`check` rejects one the model cannot represent.

    :meth:`replace`, :meth:`add`, :meth:`remove` and :meth:`with_settings`
    derive specifications as :meth:`Model.swap`, :meth:`Model.add`,
    :meth:`Model.remove` and :meth:`Model.with_settings` derive models.

    ``settings`` overrides component settings by name (``LMAX_G``,
    ``NQMAX``, ...). A setting it leaves out takes the component's own value,
    which ``DISCO_<NAME>`` in the environment or rebinding the class
    attribute (``Photons.LMAX_G = 20``) sets.
    """

    name: str
    components: tuple[type[Component], ...]
    settings: tuple[tuple[str, object], ...] = field(default=())

    def __post_init__(self):
        object.__setattr__(self, "components", tuple(self.components))
        object.__setattr__(self, "settings", tuple(sorted(dict(self.settings).items())))
        if len(set(self.components)) != len(self.components):
            raise ValueError(f"{self.name} lists a component twice")
        known = {n for cls in self.components for n in cls.setting_names()}
        unknown = sorted(dict(self.settings).keys() - known)
        if unknown:
            raise ValueError(
                f"{self.name} has no component with settings {unknown}; "
                f"its settings are {sorted(known)}"
            )
        self.parameters()  # raises if two components declare one name differently
        if not any(cls.fixes_gauge for cls in self.components):
            raise ValueError(
                f"{self.name} has no species whose rest frame fixes the "
                "synchronous gauge: keep ColdDarkMatter, at whatever density"
            )
        provided = {n for cls in self.components for n in cls.provided_names()}
        missing = [n for n in GRAVITY_INTERFACE if n not in provided]
        if missing:
            raise ValueError(
                f"{self.name} has no gravity component that provides {missing}: "
                "the species read the expansion rate adotoa (Background) and "
                "the metric sources z and sigma (Metric)"
            )

    def __repr__(self) -> str:
        return f"ModelSpec({self.name!r})"

    # --- deriving models ------------------------------------------------------

    def replace(
        self, old: type[Component], new: type[Component], *, name: str
    ) -> ModelSpec:
        """Return this model with ``new`` in place of ``old``, at its position."""

        if old not in self.components:
            raise ValueError(f"{self.name} has no {old.__name__}")
        components = tuple(new if c is old else c for c in self.components)
        return ModelSpec(name, components, self._settings_for(components))

    def add(self, *components: type[Component], name: str) -> ModelSpec:
        """Return this model with ``components`` appended."""

        return ModelSpec(name, self.components + components, self.settings)

    def remove(self, *components: type[Component], name: str) -> ModelSpec:
        """Return this model without ``components``."""

        missing = [c.__name__ for c in components if c not in self.components]
        if missing:
            raise ValueError(f"{self.name} has no {missing}")
        kept = tuple(c for c in self.components if c not in components)
        return ModelSpec(name, kept, self._settings_for(kept))

    def with_settings(self, **settings) -> ModelSpec:
        """Return this model with the given component settings (``LMAX_G=20``)."""

        merged = {**dict(self.settings), **settings}
        return ModelSpec(self.name, self.components, tuple(merged.items()))

    def _settings_for(self, components) -> tuple[tuple[str, object], ...]:
        known = {n for cls in components for n in cls.setting_names()}
        return tuple((n, v) for n, v in self.settings if n in known)

    # --- reading the model ----------------------------------------------------

    def has(self, component: type[Component]) -> bool:
        """Whether the model includes ``component`` or a subclass of it."""

        return any(issubclass(c, component) for c in self.components)

    def setting(self, name: str):
        """Return a setting's value: the model's override or the component's own."""

        overrides = dict(self.settings)
        if name in overrides:
            return overrides[name]
        for cls in self.components:
            if name in cls.setting_names():
                return getattr(cls, name)
        raise KeyError(f"{self.name} has no setting {name!r}")

    def resolved(self) -> ModelSpec:
        """Return this model with every setting explicit, at the components' current values.

        The solver caches its kernels on the resolved model, so rebinding a
        component's setting before a solve compiles a new kernel.
        """

        names = {n for cls in self.components for n in cls.setting_names()}
        values = tuple((n, self.setting(n)) for n in sorted(names))
        return ModelSpec(self.name, self.components, values)

    def instances(self) -> list[Component]:
        """Return the components instantiated with the model's settings."""

        overrides = dict(self.settings)
        return [
            cls(**{n: overrides[n] for n in cls.setting_names() if n in overrides})
            for cls in self.components
        ]

    def build(self, thermal_history: Component | None = None, extra=()) -> Equations:
        """Return the component :class:`~discoeb.component.Equations` of this cosmological model.

        ``thermal_history`` provides ``a``, ``opacity``, ``cs2``, ``rhonu`` and
        ``pnu`` as functions of ``tau``. In the solve it is the recombination
        block of :func:`discoeb.kernel.joint_kernel`, in a test anything.
        Without it the model still gives the state layout, the initial
        conditions and the sparsity pattern. ``extra`` lists component
        instances to append (the line-of-sight sources of DISCO-CMB).
        """

        components = self.instances()
        if thermal_history is not None:
            components.insert(0, thermal_history)
        components.extend(extra)
        return Equations(components, sums=SUMS)

    def parameters(self, extra=()) -> dict[str, ParameterDeclaration]:
        """Return the parameters the model's components declare, by name.

        They include those of the recombination block every solve appends
        (:func:`discoeb.kernel.joint_model`) and of the component classes
        ``extra``. The keys among them make up the cosmology dictionary
        (:meth:`parameter_keys`), the columns the kernel's parameter row
        (:func:`discoeb.kernel.parameter_layout`).
        """

        return declared_parameters((*self.components, Recombination, *extra))

    def parameter_keys(self) -> dict[str, object]:
        """Return the cosmology-dictionary keys of :meth:`parameters` and their defaults.

        :data:`~discoeb.component.REQUIRED` marks a key every cosmology gives.
        """

        return {n: p.default for n, p in self.parameters().items() if p.is_key}

    def state_names(self) -> tuple[str, ...]:
        """Return the perturbation state's element names in index order."""

        return _state_names(self.resolved())

    def sparsity(self):
        """Return the ``(nnz, 2)`` index array of the perturbation Jacobian's nonzeros.

        modax's Rodas5P kernel uses the pattern twice. It colours the column
        intersection graph, so a Jacobian costs ``n_colours + 1`` Enzyme
        sweeps rather than ``nvar + 1``. And it orders the pattern,
        factorises it symbolically and compiles a sparse LU and a pair of
        sparse triangular solves for that exact structure.
        :meth:`Equations.sparsity <discoeb.component.Equations.sparsity>` traces the
        pattern through the equations, so the tridiagonal free-streaming
        tails and the Einstein constraints that tie the metric, the fluids
        and the lowest multipoles together come straight from the code that
        writes them. The pattern is exact: the superset a colouring needs and
        no more. For flat LCDM at the default truncations it has 165
        entries, 11 colours and 198 slots of ``L + U`` after AMD's fill-in
        (``tests/test_eb_sparsity.py``).

        The kernel factorises ``I/(h gamma) - J``. modax keeps a slot for every
        diagonal and resolves ``slot(i, i)`` through that table even where
        ``J`` has no entry (an undamped free-streaming multipole has none).
        This needs modax at or after commit 9f54d12 (2026-11-01).
        """

        return self.build().sparsity()

    def check(self, cosmology) -> None:
        """Raise ``ValueError`` if ``cosmology`` sets parameters this model ignores.

        ``cosmology`` is a parameter dictionary or a cosmology object
        (:func:`~discoeb.cosmology.as_cosmology`). A key that another
        component declares but none of this model's must keep its declared
        default: a model without the dark-energy fluid needs ``w_DE_0 = -1``
        and ``w_DE_a = 0``, one without massive neutrinos ``Nmnu = 0``. Then
        each component's ``check`` hook runs on the keys it names
        (:meth:`DarkEnergy.check <discoeb.species.DarkEnergy.check>` needs
        ``w != -1``). A traced value, under ``jit``, has nothing to compare
        yet and is skipped.
        """

        values = _cosmology_dictionary(cosmology)
        self.check_keys(values)
        keys = self.parameter_keys()
        filled = {**{k: v for k, v in keys.items() if v is not REQUIRED}, **values}
        for cls in (*self.components, Recombination):
            hook = getattr(cls, "check", None)
            if hook is None:
                continue
            names = [cls._renames.get(n, n) for n in inspect.signature(hook).parameters]
            if any(n not in filled for n in names):
                continue  # a required key left out fails where it is read
            args = [filled[n] for n in names]
            if not all(map(_is_concrete, args)):
                continue  # traced under jit: nothing to compare yet
            try:
                hook(*args)
            except ValueError as e:
                raise ValueError(f"{self.name}: {e}") from None

    def check_keys(self, cosmology) -> None:
        """Raise ``ValueError`` if ``cosmology`` sets a parameter of a component this model lacks.

        The first half of :meth:`check`: a key that another component
        declares and none of this model's must keep its declared default, or
        the solve would ignore it without a word.
        """

        values = _cosmology_dictionary(cosmology)
        parameters = self.parameters()
        for key, value in values.items():
            if key in parameters or not _is_concrete(value):
                continue
            owners = [(cls, p) for cls, p in declarations_of(key) if p.is_key]
            if not owners:
                continue  # a key no component declares
            default = owners[0][1].default
            if default is REQUIRED or float(value) != default:
                names = ", ".join(sorted({cls.__name__ for cls, _ in owners}))
                raise ValueError(
                    f"{self.name} has no component with the parameter {key} "
                    f"(set to {value}, default {default}); use a model with {names}"
                )


_STATE_NAMES: dict = {}


def _state_names(model: ModelSpec) -> tuple[str, ...]:
    if model not in _STATE_NAMES:
        _STATE_NAMES[model] = tuple(model.build().state)
    return _STATE_NAMES[model]


# --- models as classes ------------------------------------------------------


class Model(Recombination):
    """A cosmological model: a class whose bases are its components. Its instances are cosmologies.

    A model lists its components as bases, in state order, after ``Model``
    or after a model whose components it extends::

        class LCDM(Model, Kinematics, Background, ..., ThomsonScattering): ...
        class LCDM_MNU(LCDM, MassiveNeutrinos): ...

    The parameters the components declare are the keyword arguments of its
    constructor: required without a default, optional with one. A type
    checker reads them off the declarations (the ``dataclass_transform`` on
    :class:`~discoeb.component.Component`), and ``help``, IPython and the
    class docstring list them. Every model has the recombination block,
    whose ``tau_reio`` it inherits from
    :class:`~discoeb.recombination.Recombination`.

    Class keywords set component settings, ``class Coarse(LCDM, LMAX_G=10)``,
    and ``name=`` the name errors and the kernel cache use. ``LCDM.spec`` is
    the :class:`ModelSpec` the solver reads.

    A cosmology is immutable and a JAX pytree whose leaves are its
    parameters, so ``jax.jit`` traces them and ``jax.grad`` returns a
    cosmology of derivatives. Differentiating one outside the model's
    :func:`~discoeb.solver.differentiable_parameters` raises, so take the
    gradient in chosen parameters: ``grad(lambda om: f(planck.replace(Omegam=om)))``.
    """

    spec: ClassVar[ModelSpec]
    """The components and settings the solver reads, on which it caches the kernel."""
    _keys: ClassVar[dict[str, Any]] = {}
    _held: dict[str, Any] = {}
    """Traced keys of other components :meth:`from_dict` was given, held for the solve."""

    def __init_subclass__(cls, *, name: str | None = None, **settings):
        super().__init_subclass__()
        components: list[type[Component]] = []
        inherited: dict[str, object] = {}
        for base in cls.__bases__:
            if base is Model or not issubclass(base, Component):
                continue
            if issubclass(base, Model):
                components += base.spec.components
                inherited.update(base.spec.settings)
            else:
                components.append(base)
        cls.spec = ModelSpec(
            name or cls.__name__,
            tuple(dict.fromkeys(components)),
            tuple({**inherited, **settings}.items()),
        )
        cls._keys = cls.spec.parameter_keys()
        cls.__signature__ = inspect.Signature(
            [
                inspect.Parameter(
                    key,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=inspect.Parameter.empty if default is REQUIRED else default,
                )
                for key, default in cls._keys.items()
            ]
        )
        cls.__doc__ = _documented(cls)
        jax.tree_util.register_pytree_node(
            cls, _flatten, functools.partial(_unflatten, cls)
        )

    def __init__(self, *args, **values):
        cls = type(self)
        if args:
            raise TypeError(
                f"{cls.__name__} takes its parameters by keyword, "
                f"{cls.__name__}(H0=..., ...); from a dictionary, "
                f"{cls.__name__}(**params) or {cls.__name__}.from_dict(params)"
            )
        unknown = [key for key in values if key not in cls._keys]
        if unknown:
            owners = sorted(
                {owner.__name__ for key in unknown for owner, _ in declarations_of(key)}
            )
            hint = f"; it is a parameter of {', '.join(owners)}" if owners else ""
            raise TypeError(f"{cls.__name__} has no parameter {unknown[0]!r}{hint}")
        missing = [
            key
            for key, default in cls._keys.items()
            if default is REQUIRED and key not in values
        ]
        if missing:
            raise TypeError(f"{cls.__name__} needs the parameters {missing}")
        filled = {key: values.get(key, default) for key, default in cls._keys.items()}
        cls.spec.check(filled)
        _set(self, filled)

    @classmethod
    def from_dict(cls, params) -> Model:
        """Return the cosmology of a parameter dictionary.

        Unlike ``cls(**params)``, it accepts keys this model does not have
        while they keep their declared default (``w_DE_0 = -1`` for a model
        without dark energy), and ignores keys no component declares
        (:meth:`ModelSpec.check_keys`). For a dictionary written for another
        model. Under ``jit`` such a key's value is traced and can't be
        checked yet, so the cosmology holds it for the solve, which raises if
        anything differentiates it rather than return a silent zero.
        """

        cls.spec.check_keys(params)
        cosmology = cls(**{key: params[key] for key in cls._keys if key in params})
        held = {
            key: value
            for key, value in params.items()
            if key not in cls._keys and declarations_of(key) and not _is_concrete(value)
        }
        object.__setattr__(cosmology, "_held", held)
        return cosmology

    def to_dict(self) -> dict[str, Any]:
        """Return the parameters by name, every one of the model's."""

        return {key: getattr(self, key) for key in type(self)._keys}

    def replace(self, **changes) -> Model:
        """Return this cosmology with ``changes`` to its parameters: ``planck.replace(H0=70.0)``."""

        cosmology = type(self)(**{**self.to_dict(), **changes})
        object.__setattr__(cosmology, "_held", self._held)
        return cosmology

    __replace__ = replace

    def __setattr__(self, name, value):
        raise AttributeError(f"a cosmology is immutable: use replace({name}=...)")

    def __repr__(self) -> str:
        values = ", ".join(f"{k}={v!r}" for k, v in self.to_dict().items())
        return f"{type(self).__name__}({values})"

    # --- the model -----------------------------------------------------------

    @classmethod
    def parameters(cls, extra=()) -> dict[str, ParameterDeclaration]:
        """Return the parameters the model declares, by name (:meth:`ModelSpec.parameters`)."""

        return cls.spec.parameters(extra)

    @classmethod
    def parameter_keys(cls) -> dict[str, Any]:
        """Return the constructor's parameters and their defaults, :data:`REQUIRED` if none."""

        return dict(cls._keys)

    @classmethod
    def state_names(cls) -> tuple[str, ...]:
        """Return the perturbation state's element names in index order."""

        return cls.spec.state_names()

    @classmethod
    def sparsity(cls):
        """Return the perturbation Jacobian's pattern (:meth:`ModelSpec.sparsity`)."""

        return cls.spec.sparsity()

    @classmethod
    def setting(cls, name: str):
        """Return a setting's value: the model's override or the component's own."""

        return cls.spec.setting(name)

    @classmethod
    def has(cls, component: type[Component]) -> bool:
        """Whether the model includes ``component`` or a subclass of it."""

        return cls.spec.has(component)

    @classmethod
    def build(cls, thermal_history: Component | None = None, extra=()) -> Equations:
        """Return the model's :class:`~discoeb.component.Equations` (:meth:`ModelSpec.build`)."""

        return cls.spec.build(thermal_history, extra)

    # --- models derived at run time --------------------------------------------

    @classmethod
    def add(cls, *components: type[Component], name: str) -> type[Model]:
        """Return a model with ``components`` appended, as ``class name(cls, *components)``."""

        return _derived(name, (cls, *components), {})

    @classmethod
    def swap(
        cls, old: type[Component], new: type[Component], *, name: str
    ) -> type[Model]:
        """Return a model with ``new`` in place of ``old``, at its position."""

        spec = cls.spec.replace(old, new, name=name)
        return _derived(name, (Model, *spec.components), dict(spec.settings))

    @classmethod
    def remove(cls, *components: type[Component], name: str) -> type[Model]:
        """Return a model without ``components``."""

        spec = cls.spec.remove(*components, name=name)
        return _derived(name, (Model, *spec.components), dict(spec.settings))

    @classmethod
    def with_settings(cls, **settings) -> type[Model]:
        """Return this model with the given component settings (``LMAX_G=20``), a subclass."""

        if not settings:
            return cls
        return _derived(cls.spec.name, (cls,), settings)


_DERIVED: dict[tuple, type[Model]] = {}
"""The models derived at run time, one class per name, bases and settings."""


def _derived(name: str, bases: tuple, settings: dict) -> type[Model]:
    key = (name, bases, tuple(sorted(settings.items())))
    if key not in _DERIVED:
        _DERIVED[key] = type(
            name, bases, {"__module__": __name__}, name=name, **settings
        )
    return _DERIVED[key]


def _set(cosmology: Model, values: dict) -> None:
    for key, value in values.items():
        object.__setattr__(cosmology, key, value)


def _flatten(cosmology: Model):
    held = cosmology._held
    values = tuple(getattr(cosmology, key) for key in type(cosmology)._keys)
    return (*values, *held.values()), tuple(held)


def _unflatten(cls: type[Model], held: tuple, leaves) -> Model:
    # No check: under jit or a transformation the leaves are tracers or
    # placeholders.
    cosmology = object.__new__(cls)
    n = len(cls._keys)
    _set(cosmology, dict(zip(cls._keys, leaves[:n])))
    if held:
        object.__setattr__(cosmology, "_held", dict(zip(held, leaves[n:])))
    return cosmology


def _documented(cls: type[Model]) -> str:
    """The class's own docstring with its constructor's parameters listed after it."""

    own = inspect.cleandoc(vars(cls).get("__doc__") or "")
    keys = cls.spec.parameter_keys()
    lines = ["Parameters", "----------"]
    for key, default in keys.items():
        line = "Required." if default is REQUIRED else f"Default {default!r}."
        if key in DIFFERENTIABLE_PARAMETERS:
            line += " Differentiable through solve_eb."
        lines += [f"{key} : float or jax.Array", f"    {line}"]
    return "\n\n".join(part for part in (own, "\n".join(lines)) if part)


class LCDM(
    Model,
    Kinematics,
    Background,
    RadiationStreaming,
    AdiabaticMode,
    Curvature,
    Metric,
    ColdDarkMatter,
    Baryons,
    Photons,
    MasslessNeutrinos,
    CosmologicalConstant,
    ThomsonScattering,
):
    """LCDM with massless neutrinos.

    CDM, baryons, photons coupled by Thomson scattering, ``Neff`` massless
    neutrino flavours and a cosmological constant. Flat unless ``Omegak`` is
    given.
    """


class LCDM_MNU(LCDM, MassiveNeutrinos, name="LCDM+mnu"):
    """LCDM with ``Nmnu`` massive neutrino flavours of mass ``mnu`` on ``NQMAX`` momentum bins."""


class W0WACDM(
    Model,
    Kinematics,
    Background,
    RadiationStreaming,
    AdiabaticMode,
    Curvature,
    Metric,
    ColdDarkMatter,
    Baryons,
    Photons,
    MasslessNeutrinos,
    DarkEnergy,
    ThomsonScattering,
    name="w0waCDM",
):
    """LCDM with a perturbed dark-energy fluid in place of the cosmological constant.

    Its equation of state is ``w = w_DE_0 + w_DE_a (1 - a)`` and its sound
    speed ``cs2_DE``. Needs ``w != -1``.
    """


class W0WACDM_MNU(W0WACDM, MassiveNeutrinos, name="w0waCDM+mnu"):
    """w0waCDM with massive neutrinos."""


DEFAULT_MODEL = W0WACDM_MNU
"""The model a bare parameter dictionary stands for where nothing names one.

:func:`~discoeb.background.density_coefficients`,
:func:`~discoeb.background.background_history`,
:func:`~discoeb.solver.conformal_ages` and
:func:`~discoeb.solver.directed_inputs` take a cosmology, or a dictionary
for this model. Its components declare every built-in parameter, and on a
cosmology another built-in model accepts, its background is that model's. A
dictionary that sets a parameter of a component outside it raises there
(:meth:`ModelSpec.check_keys`)."""

DEFAULT_SPEC = DEFAULT_MODEL.spec
"""The :class:`ModelSpec` of :data:`DEFAULT_MODEL`."""


# --- what a solve takes ---------------------------------------------------------


@dataclass(frozen=True, eq=False)
class CosmologyBatch:
    """Cosmologies of one model as the solver reads them: the spec and parameter dictionaries.

    ``model`` is the :class:`ModelSpec`, on which the compiled kernel is
    cached, ``cosmologies`` a tuple of parameter dictionaries (their values
    JAX arrays or tracers) and ``model_class`` the :class:`Model` they came
    from.
    """

    model: ModelSpec
    cosmologies: tuple
    model_class: type[Model] | None = None

    def __len__(self) -> int:
        return len(self.cosmologies)


def _batch_of(cosmologies) -> CosmologyBatch:
    """Return one cosmology or a sequence of cosmologies of one model as a :class:`CosmologyBatch`."""

    if isinstance(cosmologies, CosmologyBatch):
        return cosmologies
    batch = (cosmologies,) if isinstance(cosmologies, Model) else tuple(cosmologies)
    if not batch or not all(isinstance(c, Model) for c in batch):
        raise TypeError(
            "cosmologies must be a cosmology, such as LCDM(H0=..., ...), or a "
            f"non-empty sequence of them, not {type(cosmologies).__name__}"
        )
    classes = list(dict.fromkeys(type(c) for c in batch))
    if len(classes) > 1:
        raise TypeError(
            "the cosmologies of one solve share a model, whose kernel it compiles: "
            f"got {', '.join(c.__name__ for c in classes)}"
        )
    return CosmologyBatch(
        classes[0].spec, tuple({**c._held, **c.to_dict()} for c in batch), classes[0]
    )


def _spec_of(model) -> ModelSpec:
    """Return a model class's :class:`ModelSpec`, or a spec as it is."""

    if isinstance(model, type) and issubclass(model, Model):
        return model.spec
    if isinstance(model, ModelSpec):
        return model
    raise TypeError(f"expected a model such as LCDM, not {model!r}")
