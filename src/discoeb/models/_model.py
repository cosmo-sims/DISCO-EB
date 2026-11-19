"""The base class of cosmological models.

A model is constructed by inheriting from base components in state order. An
instance of a model is constructed by passing values for its parameters.
The result is a particular physical cosmology.

>>> class LCDM_MNU(LCDM, MassiveNeutrinos):
...     \"\"\"LCDM with massive neutrinos.\"\"\"
>>> planck = LCDM(omega_b=0.02238, omega_c=0.1201, omega_de=0.3107,
...               A_s=2.1e-9, n_s=0.966, T_cmb=2.7255, Y_He=0.2454, N_eff=3.046)
>>> solve_eb([planck, planck.replace(omega_c=0.125)], k, a, keep=("R",))
"""

from __future__ import annotations

import functools
import inspect
import math
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from ..components._component import (
    REQUIRED,
    Component,
    Equations,
    ParameterDeclaration,
    declarations_of,
    declared_parameters,
    parameter_values,
)
from ..components.gravity import H100
from ..components.recombination import Recombination

__all__ = ["Model"]


class Model(Recombination):
    """A cosmological model. Its instances are cosmologies.

    A model lists its components as bases, in state order, after ``Model``
    or after a model whose components it extends::

        class LCDM(Model, Background, ..., ThomsonScattering): ...
        class LCDM_MNU(LCDM, MassiveNeutrinos): ...

    The parameters the components declare are the keyword arguments of its
    constructor. They are required if there is no default. A type
    checker reads them off the declarations (the ``dataclass_transform`` on
    :class:`~discoeb.components._component.Component`), and ``help``, IPython and the
    class docstring list them. Every model has the recombination block,
    whose ``z_re`` it inherits from
    :class:`~discoeb.components.recombination.Recombination`.

    Class keywords set component settings, ``class Coarse(LCDM, LMAX_G=10)``,
    and ``name=`` the name errors use. A setting a model leaves out takes the
    component's own value, which ``DISCO_<NAME>`` in the environment or
    rebinding the class attribute (``Photons.LMAX_G = 20``) sets. The solver
    caches its kernel on :meth:`resolved`, the model with every setting
    explicit.

    A cosmology is an immutable JAX pytree whose leaves are its
    parameters, so ``jax.jit`` traces them and ``jax.jacfwd`` returns a
    cosmology of derivatives, one column of the solve per parameter. Take
    it in chosen parameters to integrate fewer:
    ``jacfwd(lambda om: f(planck.replace(omega_c=om)))``.
    """

    name: ClassVar[str]
    """The model's name, ``name=`` or the class's."""
    components: ClassVar[tuple[type[Component], ...]]
    """The component classes, in state order."""
    settings: ClassVar[tuple[tuple[str, object], ...]]
    """The settings the model overrides, ``(name, value)`` sorted by name."""
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
                components += base.components
                inherited.update(base.settings)
            else:
                components.append(base)
        cls.name = name or cls.__name__
        cls.components = tuple(dict.fromkeys(components))
        cls.settings = tuple(sorted({**inherited, **settings}.items()))
        known = {n for c in cls.components for n in c.setting_names()}
        unknown = sorted(dict(cls.settings).keys() - known)
        if unknown:
            raise ValueError(
                f"{cls.name} has no component with settings {unknown}; "
                f"its settings are {sorted(known)}"
            )
        if not any(c.fixes_gauge for c in cls.components):
            raise ValueError(
                f"{cls.name} has no species whose rest frame fixes the "
                "synchronous gauge: keep ColdDarkMatter, at whatever density"
            )
        # raises if two components declare one name differently
        cls._keys = cls.parameter_keys()
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

    # --- a cosmology -----------------------------------------------------------

    def __init__(self, *args, **values):
        cls = type(self)
        if args:
            raise TypeError(
                f"{cls.__name__} takes its parameters by keyword, "
                f"{cls.__name__}(omega_b=..., ...); from a dictionary, "
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
        cls.validate(filled)
        _set(self, filled)

    @classmethod
    def from_dict(cls, params) -> Model:
        """Return the cosmology of a parameter dictionary.

        Unlike ``cls(**params)``, it accepts keys this model does not have
        while they keep their declared default (``w_DE_0 = -1`` for a model
        without dark energy), and ignores keys no component declares
        (:meth:`validate_keys`). Use it for a dictionary written for another
        model. Under ``jit`` or a derivative such a key's value is traced
        and can't be checked. The solve reads none of it, so its derivative
        is zero.
        """

        cls.validate_keys(params)
        cosmology = cls(**{key: params[key] for key in cls._keys if key in params})
        held = {
            key: value
            for key, value in params.items()
            if key not in cls._keys and declarations_of(key) and not _is_concrete(value)
        }
        object.__setattr__(cosmology, "_held", held)
        return cosmology

    @classmethod
    def from_density_parameters(
        cls, *, Omega_m, Omega_b, H_0, Omega_k=0.0, **others
    ) -> Model:
        """Return the cosmology with matter fraction ``Omega_m``, baryon fraction ``Omega_b`` and Hubble constant ``H_0``.

        The model's own parameters are physical densities. This converts,
        ``omega_b = Omega_b h^2`` and ``omega_c = (Omega_m - Omega_b) h^2``, and
        sets ``omega_de`` so that the model expands at ``H_0`` (km/s/Mpc)
        today (:meth:`physical_parameters`). ``others`` are the
        model's remaining parameters (``T_cmb``, ``A_s``, ...). JAX values
        stay differentiable, so this is how to take a derivative in ``H_0``
        or ``Omega_m``. A model with no ``omega_de`` (a scalar field) has no
        such constructor.
        """

        return cls(
            **cls.physical_parameters(
                {"Omega_m": Omega_m, "Omega_b": Omega_b, "H_0": H_0, "Omega_k": Omega_k}
                | others
            )
        )

    @property
    def H_0(self):
        """The Hubble constant in km/s/Mpc, which the densities sum to (:meth:`hubble_constant`)."""

        return type(self).hubble_constant(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return the parameters by name, every one of the model's."""

        return {key: getattr(self, key) for key in type(self)._keys}

    def replace(self, **changes) -> Model:
        """Return this cosmology with ``changes`` to its parameters: ``planck.replace(H_0=70.0)``."""

        cosmology = type(self)(**{**self.to_dict(), **changes})
        object.__setattr__(cosmology, "_held", self._held)
        return cosmology

    __replace__ = replace

    def __setattr__(self, name, value):
        raise AttributeError(f"a cosmology is immutable: use replace({name}=...)")

    def __repr__(self) -> str:
        values = ", ".join(f"{k}={v!r}" for k, v in self.to_dict().items())
        return f"{type(self).__name__}({values})"

    # --- reading the model -----------------------------------------------------

    @classmethod
    def has(cls, component: type[Component]) -> bool:
        """Whether the model includes ``component`` or a subclass of it."""

        return any(issubclass(c, component) for c in cls.components)

    @classmethod
    def setting(cls, name: str):
        """Return a setting's value: the model's override or the component's own."""

        overrides = dict(cls.settings)
        if name in overrides:
            return overrides[name]
        for component in cls.components:
            if name in component.setting_names():
                return getattr(component, name)
        raise KeyError(f"{cls.name} has no setting {name!r}")

    @classmethod
    def resolved(cls) -> type[Model]:
        """Return this model with every setting explicit, at the components' current values.

        Models of one name, components and settings resolve to the same
        class, the first such model asked or a subclass of it, so the solver
        caches its kernels on it. Rebinding a component's setting before a
        solve resolves to another class and compiles a new kernel.
        """

        names = sorted({n for c in cls.components for n in c.setting_names()})
        values = {n: cls.setting(n) for n in names}
        key = (cls.name, cls.components, tuple(values.items()))
        if key not in _RESOLVED:
            explicit = dict(cls.settings) == values
            _RESOLVED[key] = cls if explicit else _derived(cls.name, (cls,), values)
        return _RESOLVED[key]

    @classmethod
    def instances(cls) -> list[Component]:
        """Return the components instantiated with the model's settings."""

        overrides = dict(cls.settings)
        return [
            c(**{n: overrides[n] for n in c.setting_names() if n in overrides})
            for c in cls.components
        ]

    @classmethod
    def build(
        cls,
        thermal_history: Component | None = None,
        extra=(),
        time: str = "tau",
        clock: str | None = None,
        last=(),
    ) -> Equations:
        """Return the component :class:`~discoeb.components._component.Equations` of this cosmological model.

        ``thermal_history`` provides ``a``, ``opacity``, ``cs2``, ``rhonu`` and
        ``pnu`` as functions of ``tau``. In the solve it is the recombination
        block of :func:`discoeb.solver._kernel.joint_kernel`, in a test anything.
        Without it the model still gives the state layout, the initial
        conditions and the sparsity pattern. ``extra`` lists component
        instances to append (the line-of-sight sources of DISCO-CMB).
        """

        components = cls.instances()
        if thermal_history is not None:
            components.insert(0, thermal_history)
        components.extend(extra)
        return Equations(components, time=time, clock=clock, last=last)

    @classmethod
    def parameters(cls, extra=()) -> dict[str, ParameterDeclaration]:
        """Return the parameters the model's components declare, by name.

        They include those of the recombination block every solve appends
        (:func:`discoeb.solver._kernel.joint_model`) and of the component classes
        ``extra``. They make up the cosmology dictionary
        (:meth:`parameter_keys`). Those a method running in the kernel
        reads are columns of its parameter row
        (:func:`discoeb.solver._kernel.parameter_layout`).
        """

        return declared_parameters((*cls.components, Recombination, *extra))

    @classmethod
    def parameter_keys(cls) -> dict[str, Any]:
        """Return the constructor's parameters and their defaults, :data:`REQUIRED` if none."""

        return {n: p.default for n, p in cls.parameters().items()}

    @classmethod
    def state_names(cls) -> tuple[str, ...]:
        """Return the perturbation state's element names in index order."""

        return _state_names(cls.resolved())

    @classmethod
    def sparsity(cls):
        """Return the ``(nnz, 2)`` index array of the perturbation Jacobian's nonzeros.

        modax's Rodas5P kernel uses the pattern twice. It colours the column
        intersection graph, so a Jacobian costs ``n_colours + 1`` Enzyme
        sweeps instead of ``nvar + 1``. It also orders the pattern,
        factorises it symbolically and compiles a sparse LU and a pair of
        sparse triangular solves for that exact structure.
        :meth:`Equations.sparsity <discoeb.components._component.Equations.sparsity>` traces the
        pattern through the equations, so the tridiagonal free-streaming
        tails and the Einstein constraints that tie the metric, the fluids
        and the lowest multipoles together come from the code that
        writes them. The pattern is exact. For flat LCDM at the default truncations it has 165
        entries, 11 colours and 198 slots of ``L + U`` after AMD's fill-in
        (``tests/test_models.py``).

        The kernel factorises ``I/(h gamma) - J``. modax keeps a slot for every
        diagonal and resolves ``slot(i, i)`` through that table even where
        ``J`` has no entry (an undamped free-streaming multipole has none).
        This needs modax at or after commit 9f54d12.
        """

        return cls.build().sparsity()

    @classmethod
    def validate(cls, cosmology) -> None:
        """Raise ``ValueError`` if ``cosmology`` sets parameters this model ignores.

        ``cosmology`` is a parameter dictionary or a cosmology. A key that another
        component declares but none of this model's must keep its declared
        default: a model without the dark-energy fluid needs ``w_DE_0 = -1``
        and ``w_DE_a = 0``, one without massive neutrinos ``N_mnu = 0``. Then
        each component's ``check`` hook runs on the keys it names
        (:meth:`DarkEnergy.check <discoeb.components.species.DarkEnergy.check>` needs
        ``w != -1``). A traced value, under ``jit``, has nothing to compare
        yet and is skipped. The method is not named ``check`` because a model
        inherits its components' ``check`` hooks.
        """

        values = _cosmology_dictionary(cosmology)
        cls.validate_keys(values)
        keys = cls.parameter_keys()
        filled = {**{k: v for k, v in keys.items() if v is not REQUIRED}, **values}
        for component in (*cls.components, Recombination):
            hook = getattr(component, "check", None)
            if hook is None:
                continue
            names = [
                component._renames.get(n, n) for n in inspect.signature(hook).parameters
            ]
            if any(n not in filled for n in names):
                continue  # a required key left out fails where it is read
            args = [filled[n] for n in names]
            if not all(map(_is_concrete, args)):
                continue  # traced under jit: nothing to compare yet
            try:
                hook(*args)
            except ValueError as e:
                raise ValueError(f"{cls.name}: {e}") from None

    @classmethod
    def validate_keys(cls, cosmology) -> None:
        """Raise ``ValueError`` if ``cosmology`` sets a parameter of a component this model lacks.

        The first half of :meth:`validate`: a key that another component
        declares and none of this model's must keep its declared default, or
        the solve would silently ignore it.
        """

        values = _cosmology_dictionary(cosmology)
        parameters = cls.parameters()
        for key, value in values.items():
            if key in parameters or not _is_concrete(value):
                continue
            owners = declarations_of(key)
            if not owners:
                continue  # a key no component declares
            default = owners[0][1].default
            if default is REQUIRED or float(value) != default:
                names = ", ".join(sorted({c.__name__ for c, _ in owners}))
                raise ValueError(
                    f"{cls.name} has no component with the parameter {key} "
                    f"(set to {value}, default {default}); use a model with {names}"
                )

    @classmethod
    def hubble_constant(cls, cosmology):
        """Return ``H_0`` in km/s/Mpc: the model's expansion rate today, ``aH(a = 1)``.

        The densities are physical and determine ``H_0``. A model
        whose expansion rate reads a state of its own (a scalar field) has
        it only from a solve, as ``aH`` today, and raises ``TypeError``.
        """

        if _expansion_reads_state(cls):
            raise TypeError(
                f"{cls.name}: the expansion rate reads a state of the model, "
                "so its value today needs a solve: keep 'aH' today"
            )
        values = _cosmology_dictionary(cosmology)
        module = math if all(map(_is_number, values.values())) else jnp
        aH = _host_equations(cls).evaluate(
            "aH",
            math=module,
            a=1.0,
            **parameter_values(cls.parameters(), values),
        )
        return aH * (100.0 / H100)

    @classmethod
    def physical_parameters(cls, params) -> dict:
        """Return a dictionary in ``Omega_m``, ``Omega_b``, ``H_0`` as one in the model's own parameters.

        ``omega_b = Omega_b h^2``, ``omega_c = (Omega_m - Omega_b) h^2``,
        ``omega_k = Omega_k h^2`` (``Omega_k`` defaults to 0), any other
        component's ``x_omega = x_Omega h^2``, and ``omega_de`` the
        dark-energy density
        that makes the model expand at ``H_0`` today. The expansion rate
        squared is linear in a density, so one secant step from two
        evaluations finds it exactly. Every other key passes through. It is plain
        JAX arithmetic, so a derivative in ``H_0`` or ``Omega_m`` goes through it.
        """

        given = dict(params)
        keys = cls.parameter_keys()
        if "omega_de" not in keys:
            raise TypeError(
                f"{cls.name} has no dark-energy density to close the budget "
                "with; give its own parameters"
            )
        if "omega_de" in given:
            raise TypeError("give omega_de or H_0, not both")
        try:
            Omega_m, Omega_b, H_0 = (
                given.pop(k) for k in ("Omega_m", "Omega_b", "H_0")
            )
        except KeyError as e:
            raise TypeError(f"from density parameters: missing {e}") from None
        h2 = (H_0 / 100.0) ** 2
        native = {
            "omega_b": Omega_b * h2,
            "omega_c": (Omega_m - Omega_b) * h2,
            "omega_k": given.pop("Omega_k", 0.0) * h2,
        }
        for key in list(given):
            physical = key.replace("Omega", "omega")
            if physical != key and declarations_of(physical):
                native[physical] = given.pop(key) * h2
        native = {**given, **native}

        def miss(omega_de):
            return cls.hubble_constant({**native, "omega_de": omega_de}) ** 2 - H_0**2

        none, all_of_it = miss(0.0 * h2), miss(h2)
        native["omega_de"] = -none * h2 / (all_of_it - none)
        return native

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

        if old not in cls.components:
            raise ValueError(f"{cls.name} has no {old.__name__}")
        components = tuple(new if c is old else c for c in cls.components)
        return _derived(name, (Model, *components), cls._settings_for(components))

    @classmethod
    def remove(cls, *components: type[Component], name: str) -> type[Model]:
        """Return a model without ``components``."""

        missing = [c.__name__ for c in components if c not in cls.components]
        if missing:
            raise ValueError(f"{cls.name} has no {missing}")
        kept = tuple(c for c in cls.components if c not in components)
        return _derived(name, (Model, *kept), cls._settings_for(kept))

    @classmethod
    def with_settings(cls, **settings) -> type[Model]:
        """Return this model with the given component settings (``LMAX_G=20``), a subclass."""

        if not settings:
            return cls
        return _derived(cls.name, (cls,), settings)

    @classmethod
    def _settings_for(cls, components) -> dict[str, object]:
        known = {n for c in components for n in c.setting_names()}
        return {n: v for n, v in cls.settings if n in known}


_DERIVED: dict[tuple, type[Model]] = {}
"""The models derived at run time, one class per name, bases and settings."""

_RESOLVED: dict[tuple, type[Model]] = {}
"""The resolved models, one class per name, components and settings."""


def _derived(name: str, bases: tuple, settings: dict) -> type[Model]:
    key = (name, bases, tuple(sorted(settings.items())))
    if key not in _DERIVED:
        _DERIVED[key] = type(
            name, bases, {"__module__": __name__}, name=name, **settings
        )
    return _DERIVED[key]


@functools.lru_cache(maxsize=None)
def _host_equations(model: type[Model]) -> Equations:
    """The model's equations for evaluating its providers on the host."""

    return model.build()


@functools.lru_cache(maxsize=None)
def _expansion_reads_state(model: type[Model]) -> bool:
    return _host_equations(model).reads_state("aH")


@functools.lru_cache(maxsize=None)
def _state_names(model: type[Model]) -> tuple[str, ...]:
    return tuple(model.build().state)


def _is_concrete(value) -> bool:
    """Whether ``value`` is a number now, not a JAX tracer of ``jit`` or a derivative."""

    return not isinstance(value, jax.core.Tracer)


def _is_number(value) -> bool:
    """Whether ``value`` is a plain number, not a JAX or numpy array."""

    return isinstance(value, (int, float))


def _cosmology_dictionary(cosmology) -> dict:
    """Return the parameters of a cosmology or of a parameter dictionary."""

    if isinstance(cosmology, dict):
        return dict(cosmology)
    return cosmology.to_dict()


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
    lines = ["Parameters", "----------"]
    for key, default in cls._keys.items():
        line = "Required." if default is REQUIRED else f"Default {default!r}."
        lines += [f"{key} : float or jax.Array", f"    {line}"]
    return "\n\n".join(part for part in (own, "\n".join(lines)) if part)


# --- what a solve takes ---------------------------------------------------------


def _batch_of(cosmologies) -> tuple[type[Model], tuple[dict, ...]]:
    """Return one cosmology or a sequence of one model's as ``(model, dicts)``.

    ``model`` is their class and ``dicts`` the cosmologies' parameter
    dictionaries (their values numbers, JAX arrays or tracers).
    """

    batch = (cosmologies,) if isinstance(cosmologies, Model) else tuple(cosmologies)
    if not batch or not all(isinstance(c, Model) for c in batch):
        raise TypeError(
            "cosmologies must be a cosmology, such as LCDM(omega_b=..., ...), or a "
            f"non-empty sequence of them, not {type(cosmologies).__name__}"
        )
    classes = list(dict.fromkeys(type(c) for c in batch))
    if len(classes) > 1:
        raise TypeError(
            "the cosmologies of one solve share a model, whose kernel it compiles: "
            f"got {', '.join(c.__name__ for c in classes)}"
        )
    return classes[0], tuple({**c._held, **c.to_dict()} for c in batch)


def _model_of(model) -> type[Model]:
    """Return ``model`` if it is a model class such as ``LCDM``, else raise ``TypeError``."""

    if isinstance(model, type) and issubclass(model, Model) and model is not Model:
        return model
    raise TypeError(f"expected a model such as LCDM, not {model!r}")
