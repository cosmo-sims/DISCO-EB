"""Cosmological models built from the species, interactions and gravity components.

A model lists the components it evolves (:class:`CosmologyModel`). Called on
cosmologies (parameter dictionaries) it gives the batch every solve takes
(:class:`CosmologyBatch`): ``solve_eb(LCDM([params]), k, tau)``. Models
derive from each other, so a new one is usually a line::

    W0WACDM = LCDM.replace(CosmologicalConstant, DarkEnergy, name="w0waCDM")
    LCDM_MNU = LCDM.add(MassiveNeutrinos, name="LCDM+mnu")
    COARSE = LCDM.with_settings(LMAX_G=10, LMAX_NR=10)
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field

from .component import (
    REQUIRED,
    Component,
    Model,
    Parameter,
    declarations_of,
    declared_parameters,
)
from .cosmology import _as_batch, _cosmology_dictionary, _is_concrete
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
    "CosmologyBatch",
    "CosmologyModel",
]


@dataclass(frozen=True)
class CosmologyModel:
    """The components of a cosmological model, in state order, and their settings.

    A model is a list of component classes: the species it evolves, the
    interactions between them and the gravity components every model needs.
    Their order is the state order. A cosmology (a parameter dictionary) is a
    point in its parameter space, and :meth:`check` rejects one the model
    cannot represent.

    Models derive from each other. :meth:`replace` swaps one component for
    another, :meth:`add` appends components and :meth:`with_settings`
    changes the hierarchy truncations, so ``LCDM.add(MassiveNeutrinos,
    name="LCDM+mnu")`` is LCDM with massive neutrinos. A model is hashable and
    the solver caches its compiled kernel per model.

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
        return f"CosmologyModel({self.name!r})"

    # --- deriving models ------------------------------------------------------

    def replace(
        self, old: type[Component], new: type[Component], *, name: str
    ) -> CosmologyModel:
        """Return this model with ``new`` in place of ``old``, at its position."""

        if old not in self.components:
            raise ValueError(f"{self.name} has no {old.__name__}")
        components = tuple(new if c is old else c for c in self.components)
        return CosmologyModel(name, components, self._settings_for(components))

    def add(self, *components: type[Component], name: str) -> CosmologyModel:
        """Return this model with ``components`` appended."""

        return CosmologyModel(name, self.components + components, self.settings)

    def remove(self, *components: type[Component], name: str) -> CosmologyModel:
        """Return this model without ``components``."""

        missing = [c.__name__ for c in components if c not in self.components]
        if missing:
            raise ValueError(f"{self.name} has no {missing}")
        kept = tuple(c for c in self.components if c not in components)
        return CosmologyModel(name, kept, self._settings_for(kept))

    def with_settings(self, **settings) -> CosmologyModel:
        """Return this model with the given component settings (``LMAX_G=20``)."""

        merged = {**dict(self.settings), **settings}
        return CosmologyModel(self.name, self.components, tuple(merged.items()))

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

    def resolved(self) -> CosmologyModel:
        """Return this model with every setting explicit, at the components' current values.

        The solver caches its kernels on the resolved model, so rebinding a
        component's setting before a solve compiles a new kernel.
        """

        names = {n for cls in self.components for n in cls.setting_names()}
        values = tuple((n, self.setting(n)) for n in sorted(names))
        return CosmologyModel(self.name, self.components, values)

    def instances(self) -> list[Component]:
        """Return the components instantiated with the model's settings."""

        overrides = dict(self.settings)
        return [
            cls(**{n: overrides[n] for n in cls.setting_names() if n in overrides})
            for cls in self.components
        ]

    def build(self, thermal_history: Component | None = None, extra=()) -> Model:
        """Return the component :class:`~discoeb.component.Model` of this cosmological model.

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
        return Model(components, sums=SUMS)

    def parameters(self, extra=()) -> dict[str, Parameter]:
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
        :meth:`Model.sparsity <discoeb.component.Model.sparsity>` traces the
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

    def __call__(self, cosmologies) -> CosmologyBatch:
        """Return this model with ``cosmologies`` to solve: ``LCDM([params, ...])``.

        ``cosmologies`` is one cosmology (a parameter dictionary or a cosmology
        object) or a sequence of them. Each is checked against the model
        (:meth:`check`). Dictionaries are kept as given, so the values of
        :data:`~discoeb.DIFFERENTIABLE_PARAMETERS` in them may be JAX values.
        """

        batch = _as_batch(cosmologies)
        if not batch:
            raise ValueError(f"{self.name} needs at least one cosmology")
        for cosmology in batch:
            self.check(cosmology)
        return CosmologyBatch(self, batch)

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


@dataclass(frozen=True, eq=False)
class CosmologyBatch:
    """A model with the cosmologies to solve it for, as ``LCDM([params, ...])`` returns.

    Every solve takes one: ``solve_eb(LCDM(params), k, tau)``.
    ``model`` is the :class:`CosmologyModel`, on which the compiled kernel is
    cached, and ``cosmologies`` the tuple of cosmologies as given. Indexing
    gives a batch of the same model: ``batch[0]`` holds the first cosmology,
    ``batch[2:5]`` three.
    """

    model: CosmologyModel
    cosmologies: tuple

    def __len__(self) -> int:
        return len(self.cosmologies)

    def __getitem__(self, index) -> CosmologyBatch:
        selected = self.cosmologies[index]
        if isinstance(index, slice):
            return CosmologyBatch(self.model, selected)
        return CosmologyBatch(self.model, (selected,))

    def __repr__(self) -> str:
        return f"{self.model.name}(<{len(self)} cosmologies>)"


def _batch_of(batch, what: str = "cosmologies") -> CosmologyBatch:
    """Return ``batch`` if it is a :class:`CosmologyBatch`, else raise a ``TypeError`` that says how to make one."""

    if isinstance(batch, CosmologyBatch):
        return batch
    raise TypeError(
        f"{what} must be a model applied to parameter dictionaries, such as "
        f"LCDM([params, ...]), not {type(batch).__name__}"
    )


_STATE_NAMES: dict = {}


def _state_names(model: CosmologyModel) -> tuple[str, ...]:
    if model not in _STATE_NAMES:
        _STATE_NAMES[model] = tuple(model.build().state)
    return _STATE_NAMES[model]


# --- the models -------------------------------------------------------------

LCDM = CosmologyModel(
    "LCDM",
    (
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
    ),
)
"""LCDM with massless neutrinos: CDM, baryons, photons coupled by Thomson
scattering, ``Neff`` massless neutrino flavours and a cosmological constant.
Flat unless the cosmology sets ``Omegak``."""

LCDM_MNU = LCDM.add(MassiveNeutrinos, name="LCDM+mnu")
"""LCDM with ``Nmnu`` massive neutrino flavours of mass ``mnu`` on ``NQMAX``
momentum bins."""

W0WACDM = LCDM.replace(CosmologicalConstant, DarkEnergy, name="w0waCDM")
"""LCDM with a dark-energy fluid of equation of state ``w = w_DE_0 + w_DE_a (1 -
a)`` and sound speed ``cs2_DE``, perturbed. Needs ``w != -1``."""

W0WACDM_MNU = W0WACDM.add(MassiveNeutrinos, name="w0waCDM+mnu")
"""w0waCDM with massive neutrinos."""

DEFAULT_MODEL = W0WACDM_MNU
"""The model a bare cosmology stands for where no batch names one.

:func:`~discoeb.background.density_coefficients`,
:func:`~discoeb.background.background_history`,
:func:`~discoeb.solver.conformal_ages` and
:func:`~discoeb.solver.directed_inputs` take a parameter dictionary or a
cosmology object as well as a batch. This model's components declare every
built-in parameter, and on a cosmology another built-in model accepts its
background is that model's. A cosmology that sets a parameter of a component
outside it (one added to a model of your own) raises there
(:meth:`CosmologyModel.check_keys`): pass the batch, ``MODEL(cosmology)``."""
