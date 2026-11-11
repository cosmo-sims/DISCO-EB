"""The Einstein-Boltzmann and CMB parts of the package meet only at :mod:`discoeb.eb`.

The CMB part (:mod:`discoeb.cmb`, :mod:`discoeb.lensing`) may import from
the Einstein-Boltzmann interface :mod:`discoeb.eb`, the generic numerics of
:mod:`discoeb.spline_interpolation` and each other; the Einstein-Boltzmann
part (everything else) may not import the CMB part at all. Checked on the
modules' import statements, so a private reach across the boundary --
``from .perturbations import _launch`` -- fails here rather than in review.
"""

import ast
from pathlib import Path

import discoeb

PACKAGE = Path(discoeb.__file__).parent
CMB_PART = {"cmb", "lensing"}
CMB_MAY_IMPORT = CMB_PART | {"eb", "spline_interpolation"}


def _imported_modules(module: str) -> set[str]:
    """The ``discoeb`` submodules a module imports, at any depth of its code."""

    tree = ast.parse((PACKAGE / f"{module}.py").read_text())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 1 and node.module:
                found.add(node.module.split(".")[0])
            elif node.level == 1:
                found.update(alias.name for alias in node.names)
            elif node.module and node.module.startswith("discoeb."):
                found.add(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("discoeb."):
                    found.add(alias.name.split(".")[1])
    return found


def _modules() -> list[str]:
    return sorted(p.stem for p in PACKAGE.glob("*.py") if p.stem != "__init__")


def test_the_cmb_part_uses_the_einstein_boltzmann_interface_only():
    for module in CMB_PART:
        outside = _imported_modules(module) - CMB_MAY_IMPORT
        assert not outside, (
            f"discoeb.{module} imports {sorted(outside)} past discoeb.eb"
        )


def test_the_einstein_boltzmann_part_does_not_import_the_cmb_part():
    for module in set(_modules()) - CMB_PART:
        crossing = _imported_modules(module) & CMB_PART
        assert not crossing, (
            f"discoeb.{module} imports the CMB part: {sorted(crossing)}"
        )


def test_the_interface_exports_what_it_lists():
    from discoeb import eb

    assert all(hasattr(eb, name) for name in eb.__all__)
