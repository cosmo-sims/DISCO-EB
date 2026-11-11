"""The Einstein-Boltzmann interface :mod:`discoeb.eb`, which DISCO-CMB builds on.

DISCO-CMB (the ``discocmb`` package) imports from :mod:`discoeb.eb` and
:mod:`discoeb.spline_interpolation` alone and checks so itself; here the
interface must export what it lists, and no module of this package may
import DISCO-CMB.
"""

import ast
from pathlib import Path

import discoeb

PACKAGE = Path(discoeb.__file__).parent


def test_the_interface_exports_what_it_lists():
    from discoeb import eb

    assert all(hasattr(eb, name) for name in eb.__all__)


def test_the_package_does_not_import_discocmb():
    for path in sorted(PACKAGE.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                continue
            assert not any(n.split(".")[0] == "discocmb" for n in names), (
                f"discoeb.{path.stem} imports DISCO-CMB"
            )
