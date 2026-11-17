"""Tests of the :mod:`discoeb` interface."""

import ast
from pathlib import Path

import discoeb

# --- The Einstein-Boltzmann interface :mod:`discoeb`, which DISCO-CMB builds on.


PACKAGE = Path(discoeb.__file__).parent


def test_the_interface_exports_what_it_lists():
    assert all(hasattr(discoeb, name) for name in discoeb.__all__)


def test_the_package_does_not_import_discocmb():
    for path in sorted(PACKAGE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                continue
            assert not any(n.split(".")[0] == "discocmb" for n in names), (
                f"{path.relative_to(PACKAGE)} imports DISCO-CMB"
            )
