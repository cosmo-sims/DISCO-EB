"""A type checker reads a model's parameters off its components' declarations.

The models are classes whose bases are their components, and
``dataclass_transform`` on ``Component`` makes the annotated
``Parameter`` declarations the constructor's keyword arguments. This runs
ty on code written as a user would write it and checks it reports exactly
the calls the constructor rejects at run time.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

USER_CODE = '''
from discoeb import LCDM, W0WACDM
from discoeb.components.species import MassiveNeutrinos

good = LCDM(
    Omegam=0.31, Omegab=0.049, A_s=2.1e-9, n_s=0.966, H0=67.0,
    Tcmb=2.7255, YHe=0.245, Neff=3.046,
)
dark = W0WACDM(
    Omegam=0.31, Omegab=0.049, A_s=2.1e-9, n_s=0.966, H0=67.0,
    Tcmb=2.7255, YHe=0.245, Neff=3.046, w_DE_0=-0.9,
)
moved = good.replace(H0=70.0)


class Mine(LCDM, MassiveNeutrinos):
    """A model of the user's: LCDM with massive neutrinos."""


mine = Mine(
    Omegam=0.31, Omegab=0.049, A_s=2.1e-9, n_s=0.966, H0=67.0,
    Tcmb=2.7255, YHe=0.245, Neff=3.046, mnu=0.06,
)

LCDM(  # missing
    Omegab=0.049, A_s=2.1e-9, n_s=0.966, H0=67.0,
    Tcmb=2.7255, YHe=0.245, Neff=3.046,
)
LCDM(  # foreign
    Omegam=0.31, Omegab=0.049, A_s=2.1e-9, n_s=0.966, H0=67.0,
    Tcmb=2.7255, YHe=0.245, Neff=3.046, mnu=0.06,
)
LCDM(  # derived
    Omegam=0.31, Omegab=0.049, A_s=2.1e-9, n_s=0.966, H0=67.0,
    Tcmb=2.7255, YHe=0.245, Neff=3.046, grhom=1.0,
)
LCDM(0.31)  # positional
'''


def _ty() -> str | None:
    beside = Path(sys.executable).parent / "ty"
    return str(beside) if beside.exists() else shutil.which("ty")


@pytest.mark.skipif(_ty() is None, reason="needs ty (the dev group)")
def test_a_type_checker_sees_the_models_parameters(tmp_path):
    code = tmp_path / "user.py"
    code.write_text(USER_CODE)
    result = subprocess.run(
        [
            str(_ty()),
            "check",
            "--python",
            sys.executable,
            "--output-format",
            "concise",
            str(code),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    errors = [line for line in result.stdout.splitlines() if "error[" in line]
    lines = USER_CODE.splitlines()

    def line_of(marker: str) -> int:
        return next(i for i, text in enumerate(lines, 1) if marker in text)

    found = {(int(e.split(":")[1]), e.split("error[")[1].split("]")[0]) for e in errors}
    assert found >= {
        (line_of("# missing"), "missing-argument"),
        (line_of("# positional"), "too-many-positional-arguments"),
    }, result.stdout
    rows = {row for row, _ in found}
    for marker in ("# foreign", "# derived"):
        start = line_of(marker)
        assert any(start <= row <= start + 3 for row in rows), (marker, result.stdout)
    # and nothing else: the valid calls and the user's model class pass
    assert all(row >= line_of("# missing") for row in rows), result.stdout
