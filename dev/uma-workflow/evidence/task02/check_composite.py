"""Check UMA + GFN2-xTB ALPB(chloroform) against central differences.

Run with the installed OET virtualenv Python. This uses one UmaCalc instance so
that the model is loaded once, while every xTB pair receives fresh scratch.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from oet.calculator.uma import DEFAULT_CACHE_DIR, UmaCalc
from oet.core.base_calc import CalculationData
from oet.core.misc import LENGTH_CONVERSION

ROOT = Path(__file__).resolve().parent
XTB = "/opt/homebrew/Caskroom/miniconda/base/envs/UMA/bin/xtb"
MODEL = "uma-s-1p2p1"
STEP_ANGSTROM = 0.001
ATOMS = ["O", "H", "H"]
COORDS = [
    [0.0, 0.0, 0.0],
    [0.758602, 0.0, 0.504284],
    [-0.758602, 0.0, 0.504284],
]


def evaluate(
    calc: UmaCalc, work_root: Path, coords: list[list[float]], gradient: bool
) -> tuple[float, list[float]]:
    """Evaluate the composite potential for one displaced geometry.

    Parameters
    ----------
    calc : UmaCalc
        Calculator retaining the loaded UMA model.
    work_root : Path
        Temporary parent for the ORCA-style request.
    coords : list[list[float]]
        Cartesian coordinates in Angstrom.
    gradient : bool
        Whether to request the analytical composite gradient.

    Returns
    -------
    tuple[float, list[float]]
        Energy in Hartree and gradient in Hartree/Bohr.
    """
    folder = work_root / f"case_{len(list(work_root.iterdir())):03d}"
    folder.mkdir()
    xyz = folder / "water.xyz"
    xyz.write_text(
        "3\nwater finite difference\n"
        + "".join(
            f"{atom} {position[0]:.9f} {position[1]:.9f} {position[2]:.9f}\n"
            for atom, position in zip(ATOMS, coords, strict=True)
        ),
        encoding="utf-8",
    )
    ext = folder / "water.ext"
    ext.write_text(f"water.xyz\n0\n1\n1\n{int(gradient)}\n", encoding="utf-8")
    request = CalculationData(str(ext), None)
    try:
        return calc.calc(
            request,
            {
                "param": "omol",
                "basemodel": MODEL,
                "device": "cpu",
                "cache_dir": DEFAULT_CACHE_DIR,
                "offline_mode": True,
                "xtb_alpb": "chloroform",
                "xtb_exe": XTB,
                "inference_settings": "batch",
            },
            [],
        )
    finally:
        request.remove_tmp()


def main() -> None:
    """Write a component-wise finite-difference comparison."""
    calc = UmaCalc()
    with tempfile.TemporaryDirectory(prefix="uma-fd-") as tmp:
        work_root = Path(tmp)
        energy, analytical = evaluate(calc, work_root, COORDS, gradient=True)
        numerical: list[float] = []
        for atom_index in range(len(ATOMS)):
            for axis in range(3):
                plus = [position.copy() for position in COORDS]
                minus = [position.copy() for position in COORDS]
                plus[atom_index][axis] += STEP_ANGSTROM
                minus[atom_index][axis] -= STEP_ANGSTROM
                e_plus, _ = evaluate(calc, work_root, plus, gradient=False)
                e_minus, _ = evaluate(calc, work_root, minus, gradient=False)
                numerical.append(
                    (e_plus - e_minus) / (2 * STEP_ANGSTROM) * LENGTH_CONVERSION["Ang"]
                )
    errors = [abs(a - n) for a, n in zip(analytical, numerical, strict=True)]
    result = {
        "model": MODEL,
        "inference_settings": "batch",
        "xtb_correction": "GFN2-xTB ALPB(chloroform) - gas",
        "step_angstrom": STEP_ANGSTROM,
        "energy_Eh": energy,
        "analytic_gradient_Eh_per_bohr": analytical,
        "finite_difference_gradient_Eh_per_bohr": numerical,
        "absolute_errors_Eh_per_bohr": errors,
        "max_absolute_error_Eh_per_bohr": max(errors),
    }
    (ROOT / "finite_difference.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"energy={energy:.12f} Eh")
    print(f"max |gradient - finite difference|={max(errors):.8g} Eh/Bohr")


if __name__ == "__main__":
    main()
