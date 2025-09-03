from pathlib import Path
from tooltoad.chemutils import xyz2ac    

f = Path("structures/ts4_TMP.xyz")
mols = {}
with open(f, "r") as file:
    xyz_block = file.read()
    # mol = xyz2mol(xyz_block)
    # mols[f.stem] = (mol, [0])
    atoms, coords = xyz2ac(xyz_block)

from tooltoad.orca import orca_calculate
orca_calculate(atoms, coords, options={"XTB2": None, "SP": None}, calc_dir="noob")