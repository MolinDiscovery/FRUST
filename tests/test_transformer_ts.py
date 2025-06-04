import pytest
from frust.transformer_ts import TSTransformer
from rdkit import Chem

# Minimal XYZ for water
WATER_XYZ = """3
water
O 0.000000 0.000000 0.000000
H 0.757000 0.586000 0.000000
H -0.757000 0.586000 0.000000
"""

@pytest.fixture
def ts_xyz(tmp_path):
    p = tmp_path / "ts_guess.xyz"
    p.write_text(WATER_XYZ)
    return str(p)

def test_ts_transformer_returns_dict(ts_xyz):
    result = TSTransformer(
        ligand_smiles="O",             # simple ligand
        ts_guess_struct=ts_xyz,
        bonds_to_remove=[(0,1)],       # remove O–H
        num_confs=2
    )
    assert isinstance(result, dict)
    assert len(result) > 0

def test_ts_transformer_structure_and_energies(ts_xyz):
    result = TSTransformer(
        ligand_smiles="O",
        ts_guess_struct=ts_xyz,
        bonds_to_remove=[(0,1)],
        num_confs=2
    )
    for name, (mol_with_confs, energies, idxs) in result.items():
        # mol type
        assert isinstance(mol_with_confs, Chem.Mol)
        assert mol_with_confs.GetNumConformers() == 2
        # energies as list of (float,int)
        assert isinstance(energies, list) and len(energies) == 2
        for e, cid in energies:
            assert isinstance(e, float)
            assert isinstance(cid, int)
        # idxs list of ints
        assert isinstance(idxs, list)
        assert all(isinstance(i, int) for i in idxs)