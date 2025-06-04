import pytest
from frust.transformer_mols import generate_smirks_skeleton, transform_mols
from rdkit import Chem

def test_smirks_skeleton_basic():
    smi = "CCO"
    skeleton = generate_smirks_skeleton(smi, start_map=5)
    # should label C atoms 5 & 6, O atom 7
    assert "[C:5]" in skeleton
    assert "[C:6]" in skeleton
    assert "[O:7]" in skeleton

def test_transform_mols_keys_and_smiles_valid():
    out = transform_mols("C1=CC=CO1", "CCO", rpos=1)
    # expect at least these keys (depending on your implementation)
    expected = {"dimer", "ligand", "catalyst", "int2", "mol2", "HBpin-ligand"}
    assert expected.issubset(out.keys())
    for key, smi in out.items():
        assert isinstance(smi, str)
        # parsed SMILES must be valid
        assert Chem.MolFromSmiles(smi) is not None

def test_transform_mols_rpos_affects_output():
    # two different rpos should give at least one differing SMILES
    out1 = transform_mols("C1=CC=CO1", "CCO", rpos=1)
    out2 = transform_mols("C1=CC=CO1", "CCO", rpos=2)
    # the intermediate or TS-like keys should differ
    assert out1 != out2