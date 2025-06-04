import pytest
from frust.embedder import embed_multiple_mol, generate_conformers
from rdkit import Chem

def test_embed_multiple_mol_conformers():
    mol = embed_multiple_mol("C1=CC=CO1", num_confs=5)
    assert isinstance(mol, Chem.Mol)
    assert mol.GetNumConformers() == 5

def test_generate_conformers_dict_keys_and_counts():
    smi_dict = {"furan": "C1=CC=CO1", "methane": "C"}
    confs = generate_conformers(smi_dict, num_of_conf=3)
    assert set(confs.keys()) == set(smi_dict.keys())
    for name, mol in confs.items():
        assert isinstance(mol, Chem.Mol)
        assert mol.GetNumConformers() == 3

def test_generate_conformers_default_number(tmp_path, monkeypatch):
    # Simulate CalcNumRotatableBonds = 1 for a given SMILES
    from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
    monkeypatch.setattr("frustactivation.embedder.CalcNumRotatableBonds", lambda m: 1)
    smi_dict = {"test": "CC"}  # two-carbon chain
    confs = generate_conformers(smi_dict, num_of_conf=None)
    # default = 5 + 10*1 = 15
    mol = confs["test"]
    assert mol.GetNumConformers() == 15