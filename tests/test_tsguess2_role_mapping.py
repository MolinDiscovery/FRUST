import pandas as pd
from rdkit import Chem

import frust as ft


def test_hbpin_roles_are_chemically_directed_for_atom5_substitution():
    systems = pd.DataFrame(
        [
            {
                "system_name": "dimethoxybenzene_1_3__cat_SO2Me_atom_5",
                "substrate_name": "dimethoxybenzene_1_3",
                "catalyst_name": "cat_SO2Me_atom_5",
                "substrate_smiles": "COC1=CC=CC(OC)=C1",
                "catalyst_smiles": "CN(C)c1ccccc1BS(C)(=O)=O",
                "rpos": 3,
            }
        ]
    )

    guesses = ft.tsguess2.create_ts_guess_dataframes(
        systems,
        ts_types=["TS3", "TS4", "INT3"],
        n_confs=1,
        spec_profile="r2scan3c",
        spec_match="exact",
    )

    for state, rows in guesses.items():
        row = rows.iloc[0]
        roles = row["constraint_roles"]
        mol = Chem.AddHs(Chem.MolFromSmiles(row["smiles"]))
        cat_b = roles["cat_B"]
        pin_b = roles["pin_B"]

        assert sum(
            neighbor.GetAtomicNum() == 8
            for neighbor in mol.GetAtomWithIdx(pin_b).GetNeighbors()
        ) == 2
        assert sum(
            neighbor.GetAtomicNum() == 8
            for neighbor in mol.GetAtomWithIdx(cat_b).GetNeighbors()
        ) < 2

        for role, atomic_number in (("transfer_H", 1), ("substrate_C", 6)):
            role_idx = roles[role]
            assert mol.GetAtomWithIdx(role_idx).GetAtomicNum() == atomic_number
            assert mol.GetBondBetweenAtoms(cat_b, role_idx) is not None
            assert mol.GetBondBetweenAtoms(pin_b, role_idx) is not None

        spec = ft.tsguess2.resolve_profile_spec(
            state,
            "r2scan3c",
            match="exact",
        ).spec
        query = Chem.MolFromSmarts(spec.core_smarts)
        assert mol.GetSubstructMatches(query) == (
            (
                roles["cat_B"],
                roles["transfer_H"],
                roles["pin_B"],
                roles["substrate_C"],
            ),
        )
