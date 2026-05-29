from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from frust.cluster import ClusterConfig, submit_chain, submit_screen_chain
from frust.cluster.inputs import prepare_screen_chain_inputs


CATALYST = "CC1(C)CCCC(C)(C)N1C2=CC=CC=C2B"


def _screen_csv(path: Path, *, rpos: str = "2,3") -> Path:
    df = pd.DataFrame(
        {
            "role": ["substrate", "catalyst"],
            "smiles": ["CN1C=CC=C1", CATALYST],
            "compound_name": ["pyrrole", "cat_a"],
            "rpos": [rpos, None],
        }
    )
    df.to_csv(path, index=False)
    return path


def _screen_target() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "system_name": ["pyrrole__cat_a"],
            "substrate_name": ["pyrrole"],
            "catalyst_name": ["cat_a"],
            "substrate_smiles": ["CN1C=CC=C1"],
            "catalyst_smiles": [CATALYST],
            "smiles": ["CN1C=CC=C1"],
            "rpos": [2],
            "ts_type": ["TS1"],
        }
    )


def _seed_ts_guess_df() -> pd.DataFrame:
    rows = []
    for cid, energy in enumerate([3.0, 1.0, 2.0]):
        rows.append(
            {
                "custom_name": "TS1(pyrrole__cat_a_rpos(2))",
                "structure_id": "TS1:pyrrole__cat_a:r2",
                "system_name": "pyrrole__cat_a",
                "substrate_name": "pyrrole",
                "catalyst_name": "cat_a",
                "structure_type": "TS1",
                "molecule_role": "ts",
                "rpos": 2,
                "cid": cid,
                "atoms": ["H", "H", "H"],
                "coords_embedded": [
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (1.0, 1.0, 0.0),
                ],
                "constraint_roles": {"a": 0, "b": 1},
                "constraint_spec": [
                    {"kind": "distance", "roles": ["a", "b"], "value": 1.0}
                ],
                "constraint_atoms": [0, 1, 2, 0, 1, 2],
                "seed-EE": energy,
            }
        )
    return pd.DataFrame(rows)


class FakeExecutor:
    def __init__(self):
        self.parameters = []
        self.submissions = []

    def update_parameters(self, **kwargs):
        self.parameters.append(kwargs)

    def submit(self, fn, **kwargs):
        self.submissions.append((fn, kwargs))
        return SimpleNamespace(job_id=f"job-{len(self.submissions)}")


class FakeStepper:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        FakeStepper.instances.append(self)

    def xtb(self, df, name, options, constraint=False, lowest=None, n_cores=None):
        self.calls.append(("xtb", name, options, constraint, lowest, n_cores))
        out = df.copy()
        if lowest:
            out = out.head(lowest).copy()
        out[f"{name}-NT"] = True
        out[f"{name}-EE"] = range(len(out))
        out[f"{name}-oc"] = out["coords_embedded"]
        return out

    def orca(self, df, name, options, constraint=False, lowest=None, **kwargs):
        self.calls.append(("orca", name, options, constraint, lowest, kwargs))
        out = df.copy()
        if lowest:
            out = out.head(lowest).copy()
        out[f"{name}-NT"] = True
        out[f"{name}-EE"] = range(len(out))
        out[f"{name}-oc"] = out["coords_embedded"]
        return out


class ScreenChainInputTests(unittest.TestCase):
    def test_prepare_screen_chain_inputs_fans_out_without_embedding(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = _screen_csv(Path(tmp) / "screen.csv")
            with patch("frust.screen.create_ts_guesses") as create_ts_guesses:
                prepared = prepare_screen_chain_inputs(csv_path, ts_types=["TS1", "TS4"])

        create_ts_guesses.assert_not_called()
        self.assertEqual(prepared["mode"], "screen_ts_per_rpos")
        self.assertEqual(prepared["run_init_arg"], "screen_target")
        self.assertEqual(
            prepared["tags"],
            [
                "TS1__pyrrole__cat_a__r2",
                "TS1__pyrrole__cat_a__r3",
                "TS4__pyrrole__cat_a__r2",
                "TS4__pyrrole__cat_a__r3",
            ],
        )
        self.assertEqual(len(prepared["payloads"]), 4)
        for payload in prepared["payloads"]:
            self.assertEqual(len(payload), 1)
            self.assertIn(payload["ts_type"].iloc[0], {"TS1", "TS4"})
            self.assertIn(int(payload["rpos"].iloc[0]), {2, 3})

    def test_prepare_screen_chain_inputs_counts_all_ts_rpos_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = _screen_csv(Path(tmp) / "screen.csv")
            prepared = prepare_screen_chain_inputs(
                csv_path,
                ts_types=["TS1", "TS2", "TS3", "TS4"],
            )

        self.assertEqual(len(prepared["payloads"]), 8)
        self.assertEqual(len(set(prepared["tags"])), 8)


class ScreenChainSubmissionTests(unittest.TestCase):
    def test_submit_screen_chain_submits_one_chain_per_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = _screen_csv(Path(tmp) / "screen.csv")
            fake = FakeExecutor()
            cluster = ClusterConfig(
                backend="slurm",
                partition="kemi1",
                log_dir=Path(tmp) / "logs",
            )

            with patch("frust.cluster.chains.create_executor", return_value=fake):
                result = submit_screen_chain(
                    csv_path=csv_path,
                    ts_types=["TS1", "TS2", "TS3", "TS4"],
                    out_dir=Path(tmp) / "out",
                    cluster=cluster,
                    n_confs=None,
                    top_n=10,
                )

        self.assertEqual(result.mode, "screen_ts_per_rpos")
        self.assertEqual(len(result.tags), 8)
        self.assertEqual(len(result.job_ids), 8 * 6)

        init_submissions = [
            kwargs for fn, kwargs in fake.submissions if getattr(fn, "__name__", "") == "run_init"
        ]
        self.assertEqual(len(init_submissions), 8)
        for kwargs in init_submissions:
            self.assertIn("screen_target", kwargs)
            self.assertNotIn("ts_struct", kwargs)
            self.assertIsNone(kwargs["n_confs"])
            self.assertEqual(kwargs["top_n"], 10)

        dependent_updates = [
            params
            for params in fake.parameters
            if params.get("slurm_additional_parameters", {}).get("dependency")
        ]
        self.assertTrue(dependent_updates)

    def test_submit_screen_chain_forwards_composite_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = _screen_csv(Path(tmp) / "screen.csv")
            fake = FakeExecutor()
            cluster = ClusterConfig(
                backend="slurm",
                partition="kemi1",
                log_dir=Path(tmp) / "logs",
            )

            with patch("frust.cluster.chains.create_executor", return_value=fake):
                submit_screen_chain(
                    csv_path=csv_path,
                    ts_types=["TS1"],
                    out_dir=Path(tmp) / "out",
                    cluster=cluster,
                    composite_method="r2SCAN-3c",
                )

        for fn, kwargs in fake.submissions:
            if getattr(fn, "__name__", "") == "run_cleanup":
                self.assertNotIn("composite_method", kwargs)
            else:
                self.assertEqual(kwargs.get("composite_method"), "r2SCAN-3c")

    def test_submit_screen_chain_rejects_mixed_composite_and_basis(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = _screen_csv(Path(tmp) / "screen.csv")
            cluster = ClusterConfig(
                backend="slurm",
                partition="kemi1",
                log_dir=Path(tmp) / "logs",
            )

            with self.assertRaisesRegex(ValueError, "composite_method.*basisset"):
                submit_screen_chain(
                    csv_path=csv_path,
                    ts_types=["TS1"],
                    out_dir=Path(tmp) / "out",
                    cluster=cluster,
                    composite_method="r2SCAN-3c",
                    basisset="def2-SVP",
                )

    def test_legacy_submit_chain_still_requires_ts_xyz(self):
        signature = inspect.signature(submit_chain)
        self.assertIs(signature.parameters["ts_xyz"].default, inspect._empty)


class ScreenChainPipelineTests(unittest.TestCase):
    def test_run_init_generates_guesses_inside_cluster_stage(self):
        from frust.pipelines import run_screen_ts_per_rpos

        FakeStepper.instances = []
        seed_df = _seed_ts_guess_df()
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.object(
                    run_screen_ts_per_rpos,
                    "create_ts_guesses",
                    return_value={"TS1": seed_df},
                ) as create_ts_guesses,
                patch.object(run_screen_ts_per_rpos, "Stepper", FakeStepper),
            ):
                out = run_screen_ts_per_rpos.run_init(
                    _screen_target(),
                    n_confs=None,
                    n_cores=7,
                    mem_gb=11,
                    top_n=2,
                    save_dir=tmp,
                    save_output_dir=False,
                )

            save_path = Path(tmp)
            self.assertTrue((save_path / "ts_guess.parquet").exists())
            self.assertTrue((save_path / "init.parquet").exists())

        create_ts_guesses.assert_called_once()
        _, kwargs = create_ts_guesses.call_args
        self.assertEqual(kwargs["ts_types"], ["TS1"])
        self.assertIsNone(kwargs["n_confs"])
        self.assertEqual(kwargs["n_cores"], 7)

        self.assertEqual(len(FakeStepper.instances), 1)
        self.assertEqual(FakeStepper.instances[0].kwargs["step_type"], None)
        self.assertEqual(FakeStepper.instances[0].kwargs["n_cores"], 7)
        self.assertEqual(FakeStepper.instances[0].kwargs["memory_gb"], 11)
        self.assertEqual(len(out), 1)

    def test_run_init_composite_method_omits_basis_keywords(self):
        from frust.pipelines import run_screen_ts_per_rpos

        FakeStepper.instances = []
        seed_df = _seed_ts_guess_df()
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.object(
                    run_screen_ts_per_rpos,
                    "create_ts_guesses",
                    return_value={"TS1": seed_df},
                ),
                patch.object(run_screen_ts_per_rpos, "Stepper", FakeStepper),
            ):
                run_screen_ts_per_rpos.run_init(
                    _screen_target(),
                    save_dir=tmp,
                    composite_method="r2SCAN-3c",
                )

        orca_calls = [
            call for call in FakeStepper.instances[0].calls if call[0] == "orca"
        ]
        self.assertEqual([call[1] for call in orca_calls], ["DFT-pre-SP", "DFT-pre-Opt"])
        for _, _, options, *_ in orca_calls:
            self.assertIn("r2SCAN-3c", options)
            self.assertNotIn("6-31G**", options)
            self.assertNotIn("6-31+G**", options)

    def test_screen_chain_composite_method_reaches_all_post_init_stages(self):
        from frust.pipelines import run_screen_ts_per_rpos

        FakeStepper.instances = []
        with tempfile.TemporaryDirectory() as tmp:
            save_path = Path(tmp)
            _seed_ts_guess_df().to_parquet(save_path / "init.parquet")
            with patch.object(run_screen_ts_per_rpos, "Stepper", FakeStepper):
                run_screen_ts_per_rpos.run_hess(
                    "init.parquet",
                    save_dir=tmp,
                    composite_method="r2SCAN-3c",
                )
                run_screen_ts_per_rpos.run_OptTS(
                    "init.hess.parquet",
                    save_dir=tmp,
                    composite_method="r2SCAN-3c",
                )
                run_screen_ts_per_rpos.run_freq(
                    "init.hess.optts.parquet",
                    save_dir=tmp,
                    composite_method="r2SCAN-3c",
                )
                run_screen_ts_per_rpos.run_solv(
                    "init.hess.optts.freq.parquet",
                    save_dir=tmp,
                    composite_method="r2SCAN-3c",
                )

        expected = ["Hess", "OptTS", "Freq", "DFT-solv"]
        seen = []
        for instance in FakeStepper.instances:
            for call in instance.calls:
                if call[0] != "orca":
                    continue
                _, name, options, *_ = call
                seen.append(name)
                self.assertIn("r2SCAN-3c", options)
                self.assertNotIn("6-31G**", options)
                self.assertNotIn("6-31+G**", options)
        self.assertEqual(seen, expected)

    def test_screen_chain_conventional_theory_still_uses_basis_keywords(self):
        from frust.pipelines import run_screen_ts_per_rpos

        self.assertEqual(
            run_screen_ts_per_rpos._resolve_theory(
                functional="B3LYP",
                basisset="def2-SVP",
                basisset_solv="def2-SVPD",
            ),
            ("B3LYP", "def2-SVP", "def2-SVPD"),
        )


if __name__ == "__main__":
    unittest.main()
