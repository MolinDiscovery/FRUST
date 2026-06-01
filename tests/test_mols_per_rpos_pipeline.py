from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import frust.pipes as pipes
from frust.cluster import ClusterConfig, Resources, submit_jobs
from frust.cluster.inputs import prepare_pipeline_inputs


def _input_csv(path: Path) -> Path:
    df = pd.DataFrame({"smiles": ["CN1C=CC=C1"], "rpos": ["2,3"]})
    df.to_csv(path, index=False)
    return path


def _mol_jobs():
    return [
        {"int2_rpos(2)": ("mol-r2", {"structure_type": "MOL", "rpos": 2})},
        {"int2_rpos(3)": ("mol-r3", {"structure_type": "MOL", "rpos": 3})},
    ]


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

    def build_initial_df(self, embedded):
        self.calls.append(("build_initial_df", embedded))
        return pd.DataFrame(
            {
                "structure_id": ["MOL:pyrrole:int2:r2", "MOL:pyrrole:int2:r2"],
                "custom_name": ["int2_rpos(2)", "int2_rpos(2)"],
                "substrate_name": ["pyrrole", "pyrrole"],
                "structure_type": ["MOL", "MOL"],
                "molecule_role": ["int2", "int2"],
                "rpos": [2, 2],
                "cid": [0, 1],
                "atoms": [["H"], ["H"]],
                "coords_embedded": [[(0.0, 0.0, 0.0)], [(1.0, 0.0, 0.0)]],
            }
        )

    def xtb(self, df, name, options, lowest=None, **kwargs):
        self.calls.append(("xtb", name, options, lowest, kwargs))
        out = df.copy()
        if lowest:
            out = out.sort_values("cid").head(lowest).copy()
        out[f"{name}-NT"] = True
        out[f"{name}-EE"] = [float(value) for value in range(len(out), 0, -1)]
        if "opt" in options:
            out[f"{name}-oc"] = out["coords_embedded"]
        return out

    def orca(self, df, name, options, lowest=None, xtra_inp_str="", **kwargs):
        self.calls.append(("orca", name, options, lowest, xtra_inp_str, kwargs))
        out = df.copy()
        if lowest:
            out = out.head(lowest).copy()
        out[f"{name}-NT"] = True
        out[f"{name}-EE"] = [float(value) for value in range(len(out), 0, -1)]
        return out


class MolsPerRposInputTests(unittest.TestCase):
    def test_prepare_pipeline_inputs_fans_out_molecule_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = _input_csv(Path(tmp) / "mols.csv")
            with patch(
                "frust.cluster.inputs.create_mol_per_rpos",
                return_value=_mol_jobs(),
            ) as create_mol_per_rpos:
                prepared = prepare_pipeline_inputs(
                    csv_path,
                    pipeline="run_mols_per_rpos",
                    select_mols="int2",
                )

        create_mol_per_rpos.assert_called_once()
        self.assertEqual(create_mol_per_rpos.call_args.kwargs["return_format"], "list")
        self.assertEqual(create_mol_per_rpos.call_args.kwargs["select_mols"], "int2")
        self.assertEqual(prepared["mode"], "mols_per_rpos")
        self.assertEqual(len(prepared["payloads"]), 2)
        self.assertEqual(prepared["tags"], ["int2_rpos_2", "int2_rpos_3"])


class MolsPerRposPipelineTests(unittest.TestCase):
    def setUp(self):
        FakeStepper.instances = []

    def test_run_mols_per_rpos_runs_one_prepared_payload(self):
        with (
            patch.object(pipes, "Stepper", FakeStepper),
            patch.object(pipes, "embed_mols", return_value={"embedded": "payload"}) as embed_mols,
            patch.object(
                pipes,
                "create_mol_per_rpos",
                side_effect=AssertionError("run_mols_per_rpos should not expand CSV input"),
            ),
        ):
            out = pipes.run_mols_per_rpos(
                _mol_jobs()[0],
                n_confs=4,
                n_cores=6,
                mem_gb=12,
                work_dir="/scratch/frust",
                DFT=False,
            )

        embed_mols.assert_called_once_with(_mol_jobs()[0], n_confs=4, n_cores=6)
        self.assertEqual(FakeStepper.instances[0].kwargs["step_type"], "MOLS")
        self.assertEqual(FakeStepper.instances[0].kwargs["work_dir"], "/scratch/frust")
        self.assertEqual(len(out), 1)
        self.assertEqual(out["cid"].iloc[0], 1)

    def test_run_mols_per_rpos_writes_output_parquet(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "one.parquet"
            with (
                patch.object(pipes, "Stepper", FakeStepper),
                patch.object(pipes, "embed_mols", return_value={"embedded": "payload"}),
            ):
                out = pipes.run_mols_per_rpos(
                    _mol_jobs()[0],
                    output_parquet=str(output_path),
                    DFT=False,
                )

            self.assertTrue(output_path.exists())
            written = pd.read_parquet(output_path)

        self.assertEqual(len(out), 1)
        self.assertEqual(len(written), 1)

    def test_run_mols_per_rpos_rejects_multi_payload_dict(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            pipes.run_mols_per_rpos({"a": "mol-a", "b": "mol-b"})


class MolsPerRposSubmissionTests(unittest.TestCase):
    def test_submit_jobs_submits_one_job_per_prepared_molecule_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = _input_csv(Path(tmp) / "mols.csv")
            fake = FakeExecutor()
            cluster = ClusterConfig(
                backend="slurm",
                partition="kemi1",
                log_dir=Path(tmp) / "logs",
            )

            with (
                patch("frust.cluster.facade.create_executor", return_value=fake),
                patch("frust.cluster.inputs.create_mol_per_rpos", return_value=_mol_jobs()),
            ):
                result = submit_jobs(
                    csv_path=csv_path,
                    pipeline="run_mols_per_rpos",
                    out_dir=Path(tmp) / "out",
                    cluster=cluster,
                    resources=Resources(cpus=4, mem_gb=10, timeout_min=60),
                    n_confs=3,
                    select_mols="int2",
                )

        self.assertEqual(result.mode, "run_mols_per_rpos")
        self.assertEqual(result.tags, ["int2_rpos_2", "int2_rpos_3"])
        self.assertEqual(len(fake.submissions), 2)
        for _, kwargs in fake.submissions:
            self.assertIn("mol_struct", kwargs)
            self.assertNotIn("ligand_smiles_df", kwargs)
            self.assertEqual(kwargs["n_confs"], 3)
            self.assertEqual(kwargs["n_cores"], 4)
            self.assertEqual(kwargs["mem_gb"], 10)


if __name__ == "__main__":
    unittest.main()
