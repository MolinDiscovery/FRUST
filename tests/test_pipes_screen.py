from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import frust.pipes as pipes
from frust import screen


CATALYST = "CC1(C)CCCC(C)(C)N1C2=CC=CC=C2B"


def _component_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "role": ["substrate", "catalyst"],
            "smiles": ["CN1C=CC=C1", CATALYST],
            "compound_name": ["pyrrole", "cat_a"],
            "rpos": ["2,3", None],
        }
    )


def _guess_df(ts_type: str, *, rpos: int = 2) -> pd.DataFrame:
    rows = []
    for cid, energy in enumerate([2.0, 0.5]):
        rows.append(
            {
                "custom_name": f"{ts_type}(pyrrole__cat_a_rpos({rpos}))",
                "structure_id": f"{ts_type}:pyrrole__cat_a:r{rpos}",
                "system_name": "pyrrole__cat_a",
                "substrate_name": "pyrrole",
                "catalyst_name": "cat_a",
                "structure_type": ts_type,
                "molecule_role": "ts",
                "rpos": rpos,
                "cid": cid,
                "atoms": ["H", "H"],
                "coords_embedded": [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]],
                "constraint_roles": [{"a": 0, "b": 1}],
                "constraint_spec": [[{"kind": "distance", "roles": ["a", "b"], "value": 1.0}]],
                "constraint_atoms": [[0, 1, 0, 1, 0, 1]],
                "mock_energy": energy,
            }
        )
    return pd.DataFrame(rows)


def _fake_create_ts_guesses(systems, *, ts_types, n_confs, n_cores, **kwargs):
    del systems, n_confs, n_cores, kwargs
    return {str(ts_type).upper(): _guess_df(str(ts_type).upper()) for ts_type in ts_types}


class FakeStepper:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        FakeStepper.instances.append(self)

    def xtb(self, df, name, options, constraint=False, lowest=None, **kwargs):
        self.calls.append(("xtb", name, options, constraint, lowest, kwargs))
        out = df.copy()
        out[f"{name}-NT"] = True
        out[f"{name}-EE"] = out["mock_energy"]
        if "opt" in options:
            out[f"{name}-oc"] = out["coords_embedded"]
        return out

    def orca(
        self,
        df,
        name,
        options,
        constraint=False,
        lowest=None,
        xtra_inp_str="",
        **kwargs,
    ):
        self.calls.append(
            ("orca", name, options, constraint, lowest, xtra_inp_str, kwargs)
        )
        out = df.copy()
        out[f"{name}-NT"] = True
        out[f"{name}-EE"] = out["mock_energy"]
        if "Opt" in options or "OptTS" in options:
            out[f"{name}-oc"] = out["coords_embedded"]
        return out


class ScreenPipesTests(unittest.TestCase):
    def setUp(self):
        FakeStepper.instances = []

    def test_run_screen_ts_per_rpos_accepts_component_dataframe(self):
        with (
            patch.object(pipes, "Stepper", FakeStepper),
            patch.object(
                pipes.screen,
                "create_ts_guesses",
                side_effect=_fake_create_ts_guesses,
            ) as create_ts_guesses,
            patch.object(pipes, "embed_ts", side_effect=AssertionError("legacy embedder used")),
        ):
            out = pipes.run_screen_ts_per_rpos(
                _component_df(),
                ts_types=["TS1", "TS2"],
                n_confs=3,
                n_cores=5,
                mem_gb=11,
                DFT=False,
            )

        create_ts_guesses.assert_called_once()
        systems_arg = create_ts_guesses.call_args.args[0]
        self.assertEqual(len(systems_arg), 1)
        self.assertEqual(create_ts_guesses.call_args.kwargs["ts_types"], ["TS1", "TS2"])
        self.assertEqual(create_ts_guesses.call_args.kwargs["n_confs"], 3)
        self.assertEqual(create_ts_guesses.call_args.kwargs["n_cores"], 5)

        self.assertEqual(len(out), 2)
        self.assertEqual(set(out["structure_type"]), {"TS1", "TS2"})
        self.assertTrue((out["cid"] == 1).all())
        self.assertEqual(FakeStepper.instances[0].kwargs["step_type"], None)
        self.assertEqual(FakeStepper.instances[0].kwargs["memory_gb"], 11)

    def test_run_screen_ts_per_rpos_accepts_expanded_systems_dataframe(self):
        expanded = screen.expand(screen.read(_component_df()))

        with (
            patch.object(pipes, "Stepper", FakeStepper),
            patch.object(
                pipes.screen,
                "read",
                side_effect=AssertionError("expanded systems should not be reread"),
            ),
            patch.object(
                pipes.screen,
                "expand",
                side_effect=AssertionError("expanded systems should not be reexpanded"),
            ),
            patch.object(
                pipes.screen,
                "create_ts_guesses",
                side_effect=_fake_create_ts_guesses,
            ) as create_ts_guesses,
        ):
            out = pipes.run_screen_ts_per_rpos(expanded, ts_types=["TS4"], DFT=False)

        create_ts_guesses.assert_called_once()
        pd.testing.assert_frame_equal(create_ts_guesses.call_args.args[0], expanded)
        self.assertEqual(len(out), 1)
        self.assertEqual(out["structure_type"].iloc[0], "TS4")

    def test_run_screen_ts_per_rpos_writes_output_parquet(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "screen.parquet"
            with (
                patch.object(pipes, "Stepper", FakeStepper),
                patch.object(
                    pipes.screen,
                    "create_ts_guesses",
                    side_effect=_fake_create_ts_guesses,
                ),
            ):
                out = pipes.run_screen_ts_per_rpos(
                    _component_df(),
                    ts_types=["TS1"],
                    output_parquet=str(output_path),
                    DFT=False,
                )

            self.assertTrue(output_path.exists())
            written = pd.read_parquet(output_path)

        self.assertEqual(len(out), 1)
        self.assertEqual(len(written), 1)
        self.assertEqual(written["structure_type"].iloc[0], "TS1")

    def test_pipes_namespace_exposes_screen_workflow(self):
        self.assertIs(pipes.run_screen_ts_per_rpos, pipes.run_screen_ts_per_rpos)


if __name__ == "__main__":
    unittest.main()
