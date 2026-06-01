import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


class PublicApiTests(unittest.TestCase):
    def test_top_level_public_names_resolve(self):
        import frust as ft
        from frust import (
            Stepper,
            show_steps,
            lowest_energy_rows,
            map_substrate_names,
            pipelines,
            pipes,
            screen,
            plot_vibs,
            summarize_ts_vibrations,
            utils,
            vis,
            write_xyz,
        )
        from frust import pipelines as pipelines_direct
        from frust import screen as screen_direct
        from frust import utils as utils_direct
        from frust import vis as vis_direct
        from frust.cluster import ClusterConfig, Resources, submit_chain, submit_jobs, submit_screen_chain
        from frust.pipes import run_mols, run_screen_ts_per_rpos
        from frust.pipelines import run_ts_per_rpos
        from frust.stepper import Stepper as StepperDirect
        from frust.utils.analytics import summarize_ts_vibrations as summarize_direct
        from frust.utils.dataframes import show_steps as show_steps_direct
        from frust.utils.dataframes import lowest_energy_rows as lowest_direct
        from frust.utils.dataframes import map_substrate_names as map_names_direct
        from frust.utils.io import write_xyz as write_xyz_direct
        from frust.utils.mols import get_molecule_name
        from frust.vis import plot_mols as plot_mols_direct
        from frust.vis import plot_vibs as plot_vibs_direct
        from frust.vis import ArrowOverlay, ScreenLabelOverlay
        from frust.vis import reaction_scene_cells

        expected = {
            "Stepper",
            "cluster",
            "pipelines",
            "pipes",
            "screen",
            "utils",
            "vis",
            "show_steps",
            "lowest_energy_rows",
            "map_substrate_names",
            "summarize_ts_vibrations",
            "plot_vibs",
            "write_xyz",
            "write_xyz_structures",
            "read_ts_type_from_xyz",
            "create_mol_per_rpos",
            "create_ts_per_rpos",
            "embed_mols",
            "embed_ts",
            "normalize_dataframe",
            "energy_columns",
            "normal_termination_columns",
            "ClusterConfig",
            "Resources",
            "submit_jobs",
            "submit_chain",
            "submit_screen_chain",
        }

        self.assertTrue(expected.issubset(set(ft.__all__)))
        self.assertIs(ft.Stepper, StepperDirect)
        self.assertIs(Stepper, StepperDirect)
        self.assertIs(ft.show_steps, show_steps_direct)
        self.assertIs(show_steps, show_steps_direct)
        self.assertIs(ft.lowest_energy_rows, lowest_direct)
        self.assertIs(lowest_energy_rows, lowest_direct)
        self.assertIs(ft.map_substrate_names, map_names_direct)
        self.assertIs(map_substrate_names, map_names_direct)
        self.assertIs(ft.summarize_ts_vibrations, summarize_direct)
        self.assertIs(summarize_ts_vibrations, summarize_direct)
        self.assertIs(ft.plot_vibs, plot_vibs_direct)
        self.assertIs(plot_vibs, plot_vibs_direct)
        self.assertIs(ft.write_xyz, write_xyz_direct)
        self.assertIs(write_xyz, write_xyz_direct)
        self.assertIs(ft.ClusterConfig, ClusterConfig)
        self.assertIs(ft.Resources, Resources)
        self.assertIs(ft.submit_jobs, submit_jobs)
        self.assertIs(ft.submit_chain, submit_chain)
        self.assertIs(ft.submit_screen_chain, submit_screen_chain)
        self.assertIs(ft.cluster.submit_jobs, submit_jobs)
        self.assertIs(ft.utils, utils_direct)
        self.assertIs(utils, utils_direct)
        self.assertIs(ft.utils.write_xyz, write_xyz_direct)
        self.assertIs(ft.utils.summarize_ts_vibrations, summarize_direct)
        self.assertIs(ft.utils.map_substrate_names, map_names_direct)
        self.assertTrue(callable(get_molecule_name))
        self.assertIs(ft.vis, vis_direct)
        self.assertIs(vis, vis_direct)
        self.assertIs(ft.vis.plot_mols, plot_mols_direct)
        self.assertIs(ft.vis.plot_vibs, plot_vibs_direct)
        self.assertIs(ft.vis.ArrowOverlay, ArrowOverlay)
        self.assertIs(ft.vis.ScreenLabelOverlay, ScreenLabelOverlay)
        self.assertIs(ft.vis.reaction_scene_cells, reaction_scene_cells)
        self.assertIs(ft.pipelines, pipelines_direct)
        self.assertIs(ft.screen, screen_direct)
        self.assertIs(screen, screen_direct)
        self.assertIs(pipelines, pipelines_direct)
        self.assertIs(ft.pipelines.run_ts_per_rpos, run_ts_per_rpos)
        self.assertIs(ft.pipes.run_mols, run_mols)
        self.assertIs(pipes.run_mols, run_mols)
        self.assertIs(ft.pipes.run_screen_ts_per_rpos, run_screen_ts_per_rpos)

    def test_import_frust_is_lazy_in_fresh_process(self):
        code = textwrap.dedent(
            """
            import json
            import sys

            import frust

            after_import = {
                "frust.cluster": "frust.cluster" in sys.modules,
                "frust.pipelines": "frust.pipelines" in sys.modules,
                "frust.pipelines.run_ts_per_rpos": "frust.pipelines.run_ts_per_rpos" in sys.modules,
                "frust.pipes": "frust.pipes" in sys.modules,
                "frust.stepper": "frust.stepper" in sys.modules,
                "frust.utils": "frust.utils" in sys.modules,
                "frust.vis": "frust.vis" in sys.modules,
                "frust.vis.molecules": "frust.vis.molecules" in sys.modules,
                "frust.utils.analytics": "frust.utils.analytics" in sys.modules,
                "frust.utils.dataframes": "frust.utils.dataframes" in sys.modules,
                "frust.utils.io": "frust.utils.io" in sys.modules,
            }

            _ = frust.show_steps
            after_show_steps = {
                "frust.utils.dataframes": "frust.utils.dataframes" in sys.modules,
                "frust.utils.io": "frust.utils.io" in sys.modules,
                "frust.cluster": "frust.cluster" in sys.modules,
                "frust.pipelines": "frust.pipelines" in sys.modules,
                "frust.pipelines.run_ts_per_rpos": "frust.pipelines.run_ts_per_rpos" in sys.modules,
                "frust.pipes": "frust.pipes" in sys.modules,
                "frust.stepper": "frust.stepper" in sys.modules,
                "frust.utils": "frust.utils" in sys.modules,
                "frust.vis": "frust.vis" in sys.modules,
                "frust.vis.molecules": "frust.vis.molecules" in sys.modules,
                "frust.utils.analytics": "frust.utils.analytics" in sys.modules,
            }

            _ = frust.Stepper
            after_stepper = {
                "frust.stepper": "frust.stepper" in sys.modules,
                "frust.vis": "frust.vis" in sys.modules,
            }

            _ = frust.write_xyz
            after_write_xyz = {
                "frust.utils.io": "frust.utils.io" in sys.modules,
                "frust.vis": "frust.vis" in sys.modules,
                "frust.stepper": "frust.stepper" in sys.modules,
            }

            _ = frust.utils
            after_utils_namespace = {
                "frust.utils": "frust.utils" in sys.modules,
                "frust.utils.analytics": "frust.utils.analytics" in sys.modules,
                "frust.utils.dataframes": "frust.utils.dataframes" in sys.modules,
                "frust.utils.io": "frust.utils.io" in sys.modules,
                "frust.utils.mols": "frust.utils.mols" in sys.modules,
            }

            _ = frust.utils.summarize_ts_vibrations
            after_utils_analytics = {
                "frust.utils.analytics": "frust.utils.analytics" in sys.modules,
            }

            _ = frust.pipes
            after_pipes = {
                "frust.pipes": "frust.pipes" in sys.modules,
                "frust.cluster": "frust.cluster" in sys.modules,
            }

            _ = frust.pipelines
            after_pipelines_namespace = {
                "frust.pipelines": "frust.pipelines" in sys.modules,
                "frust.pipelines.run_ts_per_rpos": "frust.pipelines.run_ts_per_rpos" in sys.modules,
            }

            _ = frust.pipelines.run_ts_per_rpos
            after_pipeline_module = {
                "frust.pipelines.run_ts_per_rpos": "frust.pipelines.run_ts_per_rpos" in sys.modules,
            }

            _ = frust.cluster
            after_cluster = {
                "frust.cluster": "frust.cluster" in sys.modules,
            }

            _ = frust.vis
            after_vis_namespace = {
                "frust.vis": "frust.vis" in sys.modules,
                "frust.vis.molecules": "frust.vis.molecules" in sys.modules,
                "frust.vis.vibrations": "frust.vis.vibrations" in sys.modules,
            }

            _ = frust.vis.plot_mols
            after_vis_plot_mols = {
                "frust.vis.molecules": "frust.vis.molecules" in sys.modules,
                "frust.vis.vibrations": "frust.vis.vibrations" in sys.modules,
            }

            _ = frust.plot_vibs
            after_plot_vibs = {
                "frust.vis": "frust.vis" in sys.modules,
            }

            print(
                json.dumps(
                    {
                        "after_import": after_import,
                        "after_show_steps": after_show_steps,
                        "after_stepper": after_stepper,
                        "after_write_xyz": after_write_xyz,
                        "after_utils_namespace": after_utils_namespace,
                        "after_utils_analytics": after_utils_analytics,
                        "after_pipes": after_pipes,
                        "after_pipelines_namespace": after_pipelines_namespace,
                        "after_pipeline_module": after_pipeline_module,
                        "after_cluster": after_cluster,
                        "after_vis_namespace": after_vis_namespace,
                        "after_vis_plot_mols": after_vis_plot_mols,
                        "after_plot_vibs": after_plot_vibs,
                    }
                )
            )
            """
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(proc.stdout)

        self.assertEqual(
            result["after_import"],
            {
                "frust.cluster": False,
                "frust.pipelines": False,
                "frust.pipelines.run_ts_per_rpos": False,
                "frust.pipes": False,
                "frust.stepper": False,
                "frust.utils": False,
                "frust.vis": False,
                "frust.vis.molecules": False,
                "frust.utils.analytics": False,
                "frust.utils.dataframes": False,
                "frust.utils.io": False,
            },
        )
        self.assertEqual(
            result["after_show_steps"],
            {
                "frust.utils.dataframes": True,
                "frust.utils.io": False,
                "frust.cluster": False,
                "frust.pipelines": False,
                "frust.pipelines.run_ts_per_rpos": False,
                "frust.pipes": False,
                "frust.stepper": False,
                "frust.utils": True,
                "frust.vis": False,
                "frust.vis.molecules": False,
                "frust.utils.analytics": False,
            },
        )
        self.assertEqual(
            result["after_write_xyz"],
            {
                "frust.utils.io": True,
                "frust.vis": True,
                "frust.stepper": True,
            },
        )
        self.assertEqual(
            result["after_utils_namespace"],
            {
                "frust.utils": True,
                "frust.utils.analytics": False,
                "frust.utils.dataframes": True,
                "frust.utils.io": True,
                "frust.utils.mols": False,
            },
        )
        self.assertEqual(result["after_utils_analytics"], {"frust.utils.analytics": True})
        self.assertEqual(
            result["after_stepper"],
            {
                "frust.stepper": True,
                "frust.vis": False,
            },
        )
        self.assertEqual(
            result["after_pipes"],
            {
                "frust.pipes": True,
                "frust.cluster": False,
            },
        )
        self.assertEqual(
            result["after_pipelines_namespace"],
            {
                "frust.pipelines": True,
                "frust.pipelines.run_ts_per_rpos": False,
            },
        )
        self.assertEqual(result["after_pipeline_module"], {"frust.pipelines.run_ts_per_rpos": True})
        self.assertEqual(result["after_cluster"], {"frust.cluster": True})
        self.assertEqual(
            result["after_vis_namespace"],
            {
                "frust.vis": True,
                "frust.vis.molecules": False,
                "frust.vis.vibrations": False,
            },
        )
        self.assertEqual(
            result["after_vis_plot_mols"],
            {
                "frust.vis.molecules": True,
                "frust.vis.vibrations": False,
            },
        )
        self.assertEqual(result["after_plot_vibs"], {"frust.vis": True})

    def test_screen_import_works_without_orca_configuration(self):
        repo_root = Path(__file__).resolve().parents[1]
        tooltoad_root = repo_root.parent / "tool-toad"
        code = textwrap.dedent(
            """
            import pandas as pd
            import frust as ft

            components = ft.screen.read(
                pd.DataFrame(
                    {
                        "role": ["substrate", "catalyst"],
                        "smiles": [
                            "CN1C=CC=C1",
                            "CC1(C)CCCC(C)(C)N1C2=CC=CC=C2B",
                        ],
                    }
                )
            )
            print(",".join(components["role"].tolist()))
            """
        )

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            home = tmp / "home"
            home.mkdir()
            env = dict(os.environ)
            for key in ("ORCA_EXE", "OPEN_MPI_DIR", "XTB_EXE", "GXTB_EXE", "XTBPATH"):
                env.pop(key, None)
            env["HOME"] = str(home)
            env["TOOLTOAD_DOTENV_PATH"] = str(tmp / "missing.env")
            env["PYTHONPATH"] = os.pathsep.join(
                part
                for part in [
                    str(repo_root),
                    str(tooltoad_root),
                    env.get("PYTHONPATH", ""),
                ]
                if part
            )
            proc = subprocess.run(
                [sys.executable, "-c", code],
                cwd=tmp,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertEqual(proc.stdout.strip(), "substrate,catalyst")


if __name__ == "__main__":
    unittest.main()
