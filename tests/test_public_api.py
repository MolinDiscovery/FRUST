import json
import subprocess
import sys
import textwrap
import unittest


class PublicApiTests(unittest.TestCase):
    def test_top_level_public_names_resolve(self):
        import frust as ft
        from frust import (
            Stepper,
            show_steps,
            lowest_energy_rows,
            pipes,
            plot_vibs,
            summarize_ts_vibrations,
            write_xyz,
        )
        from frust.cluster import ClusterConfig, Resources, submit_chain, submit_jobs
        from frust.pipes import run_mols
        from frust.stepper import Stepper as StepperDirect
        from frust.utils.analytics import summarize_ts_vibrations as summarize_direct
        from frust.utils.dataframes import show_steps as show_steps_direct
        from frust.utils.dataframes import lowest_energy_rows as lowest_direct
        from frust.utils.io import write_xyz as write_xyz_direct
        from frust.vis import plot_vibs as plot_vibs_direct

        expected = {
            "Stepper",
            "cluster",
            "pipes",
            "show_steps",
            "lowest_energy_rows",
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
        }

        self.assertTrue(expected.issubset(set(ft.__all__)))
        self.assertIs(ft.Stepper, StepperDirect)
        self.assertIs(Stepper, StepperDirect)
        self.assertIs(ft.show_steps, show_steps_direct)
        self.assertIs(show_steps, show_steps_direct)
        self.assertIs(ft.lowest_energy_rows, lowest_direct)
        self.assertIs(lowest_energy_rows, lowest_direct)
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
        self.assertIs(ft.cluster.submit_jobs, submit_jobs)
        self.assertIs(ft.pipes.run_mols, run_mols)
        self.assertIs(pipes.run_mols, run_mols)

    def test_import_frust_is_lazy_in_fresh_process(self):
        code = textwrap.dedent(
            """
            import json
            import sys

            import frust

            after_import = {
                "frust.cluster": "frust.cluster" in sys.modules,
                "frust.pipes": "frust.pipes" in sys.modules,
                "frust.stepper": "frust.stepper" in sys.modules,
                "frust.vis": "frust.vis" in sys.modules,
                "frust.utils.analytics": "frust.utils.analytics" in sys.modules,
                "frust.utils.dataframes": "frust.utils.dataframes" in sys.modules,
                "frust.utils.io": "frust.utils.io" in sys.modules,
            }

            _ = frust.show_steps
            after_show_steps = {
                "frust.utils.dataframes": "frust.utils.dataframes" in sys.modules,
                "frust.utils.io": "frust.utils.io" in sys.modules,
                "frust.cluster": "frust.cluster" in sys.modules,
                "frust.pipes": "frust.pipes" in sys.modules,
                "frust.stepper": "frust.stepper" in sys.modules,
                "frust.vis": "frust.vis" in sys.modules,
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

            _ = frust.pipes
            after_pipes = {
                "frust.pipes": "frust.pipes" in sys.modules,
                "frust.cluster": "frust.cluster" in sys.modules,
            }

            _ = frust.cluster
            after_cluster = {
                "frust.cluster": "frust.cluster" in sys.modules,
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
                        "after_pipes": after_pipes,
                        "after_cluster": after_cluster,
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
                "frust.pipes": False,
                "frust.stepper": False,
                "frust.vis": False,
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
                "frust.pipes": False,
                "frust.stepper": False,
                "frust.vis": False,
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
        self.assertEqual(result["after_cluster"], {"frust.cluster": True})
        self.assertEqual(result["after_plot_vibs"], {"frust.vis": True})


if __name__ == "__main__":
    unittest.main()
