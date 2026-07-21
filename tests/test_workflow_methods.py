from __future__ import annotations

import unittest
import importlib

import frust as ft
from frust.workflows import methods


class WorkflowMethodTests(unittest.TestCase):
    def test_r2scan_3c_preset_uses_composite_without_basis(self):
        method = ft.workflows.methods.preset("r2SCAN-3c")
        spec = method.for_stage("dft_pre_sp")

        self.assertEqual(spec.engine, "orca")
        self.assertIn("r2SCAN-3c", spec.options)
        self.assertIn("SP", spec.options)
        self.assertNotIn("6-31G**", spec.options)
        self.assertNotIn("def2-SVP", spec.options)

    def test_gxtb_spec_does_not_add_xtb_gfn_option(self):
        spec = methods.gxtb(job="sp")

        self.assertEqual(spec.engine, "gxtb")
        self.assertEqual(spec.options, {})

    def test_builtin_presets_use_gxtb_for_low_cost_init_stages(self):
        for name in (
            "r2scan-3c",
            "wb97xd3-631g",
            "r2scan-3c-solv",
            "wb97xd3-631g-solv",
            "r2scan-def2svp",
        ):
            with self.subTest(name=name):
                method = methods.preset(name)

                self.assertEqual(method.for_stage("xtb_sp").engine, "gxtb")
                self.assertEqual(method.for_stage("xtb_sp").options, {})
                self.assertEqual(method.for_stage("xtb_opt").engine, "gxtb")
                self.assertEqual(method.for_stage("xtb_opt").options, {"opt": None})

    def test_solvent_inclusive_presets_solvate_every_dft_stage(self):
        dft_stages = (
            "dft_rank_sp",
            "dft_preopt",
            "dft_opt",
            "dft_hessian",
            "dft_ts_opt",
            "dft_freq",
        )
        for name in ("r2scan-3c-solv", "wb97xd3-631g-solv"):
            with self.subTest(name=name):
                method = methods.preset(name)

                self.assertFalse(method.include_terminal_solv_sp)
                self.assertNotIn("dft_solv_sp", method.stages)
                for stage in dft_stages:
                    self.assertIn(
                        'SMDSOLVENT "chloroform"',
                        method.for_stage(stage).xtra_inp_str,
                    )
                    self.assertEqual(method.for_stage(stage).solvent, "chloroform")

    def test_wb97xd3_solvent_preset_keeps_the_regular_dft_basis(self):
        method = methods.preset("wb97xd3-631g-solv")

        for stage in (
            "dft_rank_sp",
            "dft_preopt",
            "dft_opt",
            "dft_hessian",
            "dft_ts_opt",
            "dft_freq",
        ):
            with self.subTest(stage=stage):
                options = method.for_stage(stage).options
                self.assertIn("6-31G**", options)
                self.assertNotIn("6-31+G**", options)

    def test_method_replace_updates_only_named_stage(self):
        base = methods.preset("r2scan-3c")
        updated = base.replace(xtb_sp=methods.xtb(gfn=2))

        self.assertEqual(updated.for_stage("xtb_sp").engine, "xtb")
        self.assertEqual(updated.for_stage("xtb_opt").engine, "gxtb")
        self.assertEqual(base.for_stage("xtb_sp").engine, "gxtb")
        self.assertEqual(base.for_stage("xtb_opt").engine, "gxtb")

    def test_with_ts_mode_following_preserves_method_and_solvent(self):
        base = methods.preset("r2scan-3c-solv")

        updated = methods.with_ts_mode_following(
            base,
            mode_roles=("pin_B", "substrate_C"),
            active_roles=("cat_B", "transfer_H", "pin_B", "substrate_C"),
            active_atoms_factor=1.5,
            recalc_hess=3,
            trust_radius=0.15,
        )

        spec = updated.for_stage("dft_ts_opt")
        self.assertEqual(spec.method, "r2SCAN-3c")
        self.assertEqual(spec.solvent, "chloroform")
        self.assertIn("TightOpt", spec.options)
        self.assertEqual(spec.kwargs["ts_mode"], ("pin_B", "substrate_C"))
        self.assertEqual(spec.kwargs["recalc_hess"], 3)
        self.assertNotIn("TightOpt", base.for_stage("dft_ts_opt").options)

    def test_legacy_stage_alias_updates_canonical_stage(self):
        base = methods.preset("r2scan-3c")
        replacement = methods.orca(method="PBE0", basis="def2-SVP")

        updated = base.replace(dft_pre_sp=replacement)

        self.assertIs(updated.for_stage("dft_rank_sp"), replacement)
        self.assertIs(updated.for_stage("dft_pre_sp"), replacement)

    def test_register_user_preset(self):
        method = methods.preset("r2scan-3c").replace(
            xtb_opt=methods.xtb(gfn=2, opt=True),
        )
        methods.register_preset("unit-test-xtb-opt", method)

        self.assertIs(methods.preset("unit-test-xtb-opt"), method)

    def test_register_user_preset_does_not_hide_builtins(self):
        fresh_methods = importlib.reload(methods)
        method = fresh_methods.MethodPlan(
            name="minimal",
            stages={"xtb_sp": fresh_methods.gxtb(job="sp")},
        )
        fresh_methods.register_preset("minimal-custom", method)

        self.assertEqual(fresh_methods.preset("r2scan-3c").name, "r2scan-3c")


if __name__ == "__main__":
    unittest.main()
