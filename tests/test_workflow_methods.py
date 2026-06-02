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

    def test_method_replace_updates_only_named_stage(self):
        base = methods.preset("r2scan-3c")
        updated = base.replace(xtb_sp=methods.gxtb(job="sp"))

        self.assertEqual(updated.for_stage("xtb_sp").engine, "gxtb")
        self.assertEqual(updated.for_stage("xtb_opt").engine, "xtb")
        self.assertEqual(base.for_stage("xtb_sp").engine, "xtb")

    def test_register_user_preset(self):
        method = methods.preset("r2scan-3c").replace(
            xtb_sp=methods.gxtb(job="sp"),
            xtb_opt=methods.gxtb(job="opt"),
        )
        methods.register_preset("unit-test-gxtb", method)

        self.assertIs(methods.preset("unit-test-gxtb"), method)

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
