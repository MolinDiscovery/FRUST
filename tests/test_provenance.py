import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from frust.utils.provenance import env_executable


def _executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return path


class ProvenanceResolverTests(unittest.TestCase):
    def test_env_executable_resolves_path_to_absolute_path(self):
        with tempfile.TemporaryDirectory() as td:
            exe = _executable(Path(td) / "xtb")
            with patch.dict(os.environ, {"FRUST_TEST_EXE": str(exe)}):
                meta = env_executable("FRUST_TEST_EXE")

        self.assertEqual(meta["path"], str(exe.resolve()))
        self.assertEqual(meta["configured"], str(exe))
        self.assertEqual(meta["source"], "FRUST_TEST_EXE")
        self.assertTrue(meta["resolved"])

    def test_env_executable_falls_back_to_path_command(self):
        with tempfile.TemporaryDirectory() as td:
            bin_dir = Path(td)
            exe = _executable(bin_dir / "fake-xtb")
            env = {"PATH": str(bin_dir)}
            with patch.dict(os.environ, env, clear=True):
                meta = env_executable("FRUST_MISSING_EXE", fallback_command="fake-xtb")

        self.assertEqual(meta["path"], str(exe.resolve()))
        self.assertEqual(meta["configured"], "fake-xtb")
        self.assertEqual(meta["source"], "PATH")
        self.assertTrue(meta["resolved"])

    def test_env_executable_records_missing_command_without_raising(self):
        with patch.dict(os.environ, {"PATH": ""}, clear=True):
            meta = env_executable("FRUST_MISSING_EXE", fallback_command="fake-xtb")

        self.assertIsNone(meta["path"])
        self.assertEqual(meta["configured"], "fake-xtb")
        self.assertEqual(meta["source"], "PATH")
        self.assertFalse(meta["resolved"])


if __name__ == "__main__":
    unittest.main()
