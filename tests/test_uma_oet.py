import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from frust.config import get_oet_tools
from frust.utils.uma import (
    DEFAULT_UMA_MODEL,
    parse_uma_spec,
    uma_orca_block,
    uma_server,
)


def _fake_oet_root(tmp_path: Path) -> Path:
    root = tmp_path / "oet"
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True)
    for name in ("oet_client", "oet_uma", "oet_server"):
        path = bin_dir / name
        path.write_text("#!/bin/sh\n")
    return root


class UmaOetTests(unittest.TestCase):
    def test_get_oet_tools_prefers_oet_tools(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            oet_root = tmp / "new-oet"
            uma_root = tmp / "legacy-uma"
            oet_root.mkdir()
            uma_root.mkdir()
            with patch.dict(
                os.environ,
                {"OET_TOOLS": str(oet_root), "UMA_TOOLS": str(uma_root)},
                clear=False,
            ):
                self.assertEqual(get_oet_tools(), oet_root)

    def test_get_oet_tools_falls_back_to_uma_tools(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            uma_root = tmp / "legacy-uma"
            uma_root.mkdir()
            env = dict(os.environ)
            env.pop("OET_TOOLS", None)
            env["UMA_TOOLS"] = str(uma_root)
            with patch.dict(os.environ, env, clear=True):
                self.assertEqual(get_oet_tools(), uma_root)

    def test_get_oet_tools_missing_is_lazy_failure(self):
        env = dict(os.environ)
        env.pop("OET_TOOLS", None)
        env.pop("UMA_TOOLS", None)
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Set OET_TOOLS"):
                get_oet_tools()

    def test_parse_uma_spec_task_only_uses_default_model(self):
        spec = parse_uma_spec("omol")

        self.assertEqual(spec.task, "omol")
        self.assertEqual(spec.model, DEFAULT_UMA_MODEL)
        self.assertEqual(spec.device, "cpu")

    def test_parse_uma_spec_task_and_model(self):
        spec = parse_uma_spec("omol@uma-s-1p1", device="cuda")

        self.assertEqual(spec.task, "omol")
        self.assertEqual(spec.model, "uma-s-1p1")
        self.assertEqual(spec.device, "cuda")

    def test_parse_uma_spec_rejects_empty_parts(self):
        for value in ("", "   ", "@uma-s-1p1", "omol@"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_uma_spec(value)

    def test_server_orca_block_uses_oet_client_and_bind(self):
        with tempfile.TemporaryDirectory() as td:
            root = _fake_oet_root(Path(td))
            spec = parse_uma_spec("omol@uma-s-1p1")

            block = uma_orca_block(spec, server=True, bind="127.0.0.1:12345", tools=root)

        self.assertIn(f'ProgExt "{root}/bin/oet_client"', block)
        self.assertIn(
            'Ext_Params "-b 127.0.0.1:12345 -t omol -m uma-s-1p1 -d cpu"',
            block,
        )

    def test_standalone_orca_block_uses_oet_uma_with_optional_args(self):
        with tempfile.TemporaryDirectory() as td:
            root = _fake_oet_root(Path(td))
            spec = parse_uma_spec(
                "omol@uma-s-1p1",
                device="cuda",
                cache_dir="/tmp/fairchem-cache",
                offline=True,
            )

            block = uma_orca_block(spec, server=False, tools=root)

        self.assertIn(f'ProgExt "{root}/bin/oet_uma"', block)
        self.assertNotIn("-b 127.0.0.1", block)
        self.assertIn(
            'Ext_Params "-t omol -m uma-s-1p1 -d cuda -c /tmp/fairchem-cache -o True"',
            block,
        )

    def test_installed_oet_smoke(self):
        root_value = os.environ.get("OET_TOOLS") or os.environ.get("UMA_TOOLS")
        if not root_value:
            self.skipTest("OET_TOOLS/UMA_TOOLS is not configured")
        root = get_oet_tools()
        server = root / "bin" / "oet_server"
        client = root / "bin" / "oet_client"
        if not server.exists() or not client.exists():
            self.skipTest("OET 2 bin scripts are not installed")

        self.assertTrue(
            subprocess.run([server, "--version"], check=True, capture_output=True, text=True).stdout
        )
        self.assertTrue(
            subprocess.run([client, "--help"], check=True, capture_output=True, text=True).stdout
        )

        with tempfile.TemporaryDirectory() as td:
            with uma_server(log_dir=td, server_cores=1, memory_per_thread_mib=500):
                pass

    def test_uma_server_removes_success_log_with_on_failure_policy(self):
        root_value = os.environ.get("OET_TOOLS") or os.environ.get("UMA_TOOLS")
        if not root_value:
            self.skipTest("OET_TOOLS/UMA_TOOLS is not configured")
        root = get_oet_tools()
        if not (root / "bin" / "oet_server").exists():
            self.skipTest("OET 2 server script is not installed")

        with tempfile.TemporaryDirectory() as td:
            with uma_server(log_dir=td, server_cores=1, memory_per_thread_mib=500) as server:
                log_path = Path(server.log_path)

            self.assertFalse(log_path.exists())

    def test_uma_server_preserves_failure_log_with_default_policy(self):
        root_value = os.environ.get("OET_TOOLS") or os.environ.get("UMA_TOOLS")
        if not root_value:
            self.skipTest("OET_TOOLS/UMA_TOOLS is not configured")
        root = get_oet_tools()
        if not (root / "bin" / "oet_server").exists():
            self.skipTest("OET 2 server script is not installed")

        with tempfile.TemporaryDirectory() as td:
            log_dir = Path(td) / "preserved"
            with self.assertRaisesRegex(RuntimeError, "forced failure"):
                with uma_server(
                    log_dir=str(log_dir),
                    server_cores=1,
                    memory_per_thread_mib=500,
                ):
                    raise RuntimeError("forced failure")

            self.assertTrue(list(log_dir.glob("oet_uma_server_*.log")))


if __name__ == "__main__":
    unittest.main()
