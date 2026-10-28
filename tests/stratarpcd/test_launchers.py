"""Launch-script dry runs: no models, GPUs or Windows filesystem required."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
PWSH = os.environ.get("RPCD_PWSH") or shutil.which("powershell") or shutil.which("pwsh")


@unittest.skipUnless(PWSH, "PowerShell is not installed")
class Launchers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="rpcd scripts ")
        self.root = Path(self.tmp.name)
        self.engine = self.root / "engine-rpcd"
        self.engine.mkdir()
        for script in (ROOT / "scripts/stratarpcd").glob("start-*.ps1"):
            shutil.copy2(script, self.engine)
        (self.engine / "strata-dsv41.exe").touch()

    def tearDown(self):
        self.tmp.cleanup()

    def run_script(self, role, config, *switches):
        path = self.root / "input.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        env = dict(os.environ, STRATA_RPC_TOKEN="test-only")
        args = [PWSH, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                str(self.engine / f"start-{role}.ps1"), "-Config", str(path), "-DryRun", *switches]
        if role == "coordinator":
            args += ["-Worker", "192.168.50.143:39501"]
        result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_coordinator_preserves_boolean_flags_and_pack(self):
        cfg = json.loads((ROOT / "configs/stratarpcd/strata-stratarpcd-coordinator.json").read_text())
        self.run_script("coordinator", cfg, "-Context", "1048576", "-Wire", "bf16", "-Port", "8081")
        out = json.loads((self.root / "strata-stratarpcd-coordinator.json").read_text())
        args = out["args"]
        for flag in ("--pack", "--max-context", "--dspark", "--experts", "--layer-stage-client",
                     "--layer-stage-vision", "--layer-stage-sessions"):
            self.assertEqual(args.count(flag), 1, flag)
        self.assertEqual(args[args.index("--dspark") + 1], "none")
        self.assertEqual(out["slot_save_path"], r"C:\StrataRPCD\sessions")
        self.assertIn("2.25bpw", out["tokenizer"])

    def test_worker_uses_its_own_session_directory(self):
        cfg = json.loads((ROOT / "configs/stratarpcd/strata-stratarpcd-worker.json").read_text())
        out = self.run_script("worker", cfg, "-Context", "1048576", "-Wire", "bf16")
        self.assertIn("--layer-stage-session-dir C:\\StrataRPCD\\sessions-worker", out)
        self.assertIn("--layer-stage-vision", out)
        self.assertIn("--gpus 0 --experts 6", out)
        self.assertNotIn("--layer-stage-client", out)

    def test_features_are_opt_in_for_legacy_configs(self):
        cfg = {"exe": "unused.exe", "args": ["--pack", "p", "--vision", "off", "--max-context", "8192"], "port": 8081}
        self.run_script("coordinator", cfg)
        args = json.loads((self.root / "strata-stratarpcd-coordinator.json").read_text())["args"]
        self.assertNotIn("--layer-stage-sessions", args)
        self.assertNotIn("--layer-stage-vision", args)
        self.run_script("coordinator", cfg, "-Vision", "-Sessions")
        out = json.loads((self.root / "strata-stratarpcd-coordinator.json").read_text())
        self.assertIn("--layer-stage-sessions", out["args"])
        self.assertEqual(out["args"][out["args"].index("--vision") + 1], "on")
        self.assertTrue(out["slot_save_path"].endswith("sessions"))


if __name__ == "__main__":
    unittest.main()
