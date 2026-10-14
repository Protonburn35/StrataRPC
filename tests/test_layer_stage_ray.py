import os
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.layer_stage_ray import worker_command, worker_environment, worker_node_id


class LayerStageRayHelpers(unittest.TestCase):
    def test_worker_command_preserves_model_args_and_adds_server_role(self):
        cfg = {"exe": r"C:\Strata\engine\strata.exe",
               "args": ["--pack", r"C:\Strata-data\pack", "--prefill", "2048", "--spec", "4"]}
        self.assertEqual(worker_command(cfg, "0.0.0.0:39001", 24), [
            cfg["exe"], "--serve", *cfg["args"], "--layer-stage-server", "0.0.0.0:39001",
            "--layer-stage-split", "24",
        ])

    def test_worker_command_does_not_duplicate_serve_or_accept_conflicting_role(self):
        cfg = {"exe": "strata.exe", "args": ["--serve", "--pack", "pack"]}
        self.assertEqual(worker_command(cfg, "0.0.0.0:1", 2).count("--serve"), 1)
        with self.assertRaisesRegex(ValueError, "remove layer-stage flags"):
            worker_command({"exe": "strata.exe", "args": ["--layer-stage-client", "host:1"]},
                           "0.0.0.0:1", 2)

    def test_worker_environment_merges_config_and_preserves_ray_environment(self):
        cfg = {"env": {"STRATA_TEST_SETTING": 7}, "lib_dirs": ["/cuda/lib"]}
        lib_path = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        env = worker_environment(cfg, "secret", {lib_path: "/usr/bin", "RAY_VISIBLE": "1"})
        self.assertEqual(env["STRATA_TEST_SETTING"], "7")
        self.assertEqual(env["STRATA_RPC_TOKEN"], "secret")
        self.assertEqual(env["RAY_VISIBLE"], "1")
        self.assertTrue(env[lib_path].startswith("/cuda/lib" + os.pathsep))

    def test_node_selection_requires_one_live_exact_address(self):
        nodes = [
            {"Alive": True, "NodeManagerAddress": "192.168.1.20", "NodeID": "host"},
            {"Alive": True, "NodeManagerAddress": "192.168.1.30", "NodeID": "worker"},
            {"Alive": False, "NodeManagerAddress": "192.168.1.30", "NodeID": "old"},
        ]
        self.assertEqual(worker_node_id(nodes, "192.168.1.30"), "worker")
        with self.assertRaisesRegex(ValueError, "found 0"):
            worker_node_id(nodes, "192.168.1.99")


if __name__ == "__main__":
    unittest.main()
