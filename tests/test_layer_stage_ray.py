import json
import os
import tempfile
import threading
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.layer_stage_ray import make_worker_actor, ray_get_bounded, worker_command, worker_environment, worker_node_id


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

    def test_ray_actor_gpu_selection_is_a_local_visible_index(self):
        base = {"exe": "strata.exe", "gpu": 0, "args": ["--pack", "pack"]}
        self.assertNotIn("--gpu", worker_command(base, "0.0.0.0:1", 2))
        for args in (["--gpu", "0"], ["--gpu", "1"], ["--gpu=0"], ["--gpu", "0,1"]):
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, "override Ray's CUDA_VISIBLE_DEVICES"):
                worker_command({"exe": "strata.exe", "args": args}, "0.0.0.0:1", 2)
        with self.assertRaisesRegex(ValueError, "local index 0"):
            worker_command({"exe": "strata.exe", "gpu": 1, "args": ["--pack", "pack"]}, "0.0.0.0:1", 2)

    def test_worker_environment_merges_config_and_preserves_ray_environment(self):
        cfg = {"env": {"STRATA_TEST_SETTING": 7}, "lib_dirs": ["/cuda/lib"]}
        lib_path = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        env = worker_environment(cfg, "secret", {lib_path: "/usr/bin", "RAY_VISIBLE": "1",
                                                   "CUDA_VISIBLE_DEVICES": "GPU-aaa"})
        self.assertEqual(env["STRATA_TEST_SETTING"], "7")
        self.assertEqual(env["STRATA_RPC_TOKEN"], "secret")
        self.assertEqual(env["RAY_VISIBLE"], "1")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "GPU-aaa")
        self.assertTrue(env[lib_path].startswith("/cuda/lib" + os.pathsep))

    def test_worker_environment_rejects_gpu_visibility_override_or_missing_assignment(self):
        assigned = {"CUDA_VISIBLE_DEVICES": "GPU-aaa"}
        with self.assertRaisesRegex(ValueError, "cannot override"):
            worker_environment({"env": {"CUDA_VISIBLE_DEVICES": "0"}}, "secret", assigned)
        with self.assertRaisesRegex(ValueError, "did not assign"):
            worker_environment({}, "secret", {})
        with self.assertRaisesRegex(ValueError, "exactly one"):
            worker_environment({}, "secret", {"CUDA_VISIBLE_DEVICES": "0,1"})
        with self.assertRaisesRegex(ValueError, "exactly one"):
            worker_environment({}, "secret", {"CUDA_VISIBLE_DEVICES": "0,"})

    def test_bounded_ray_get_only_gets_completed_refs(self):
        class FakeRay:
            def __init__(self, ready):
                self.ready = ready
                self.got = []

            def wait(self, refs, timeout, num_returns):
                return (refs if self.ready else [], [])

            def get(self, ref):
                self.got.append(ref)
                return "result"

        pending = object()
        ray = FakeRay(False)
        with self.assertRaisesRegex(TimeoutError, "worker startup"):
            ray_get_bounded(ray, pending, 0.01, "worker startup")
        self.assertEqual(ray.got, [])
        ray.ready = True
        self.assertEqual(ray_get_bounded(ray, pending, 0.01, "worker startup"), "result")
        self.assertEqual(ray.got, [pending])

    def test_worker_start_timeout_terminates_child_and_keeps_ray_gpu_assignment(self):
        class FakeRay:
            @staticmethod
            def remote(**_options):
                return lambda actor: actor

        class BlockingOutput:
            def __init__(self, stopped):
                self.stopped = stopped

            def __iter__(self):
                self.stopped.wait(1)
                return iter(())

            def close(self):
                self.stopped.set()

        class FakeProcess:
            def __init__(self, **kwargs):
                self.command = kwargs["command"]
                self.env = kwargs["env"]
                self.stopped = threading.Event()
                self.stdout = BlockingOutput(self.stopped)
                self.exit_code = None
                self.terminated = False

            def poll(self):
                return self.exit_code

            def terminate(self):
                self.terminated = True
                self.exit_code = 0
                self.stopped.set()

            def kill(self):
                self.terminate()

            def wait(self, timeout):
                self.stopped.wait(timeout)
                return self.exit_code

        with tempfile.TemporaryDirectory() as temp:
            config_path = Path(temp) / "worker.json"
            config_path.write_text(json.dumps({"exe": "strata.exe", "args": []}), encoding="utf-8")
            process = FakeProcess.__new__(FakeProcess)

            def spawn(*args, **kwargs):
                FakeProcess.__init__(process, command=args[0], env=kwargs["env"])
                return process

            actor = make_worker_actor(FakeRay)()
            with patch("tools.layer_stage_ray.subprocess.Popen", side_effect=spawn), \
                    patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-ray-assigned"}):
                with self.assertRaisesRegex(TimeoutError, "did not become ready"):
                    actor.start(str(config_path), "0.0.0.0:39001", 24, "secret", 0.05)
            self.assertTrue(process.terminated)
            self.assertEqual(process.env["CUDA_VISIBLE_DEVICES"], "GPU-ray-assigned")
            self.assertNotIn("--gpu", process.command)
            self.assertFalse(actor.reader.is_alive())

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
