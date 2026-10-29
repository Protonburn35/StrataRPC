"""StrataRPCG's disk sessions in the server (POST /slots/0?action=save|restore): the file names it takes, the exchange
with the engine (SAVE / RESTORE, SWAIT, SERR, SAVED / RESTORED) and the answers - without an engine or a GPU."""
import os
import queue
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serve import server as S  # noqa: E402


class FakeStdin:
    def __init__(self):
        self.written = []

    def write(self, s):
        self.written.append(s)

    def flush(self):
        pass


class FakeProc:
    def __init__(self):
        self.stdin = FakeStdin()
        self.killed = False

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return 0

    def poll(self):
        return None


def engine_with(lines):
    e = S.StrataEngine.__new__(S.StrataEngine)
    e.proc = FakeProc()
    e.lines = queue.Queue()
    for ln in lines:
        e.lines.put(ln)
    e.exit_code = lambda: 0
    return e


class FileNames(unittest.TestCase):
    def test_plain_names(self):
        self.assertIsNone(S.slot_filename_problem("chat-1.bin"))
        self.assertIsNone(S.slot_filename_problem("my chat.bin"))

    def test_refused_names(self):
        for bad in ("", "../x", "a/b", "a\\b", "C:x", "a:b", ".hidden", "x.", "x ", "NUL.txt", "com1", "a\nb",
                    "x" * 201, None, 3):
            self.assertIsNotNone(S.slot_filename_problem(bad), bad)

    def test_dir(self):
        with tempfile.TemporaryDirectory() as d:
            p = S.slot_save_dir("sessions", d)
            self.assertTrue(os.path.isdir(p))
            self.assertTrue(os.path.isabs(p))
            with self.assertRaises(ValueError):
                S.slot_save_dir("")


class Exchange(unittest.TestCase):
    def test_save(self):
        e = engine_with(["NOTE something\n", "SWAIT worker 600\n", "SWAIT disk 600\n", "SAVED 1234 5678 42.5\n"])
        r = e.session_file("save", "/tmp/x.bin")
        self.assertEqual(r, {"tokens": 1234, "bytes": 5678, "ms": 42.5})
        self.assertEqual(e.proc.stdin.written, ["SAVE /tmp/x.bin\n"])

    def test_restore(self):
        e = engine_with(["RESTORED 10 20 1.0\n"])
        self.assertEqual(e.session_file("restore", "/tmp/y.bin")["tokens"], 10)
        self.assertEqual(e.proc.stdin.written, ["RESTORE /tmp/y.bin\n"])

    def test_refused(self):
        e = engine_with(["SERR invalid 0 the worker's half: missing\n"])
        with self.assertRaises(S.SessionRefused) as c:
            e.session_file("restore", "/tmp/z.bin")
        self.assertEqual(c.exception.status, 400)
        self.assertIn("worker", str(c.exception))
        e = engine_with(["SERR storage 1 the disk is full\n"])
        with self.assertRaises(S.SessionRefused) as c:
            e.session_file("save", "/tmp/z.bin")
        self.assertEqual(c.exception.status, 507)
        self.assertTrue(c.exception.published)

    def test_out_of_step(self):
        for bad in ("T 5\n", "SAVED 1 2\n", "SAVED -1 2 3\n", "SWAIT disk 99999\n", "SERR weird 0 x\n"):
            e = engine_with([bad])
            with self.assertRaises(S.EngineDied):
                e.session_file("save", "/tmp/q.bin")
            self.assertTrue(e.proc.killed, bad)

    def test_engine_gone(self):
        e = engine_with([None])
        with self.assertRaises(S.EngineDied):
            e.session_file("save", "/tmp/q.bin")

    def test_silence(self):
        e = engine_with([])
        t0 = time.monotonic()
        with self.assertRaises(S.EngineDied):
            e.session_file("save", "/tmp/q.bin", silence_s=0.3)
        self.assertLess(time.monotonic() - t0, 5)

    def test_bad_command(self):
        e = engine_with([])
        with self.assertRaises(ValueError):
            e.session_file("save", "/tmp/a\nb")
        with self.assertRaises(ValueError):
            e.session_file("delete", "/tmp/a")


class FakeEngine:
    multi = 0

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def alive(self):
        return True

    def session_file(self, action, path):
        self.calls.append((action, path))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def service(engine, slot_dir):
    svc = S.Service.__new__(S.Service)
    svc.engine = engine
    svc.slot_save_path = slot_dir
    svc.fifo = S.FairLock()
    svc.status = {"busy": False, "queued": 0}
    svc.status_lock = threading.Lock()
    svc.last_request_at = None
    return svc


class Action(unittest.TestCase):
    def test_off(self):
        code, body = service(FakeEngine({}), None).slot_action("0", "save", "a.bin")
        self.assertEqual(code, 501)

    def test_save_and_restore(self):
        with tempfile.TemporaryDirectory() as d:
            eng = FakeEngine({"tokens": 7, "bytes": 99, "ms": 3.0})
            svc = service(eng, d)
            code, body = svc.slot_action("0", "save", "chat.bin")
            self.assertEqual(code, 200)
            self.assertEqual(body["n_saved"], 7)
            self.assertEqual(eng.calls, [("save", os.path.join(d, "chat.bin"))])
            code, body = svc.slot_action("0", "restore", "nothing.bin")
            self.assertEqual(code, 404)
            Path(d, "chat.bin").write_bytes(b"x")
            code, body = svc.slot_action("0", "restore", "chat.bin")
            self.assertEqual(code, 200)
            self.assertEqual(body["n_restored"], 7)
            self.assertFalse(svc.status["busy"])
            self.assertEqual(svc.status["queued"], 0)

    def test_refusals(self):
        with tempfile.TemporaryDirectory() as d:
            svc = service(FakeEngine(S.SessionRefused("storage", "full")), d)
            code, body = svc.slot_action("0", "save", "a.bin")
            self.assertEqual(code, 507)
            self.assertEqual(body["error"]["kind"], "storage")
            self.assertEqual(svc.slot_action("1", "save", "a.bin")[0], 400)
            self.assertEqual(svc.slot_action("0", "delete", "a.bin")[0], 400)
            self.assertEqual(svc.slot_action("0", "save", "../a.bin")[0], 400)
            multi = FakeEngine({})
            multi.multi = 2
            self.assertEqual(service(multi, d).slot_action("0", "save", "a.bin")[0], 501)


if __name__ == "__main__":
    unittest.main()
