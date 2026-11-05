"""Executed autonomy scenarios; all trajectories come from the local controller."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from agmen_simulation import simulate
from edge import EdgeRuntime, build_package, load_package, PackageError


class EdgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = {"drone_count": 6, "steps": 160, "max_speed": 4., "safe_distance": 2.5,
                      "seed": 13, "controller": "baseline", "obstacle": "central", "scenario": "clusters", "targets": []}
        cls.mission = {"id": "abcd1234abcd", "status": "completed", "name": "Offline test mission",
                       "config": cls.config, **simulate(cls.config)}

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="agmen-edge-test-")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def package(self, policy="continue", coordination="independent", drone_id=1):
        return build_package(self.mission, drone_id, policy, coordination)

    def runtime(self, name, **kwargs):
        return EdgeRuntime(self.package(**kwargs), self.root / name)

    def test_archive_integrity_deterministic_identity_and_missing_map(self):
        path = self.root / "mission.zip"
        metadata = build_package(self.mission, 1, output_path=path)
        self.assertEqual(metadata["id"], self.package()["manifest"]["id"])
        self.assertEqual(load_package(path)["manifest"]["id"], metadata["id"])
        self.assertIsNone(load_package(path)["map"]["geographic_origin"])
        with zipfile.ZipFile(path) as archive:
            members = {name: archive.read(name) for name in archive.namelist()}
        broken = self.root / "tampered.zip"
        with zipfile.ZipFile(broken, "w") as archive:
            for name, data in members.items():
                archive.writestr(name, b"{}" if name == "fields.json" else data)
        with self.assertRaisesRegex(PackageError, "integrity"):
            load_package(broken)
        missing = self.root / "missing-map.zip"
        with zipfile.ZipFile(missing, "w") as archive:
            for name, data in members.items():
                if name != "map.json":
                    archive.writestr(name, data)
        with self.assertRaises(PackageError):
            load_package(missing)
        traversal = self.root / "traversal.zip"
        with zipfile.ZipFile(traversal, "w") as archive:
            for name, data in members.items():
                archive.writestr(name, data)
            archive.writestr("../escape.py", b"pass")
        with self.assertRaises(PackageError):
            load_package(traversal)
        corrupted = self.package()
        corrupted["fields"]["flux_x"][0][0] += 1
        with self.assertRaises(PackageError):
            EdgeRuntime(corrupted, self.root / "corrupted")

    def test_internet_before_and_during_execution_has_identical_real_trajectory_without_network(self):
        online = self.runtime("online")
        offline = self.runtime("offline")
        during = self.runtime("during")
        before = offline.snapshot()["position_m"]
        with patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden")), patch("socket.create_connection", side_effect=AssertionError("Network forbidden")), patch("socket.getaddrinfo", side_effect=AssertionError("Network forbidden")):
            offline.set_connectivity(internet=False)
            a = online.run_steps(40)
            b = offline.run_steps(40)
            during.run_steps(15)
            during.set_connectivity(internet=False)
            c = during.run_steps(25)
        np.testing.assert_allclose(a["position_m"], b["position_m"], atol=1e-12)
        np.testing.assert_allclose(a["position_m"], c["position_m"], atol=1e-12)
        np.testing.assert_allclose([row["distribution_mae"] for row in a["history"]], [row["distribution_mae"] for row in b["history"]], atol=1e-12)
        self.assertGreater(np.linalg.norm(np.asarray(before) - b["position_m"]), 1)
        self.assertGreater(b["metrics"]["task_fraction"], 0)
        self.assertEqual(b["mode"], "OFFLINE")
        self.assertEqual(b["last_confirmed_telemetry"]["step"], 40)  # independent ground radio
        self.assertFalse(b["telemetry_stale"])
        self.assertEqual(b["metrics"]["constraint_violations"], 0)
        self.assertTrue(all(row["speed_mps"] <= self.config["max_speed"] + 1e-8 for row in b["history"]))

    def test_ground_and_peer_loss_policies_and_stale_telemetry(self):
        hold = self.runtime("hold", policy="hold")
        hold.run_steps(5)
        start = hold.snapshot()
        hold.set_connectivity(ground_radio=False, peer_radio=False)
        result = hold.run_steps(10)
        np.testing.assert_allclose(result["position_m"], start["position_m"])
        self.assertEqual(result["status"], "holding")
        self.assertEqual(result["mode"], "LINK_LOST")
        self.assertEqual(result["last_confirmed_telemetry"]["step"], start["step"])
        self.assertTrue(result["telemetry_stale"])
        self.assertIsNotNone(result["predicted_position"])
        with self.assertRaises(ValueError):
            hold.sync()
        continuing = self.runtime("continue")
        continuing.set_connectivity(internet=False, ground_radio=False, peer_radio=False)
        result = continuing.run_steps(10)
        self.assertEqual(result["status"], "running")
        self.assertGreater(result["metrics"]["task_fraction"], 0)
        self.assertEqual(result["predicted_position"]["source"], "dead_reckoning_from_last_confirmed")
        self.assertNotAlmostEqual(result["predicted_position"]["x"], result["simulated_position"]["x"])
        self.assertFalse(result["predicted_position"]["obstacle_guarantee"])
        returning = self.runtime("return", policy="return")
        returning.run_steps(20)
        returning.set_connectivity(ground_radio=False, peer_radio=False)
        distance_before = np.linalg.norm(np.asarray(returning.snapshot()["position_m"]) - returning.mission["home_position_m"])
        result = returning.run_steps(40)
        self.assertLess(np.linalg.norm(np.asarray(result["position_m"]) - returning.mission["home_position_m"]), distance_before)
        self.assertIn(result["status"], {"returning", "returned"})

    def test_restart_navigation_low_battery_and_restored_radio_sync(self):
        runtime = self.runtime("restart")
        runtime.set_connectivity(internet=False, ground_radio=False)
        before = runtime.run_steps(12)
        restored = EdgeRuntime(self.package(), self.root / "restart")
        np.testing.assert_allclose(restored.snapshot()["position_m"], before["position_m"])
        self.assertEqual(restored.snapshot()["step"], before["step"])
        restored.set_connectivity(gnss=False)
        stationary = restored.snapshot()["position_m"]
        restored.run_steps(5)
        np.testing.assert_allclose(restored.snapshot()["position_m"], stationary)
        self.assertEqual(restored.snapshot()["status"], "holding")
        restored.set_connectivity(gnss=True)
        restored.set_battery(12)
        result = restored.run_steps(2)
        self.assertEqual(result["status"], "returning")
        result = restored.set_connectivity(ground_radio=True)
        self.assertIsNotNone(result["last_sync"])
        self.assertEqual(result["last_sync"]["step"], result["step"])
        self.assertFalse(result["last_sync"]["cloud_upload"])
        self.assertFalse(result["connectivity"]["internet"])
        self.assertGreater(result["metrics"]["recovery_steps"], 0)
        self.assertFalse(result["telemetry_stale"])
        self.assertTrue((self.root / "restart" / "journal.jsonl").is_file())
        restored.set_navigation(False)
        result = restored.run_steps(2)
        self.assertEqual(result["status"], "holding")

    def test_peer_data_requires_explicit_fresh_radio_observations(self):
        runtime = self.runtime("peer", coordination="peer")
        peer = {"drone_id": 2, "position_m": [12., 35.], "step": 0}
        result = runtime.run_steps(1, peer_observations=[peer])
        self.assertEqual(result["peer_observations_used"], 1)
        runtime.set_connectivity(peer_radio=False)
        result = runtime.run_steps(1, peer_observations=[peer])
        self.assertEqual(result["peer_observations_used"], 0)
        runtime.set_connectivity(peer_radio=True)
        result = runtime.run_steps(12, peer_observations=[peer])
        self.assertEqual(result["peer_observations_used"], 0)  # stale after ten steps
        with self.assertRaises(ValueError):
            runtime.run_steps(501)

    def test_cli_validate_and_offline_restart(self):
        path = self.root / "mission.zip"
        build_package(self.mission, 1, output_path=path)
        guard = self.root / "network_guard"
        guard.mkdir()
        (guard / "sitecustomize.py").write_text(
            "import socket\n"
            "def deny(*args, **kwargs): raise AssertionError('Network calls are forbidden in the offline CLI test')\n"
            "socket.socket.connect = deny\n"
            "socket.socket.connect_ex = deny\n"
            "socket.create_connection = deny\n"
            "socket.getaddrinfo = deny\n", encoding="utf-8")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(guard) + os.pathsep + environment.get("PYTHONPATH", "")
        validated = subprocess.run([sys.executable, "-m", "edge", "validate", str(path)], check=True, capture_output=True, text=True, env=environment, timeout=30)
        self.assertTrue(json.loads(validated.stdout)["valid"])
        arguments = [sys.executable, "-m", "edge", "run", str(path), "--state", str(self.root / "cli"), "--steps", "3", "--offline"]
        first = subprocess.run(arguments, check=True, capture_output=True, text=True, env=environment, timeout=30)
        second = subprocess.run(arguments, check=True, capture_output=True, text=True, env=environment, timeout=30)
        self.assertEqual(json.loads(first.stdout)["step"], 3)
        self.assertEqual(json.loads(second.stdout)["step"], 6)
        self.assertFalse(json.loads(second.stdout)["connectivity"]["internet"])

    def test_included_neural_weights_run_local_inference_without_network(self):
        from ml_lab import get_inference_model
        config = {**self.config, "controller": "hybrid", "model_id": "bundled", "steps": 5}
        model, device = get_inference_model()
        mission = {"id": "abcd9876abcd", "status": "completed", "name": "Neural offline test", "config": config,
                   **simulate(config, model, device)}
        archive = self.root / "neural.zip"
        build_package(mission, 1, output_path=archive)
        package = load_package(archive)
        self.assertEqual(package["model"]["encoding"], "float32-json-v1")
        with patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden")), patch("socket.getaddrinfo", side_effect=AssertionError("Network forbidden")):
            runtime = EdgeRuntime(package, self.root / "neural")
            self.assertIsNotNone(runtime.model)
            runtime.set_connectivity(internet=False, ground_radio=False, peer_radio=False)
            result = runtime.run_steps(3)
        self.assertEqual(result["step"], 3)
        self.assertGreater(result["metrics"]["task_fraction"], 0)
        self.assertEqual(result["metrics"]["constraint_violations"], 0)


if __name__ == "__main__":
    unittest.main()
