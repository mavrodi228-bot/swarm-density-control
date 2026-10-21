"""Functional tests for checkpoint integrity and isolated lab training."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
import torch

import ml_lab
from model import SwarmUNet, load_model


class LabTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix="agmen-lab-test-")
        self.environment = patch.dict(os.environ, {
            "AGMEN_DATA_DIR": self.folder.name, "AGMEN_PROFILE": "local",
            "AGMEN_MAX_RSS_MB": "4096",
        })
        self.environment.start()
        app = FastAPI()
        app.include_router(ml_lab.router)
        self.client = TestClient(app)

    def tearDown(self):
        for process in list(ml_lab._WORKER_PROCESSES.values()):
            process.wait(timeout=30)
        ml_lab._WORKER_PROCESSES.clear()
        self.environment.stop()
        self.folder.cleanup()

    def test_missing_weights_fail_and_nested_weights_preserve_architecture(self):
        with self.assertRaises(FileNotFoundError):
            load_model(Path(self.folder.name) / "missing.pt")
        path = Path(self.folder.name) / "nested.pt"
        original = SwarmUNet()
        torch.save({"model_state_dict": original.state_dict(), "epoch": 2}, path)
        restored = load_model(path)
        self.assertFalse(restored.training)
        for name, parameter in original.state_dict().items():
            self.assertTrue(torch.equal(parameter, restored.state_dict()[name]))

    def test_split_is_deterministic_disjoint_and_exhaustive(self):
        split = ml_lab._split_indices(64, 123)
        self.assertEqual(split, ml_lab._split_indices(64, 123))
        self.assertNotEqual(split, ml_lab._split_indices(64, 124))
        groups = [set(indices) for indices in split]
        self.assertEqual(set.union(*groups), set(range(64)))
        self.assertTrue(all(groups[a].isdisjoint(groups[b]) for a, b in [(0, 1), (0, 2), (1, 2)]))

    def test_config_limits_paths_and_public_profile(self):
        for value in ({"epochs": 51}, {"batch_size": 0}, {"samples": 2048}, {"grid_size": 47}, {"learning_rate": 4}, {"optimizer": "shell"}):
            response = self.client.post("/api/lab/runs", json=value)
            self.assertEqual(response.status_code, 422, response.text)
        response = self.client.post("/api/lab/runs", json={"checkpoint": "../../secret.pt"})
        self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(self.client.get("/api/lab/runs/not-a-run").status_code, 404)
        with patch.dict(os.environ, {"AGMEN_PROFILE": "public"}):
            self.assertEqual(self.client.post("/api/lab/runs", json={}).status_code, 403)
        bundled = self.client.get("/api/lab/models").json()[0]
        self.assertEqual(bundled["id"], "bundled")
        self.assertFalse(bundled["validated"])
        self.assertFalse(bundled["hardware_promoted"])
        self.assertNotIn("path", bundled)

    def _wait_for(self, run_id, target, timeout=180):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            run = self.client.get(f"/api/lab/runs/{run_id}").json()
            if run["status"] in target:
                return run
            if run["status"] == "failed":
                log = (ml_lab._run_dir(run_id) / "worker.log").read_text(encoding="utf-8", errors="replace")
                self.fail(f"Training failed: {run['error']}\n{log}")
            time.sleep(0.1)
        self.client.post(f"/api/lab/runs/{run_id}/stop")
        self.fail(f"Worker timed out waiting for {target}: {run}")

    def test_actual_worker_stop_resume_and_held_out_evaluation(self):
        response = self.client.post("/api/lab/runs", json={
            "epochs": 2, "samples": 16, "batch_size": 4,
            "grid_size": 24, "checkpoint": None, "seed": 19,
        })
        self.assertEqual(response.status_code, 202, response.text)
        run_id = response.json()["id"]
        # An active worker reserves the shared computation slot.
        self.assertEqual(self.client.post("/api/lab/runs", json={"checkpoint": None}).status_code, 409)
        stopped = self.client.post(f"/api/lab/runs/{run_id}/stop")
        self.assertEqual(stopped.status_code, 200, stopped.text)
        self._wait_for(run_id, {"stopped"})
        checkpoint = torch.load(ml_lab._run_dir(run_id) / "checkpoint.pt", map_location="cpu", weights_only=True)
        for key in ["model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "best_model_state_dict", "torch_rng_state", "batch_offset"]:
            self.assertIn(key, checkpoint)
        response = self.client.post(f"/api/lab/runs/{run_id}/resume")
        self.assertEqual(response.status_code, 202, response.text)
        run = self._wait_for(run_id, {"completed"})
        self.assertEqual(len(run["history"]), 2)
        self.assertEqual(run["model_id"], run_id)
        self.assertTrue(run["validated"])
        self.assertTrue(run["dataset"]["disjoint"])
        self.assertGreater(run["metrics"]["test_samples"], 0)
        self.assertGreaterEqual(run["metrics"]["test_mae"], 0)
        self.assertFalse(run["hardware_promoted"])
        restored = load_model(ml_lab._run_dir(run_id) / "best.pt")
        self.assertEqual(tuple(restored(torch.zeros(1, 3, 24, 24)).shape), (1, 2, 24, 24))
        models = self.client.get("/api/lab/models").json()
        self.assertTrue(any(model["id"] == run_id and model["validated"] for model in models))


if __name__ == "__main__":
    unittest.main()
