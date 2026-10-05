"""Tests for LoRA Fine-Tuning API."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_generate_dataset():
    r = client.post("/generate-dataset", params={"n_samples": 20})
    assert r.status_code == 200
    assert r.json()["metrics"]["n_samples"] == 20


def test_train_mock():
    payload = {"n_samples": 20, "epochs": 1, "lora_r": 4}
    r = client.post("/train", json=payload)
    assert r.status_code == 200
    assert "metrics" in r.json()


def test_evaluate():
    r = client.post("/evaluate")
    assert r.status_code == 200
    body = r.json()
    assert "base_accuracy" in body
    assert "improvement_pct" in body


def test_infer_valid():
    payload = {"instruction": "What is LoRA?", "max_new_tokens": 50}
    r = client.post("/infer", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "response" in body
    assert body["mode"] in ("real", "mock")


def test_infer_invalid():
    r = client.post("/infer", json={"instruction": "a"})  # too short
    assert r.status_code == 422
