# python
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_ingest_and_query():
    texts = ["Hello world", "FastAPI is great", "We build RAG demos"]
    r = client.post("/ingest", json={"texts": texts})
    assert r.status_code == 200
    assert r.json()["inserted"] == len(texts)

    r2 = client.post("/query", json={"question": "What is FastAPI", "top_k": 2})
    assert r2.status_code == 200
    assert "results" in r2.json()
