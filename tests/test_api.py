from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from api import create_app
from health_db import init_db


@pytest.fixture
def client() -> TestClient:
    temp_dir = Path(".tmp-tests") / uuid4().hex
    temp_dir.mkdir(parents=True, exist_ok=True)
    db_path = temp_dir / "health.db"
    init_db(db_path)
    app = create_app(db_path=db_path)
    return TestClient(app)


def test_get_health_returns_latest_record_for_user_id(client: TestClient) -> None:
    response = client.get("/health", params={"id": "user_001"})

    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == "user_001"
    assert body["updated_at"] == "2026-05-01 08:30:00"
    assert body["weight_kg"] == 86.5


def test_get_health_returns_404_for_unknown_user_id(client: TestClient) -> None:
    response = client.get("/health", params={"id": "missing-user"})

    assert response.status_code == 404
    assert response.json()["detail"] == "Health record not found"


def test_get_health_lists_returns_all_records(client: TestClient) -> None:
    response = client.get("/health/lists")

    assert response.status_code == 200
    body = response.json()
    assert len(body) >= 5
    assert body[0]["uuid"] == "a1b2c3d4-0001-0001-0001-000000000001"
    assert {item["user_id"] for item in body} >= {
        "user_001",
        "user_002",
        "user_003",
        "user_004",
    }


def test_post_health_creates_new_record(client: TestClient) -> None:
    payload = {
        "name": "Test User",
        "gender": "male",
        "age_years": 30,
        "age_months": 0,
        "age_days": 0,
        "user_id": "user_999",
        "height_cm": 180.0,
        "weight_kg": 80.0,
        "hbpm": 75,
        "blood_pressure": "120/80",
        "o2": 98.0,
        "body_temp": 36.7,
    }

    response = client.post("/health", json=payload)

    assert response.status_code == 201
    body = response.json()
    assert body["user_id"] == "user_999"
    assert body["name"] == "Test User"
    assert body["bmi"] == pytest.approx(24.7, rel=0, abs=0.1)
    assert body["bmr"] == pytest.approx(1780.0, rel=0, abs=0.1)
    assert body["uuid"]
    assert body["updated_at"]

    fetch = client.get("/health", params={"id": "user_999"})
    assert fetch.status_code == 200
    assert fetch.json()["uuid"] == body["uuid"]
