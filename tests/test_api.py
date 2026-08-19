import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
import api.app as api_app

client = TestClient(api_app.app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict(monkeypatch):

    # Fake model prediction
    class FakeModel:
        def predict(self, data):
            return [20.0]

    monkeypatch.setattr(api_app, "model", FakeModel())

    # Fake database cursor
    class FakeCursor:
        rowcount = 1

        def execute(self, query, params):
            pass

        def fetchone(self):
            return [999]

        def close(self):
            pass

    # Fake database connection
    class FakeConnection:
        def cursor(self):
            return FakeCursor()

        def commit(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(
        api_app.psycopg2,
        "connect",
        lambda **kwargs: FakeConnection()
    )

    payload = {
        "DAY_OF_MONTH": 1,
        "DAY_OF_WEEK": 4,
        "OP_UNIQUE_CARRIER": "AA",
        "TAIL_NUM": "N101NN",
        "OP_CARRIER_FL_NUM": 1480,
        "ORIGIN_AIRPORT_ID": 12892,
        "DEST_AIRPORT_ID": 10721,
        "CRS_DEP_TIME": 1358,
        "CRS_ARR_TIME": 2229,
        "CRS_ELAPSED_TIME": 331
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json()["prediction_id"] == 999
    assert response.json()["predicted_arrival_delay_minutes"] == 20.0
    assert response.json()["delay_over_15_minutes"] is True