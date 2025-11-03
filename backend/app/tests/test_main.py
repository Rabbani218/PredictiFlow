
import sys
import pathlib
import pytest

# Ensure backend package is importable (add backend/ to sys.path)
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'backend'))

pd = pytest.importorskip('pandas')
pytest.importorskip('fastapi')
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_predict_endpoint_happy_path():
    # read sample CSV bytes to simulate upload
    sample_path = ROOT / 'data' / 'sample_sales.csv'
    assert sample_path.exists(), f"Sample CSV not found at {sample_path}"
    with open(sample_path, "rb") as f:
        files = {"file": ("sample_sales.csv", f, "text/csv")}
        response = client.post("/predict", files=files)

    assert response.status_code == 200, response.text
    data = response.json()
    # Basic shape checks
    assert "historical_data" in data
    assert "forecast_data" in data
    assert "best_model" in data
    assert "metrics" in data
    assert isinstance(data["historical_data"]["timestamp"], list)
    assert isinstance(data["forecast_data"]["timestamp"], list)
