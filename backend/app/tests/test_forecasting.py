import sys
import pathlib
import pytest

# Add backend/ to sys.path so 'app' package can be imported
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'backend'))

pd = pytest.importorskip('pandas')
np = pytest.importorskip('numpy')
import asyncio

from app.core.forecasting import forecast_timeseries, prepare_df


def make_sample_df(n=60):
    dates = pd.date_range(start="2023-01-01", periods=n, freq='D')
    values = np.linspace(100, 200, n) + np.random.normal(0, 2, n)
    return pd.DataFrame({'ds': dates, 'y': values})


def test_forecast_timeseries_basic():
    df = make_sample_df(60)
    # use asyncio.run to create/close event loop cleanly
    result = asyncio.run(forecast_timeseries(df, periods=14))

    assert 'historical_data' in result
    assert 'forecast_data' in result
    assert isinstance(result['forecast_data']['timestamp'], list)
    assert len(result['forecast_data']['timestamp']) >= 14
