import sys
import pathlib
import json
import pandas as pd

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'backend'))
# Import the forecasting helpers. Use a dynamic fallback import so static analyzers
# (like Pylance) that don't pick up the runtime-modified sys.path won't always
# show a missing-import warning in the editor.
try:
    from app.core.forecasting import prepare_df, forecast_timeseries  # type: ignore
except Exception:
    # dynamic import directly from file path as a fallback
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "app.core.forecasting",
        str(root / 'backend' / 'app' / 'core' / 'forecasting.py')
    )
    forecasting = importlib.util.module_from_spec(spec)
    loader = spec.loader
    if loader is not None:
        loader.exec_module(forecasting)
        prepare_df = forecasting.prepare_df
        forecast_timeseries = forecasting.forecast_timeseries
    else:
        raise ImportError('Could not import forecasting module')
import asyncio

p = None
if len(sys.argv) > 1:
    p = pathlib.Path(sys.argv[1])
else:
    # prefer repo data file if present
    p = root / 'data' / 'harga_laptop.csv'

print('Using file:', p)
if not p.exists():
    print('File not found:', p)
    sys.exit(2)

try:
    raw_df = pd.read_csv(p, encoding='utf-8')
except Exception:
    # try a fallback encoding for messy CSVs
    raw_df = pd.read_csv(p, encoding='latin-1')
df = prepare_df(raw_df)
print('Prepared df shape:', df.shape)
print(df.head().to_string())
res = asyncio.run(forecast_timeseries(df, periods=12))
print('Best model:', res.get('best_model'))
print('Metrics:', json.dumps(res.get('metrics'), indent=2))
print('\nForecast sample (first 5):')
for t, v in zip(res['forecast_data']['timestamp'][:5], res['forecast_data']['value'][:5]):
    print(t, v)
