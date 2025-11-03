import pandas as pd
import numpy as np
from typing import Dict, Tuple

# Optional imports for heavier models. If not available we fall back to simple baselines.
try:
    from prophet import Prophet  # type: ignore
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore
    from statsmodels.tsa.seasonal import seasonal_decompose  # type: ignore
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error  # type: ignore
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = np.where(np.abs(y_true) < 1e-8, 1e-8, y_true)
        return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


async def _train_prophet_or_baseline(data: pd.DataFrame, periods: int = 30) -> Tuple[str, pd.DataFrame]:
    """Train Prophet when available, otherwise return a simple naive forecast (last value repeated).

    Returns (model_name, forecast_df) where forecast_df contains 'ds' and 'yhat'.
    """
    df = data.copy()
    if PROPHET_AVAILABLE:
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        m.fit(df)
        future = m.make_future_dataframe(periods=periods)
        forecast = m.predict(future)[['ds', 'yhat']]
        return 'prophet', forecast

    # Baseline: repeat last value
    last_date = pd.to_datetime(df['ds'].iloc[-1])
    freq = (pd.to_datetime(df['ds'].iloc[1]) - pd.to_datetime(df['ds'].iloc[0])) if len(df) > 1 else pd.Timedelta(days=1)
    future_dates = [last_date + (i + 1) * freq for i in range(periods)]
    last_val = float(df['y'].iloc[-1])
    forecast = pd.DataFrame({'ds': future_dates, 'yhat': [last_val] * periods})
    return 'baseline_prophet', forecast


async def _train_arima_or_baseline(data: pd.DataFrame, periods: int = 30) -> Tuple[str, pd.DataFrame]:
    df = data.copy()
    if STATSMODELS_AVAILABLE:
        import warnings
        # Use relaxed constraints to avoid non-stationary/invertible starting-parameter warnings
        # and limit optimizer iterations to keep tests fast. Also suppress warnings emitted
        # by statsmodels during fitting so test output stays clean.
        model = SARIMAX(df['y'], order=(1, 1, 1), seasonal_order=(0, 0, 0, 0), enforce_stationarity=False, enforce_invertibility=False)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                # limit iterations for speed; lbfgs supports maxiter
                results = model.fit(disp=False, method='lbfgs', maxiter=50)
        except Exception:
            # fallback to default fit without extra args if the solver call fails
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                results = model.fit(disp=False)
        pred = results.get_forecast(steps=periods)
        forecast_index = pd.date_range(start=pd.to_datetime(df['ds'].iloc[-1]) + pd.Timedelta(days=1), periods=periods)
        forecast = pd.DataFrame({'ds': forecast_index, 'yhat': pred.predicted_mean.values})
        return 'arima', forecast

    # Baseline: linear extrapolation using last two points if available
    if len(df) >= 2:
        x = np.arange(len(df))
        coef = np.polyfit(x[-2:], df['y'].values[-2:], 1)
        future_x = np.arange(len(df), len(df) + periods)
        preds = np.polyval(coef, future_x)
    else:
        preds = np.array([float(df['y'].iloc[-1])] * periods)
    last_date = pd.to_datetime(df['ds'].iloc[-1])
    freq = (pd.to_datetime(df['ds'].iloc[1]) - pd.to_datetime(df['ds'].iloc[0])) if len(df) > 1 else pd.Timedelta(days=1)
    future_dates = [last_date + (i + 1) * freq for i in range(periods)]
    forecast = pd.DataFrame({'ds': future_dates, 'yhat': preds.tolist()})
    return 'baseline_arima', forecast


async def _decompose_or_empty(data: pd.DataFrame) -> Dict:
    # If statsmodels is available, attempt a decomposition only when there is
    # enough data for the requested period. seasonal_decompose requires at
    # least two full cycles (period * 2) of data; for period=30 that is 60
    # observations. If not enough data, fall back to a lightweight estimate
    # to avoid raising an exception in the pipeline.
    if STATSMODELS_AVAILABLE:
        try:
            y_non_na = pd.Series(data['y']).dropna()
            if len(y_non_na) >= 60:
                decomposition = seasonal_decompose(y_non_na, period=30, model='additive', extrapolate_trend='freq')
                return {
                    'trend': decomposition.trend.reindex(range(len(data))).bfill().ffill().tolist(),
                    'seasonal': decomposition.seasonal.reindex(range(len(data))).fillna(0).tolist(),
                    'residual': decomposition.resid.reindex(range(len(data))).fillna(0).tolist()
                }
            # Not enough data to decompose safely; fall through to fallback below
        except Exception:
            # If decomposition fails for any reason, continue to fallback
            pass
    # minimal decomposition fallback: rolling mean as trend, zeros for seasonal/residual
    trend = pd.Series(data['y']).rolling(window=min(7, max(1, len(data['y']))), min_periods=1).mean().tolist()
    return {
        'trend': [float(v) for v in trend],
        'seasonal': [0.0] * len(data['y']),
        'residual': [0.0] * len(data['y'])
    }


async def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    if SKLEARN_AVAILABLE:
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    else:
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = _safe_mape(y_true, y_pred)
    return {'mae': mae, 'rmse': rmse, 'mape': mape}


async def _train_exp_smoothing_or_baseline(data: pd.DataFrame, periods: int = 30) -> Tuple[str, pd.DataFrame]:
    df = data.copy()
    # Prefer statsmodels Holt-Winters if available
    if STATSMODELS_AVAILABLE:
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES  # type: ignore
            model = HWES(df['y'], trend='add', seasonal=None)
            res = model.fit()
            preds = res.forecast(steps=periods)
            forecast_index = pd.date_range(start=pd.to_datetime(df['ds'].iloc[-1]) + pd.Timedelta(days=1), periods=periods)
            return 'exp_smoothing', pd.DataFrame({'ds': forecast_index, 'yhat': preds.tolist()})
        except Exception:
            pass

    # Fallback: simple EWM trend extrapolation
    y = df['y'].values
    if len(y) < 2:
        preds = [float(y[-1]) if len(y) > 0 else 0.0] * periods
    else:
        span = min(10, len(y))
        ew = pd.Series(y).ewm(span=span).mean()
        last = float(y[-1])
        prev = float(ew.iloc[-2]) if len(ew) > 1 else float(ew.iloc[-1])
        slope = last - prev
        preds = [last + (i + 1) * slope for i in range(periods)]

    last_date = pd.to_datetime(df['ds'].iloc[-1])
    freq = (pd.to_datetime(df['ds'].iloc[1]) - pd.to_datetime(df['ds'].iloc[0])) if len(df) > 1 else pd.Timedelta(days=1)
    future_dates = [last_date + (i + 1) * freq for i in range(periods)]
    return 'baseline_exp_smoothing', pd.DataFrame({'ds': future_dates, 'yhat': preds})


async def forecast_timeseries(df: pd.DataFrame, periods: int = 30) -> Dict:
    """Main forecasting function that orchestrates the entire process and returns a serializable dict.

    The function is defensive: it standardizes input, handles missing values, uses lightweight fallbacks
    if heavier libraries aren't installed, and returns consistent output for the API.
    """
    # Standardize input: prefer using prepare_df which has robust column detection
    df = prepare_df(df)

    decomposition = await _decompose_or_empty(df)

    # Train models (or fallbacks): prophet, arima, exp_smoothing
    prophet_name, prophet_forecast = await _train_prophet_or_baseline(df, periods=periods)
    arima_name, arima_forecast = await _train_arima_or_baseline(df, periods=periods)
    exp_name, exp_forecast = await _train_exp_smoothing_or_baseline(df, periods=periods)

    # Evaluate on last `periods` points if we have them
    horizon = min(periods, len(df))
    y_true = df['y'].values[-horizon:]

    def _align_pred(forecast_df, which='tail'):
        try:
            if which == 'tail':
                return np.array(forecast_df['yhat'].values[-horizon:])
            return np.array(forecast_df['yhat'].values[:horizon])
        except Exception:
            val = float(forecast_df['yhat'].iloc[0]) if len(forecast_df['yhat'])>0 else 0.0
            return np.array([val]*horizon)

    prophet_pred = _align_pred(prophet_forecast, which='tail')
    arima_pred = _align_pred(arima_forecast, which='head')
    exp_pred = _align_pred(exp_forecast, which='head')

    prophet_metrics = await _calculate_metrics(y_true, prophet_pred)
    arima_metrics = await _calculate_metrics(y_true, arima_pred)
    exp_metrics = await _calculate_metrics(y_true, exp_pred)

    # Select best model by RMSE among all three
    metrics_map = {prophet_name: prophet_metrics, arima_name: arima_metrics, exp_name: exp_metrics}
    best_model = min(metrics_map.keys(), key=lambda k: metrics_map[k]['rmse'])

    # Choose forecast data accordingly
    chosen_forecast = ({prophet_name: prophet_forecast, arima_name: arima_forecast, exp_name: exp_forecast})[best_model]
    # Ensure forecast timestamps are strings for JSON
    forecast_timestamps = [str(x) for x in pd.to_datetime(chosen_forecast['ds']).astype(str).tolist()]
    forecast_values = [float(x) for x in chosen_forecast['yhat'].tolist()]

    return {
        'historical_data': {
            'timestamp': [str(x) for x in df['ds'].astype(str).tolist()],
            'value': [float(x) for x in df['y'].tolist()]
        },
        'forecast_data': {
            'timestamp': forecast_timestamps,
            'value': forecast_values
        },
        'best_model': best_model,
        'metrics': prophet_metrics if best_model == prophet_name else arima_metrics,
        'decomposition': decomposition
    }


# --- Synchronous helper wrappers used by notebooks and tests ---
def prepare_df(input_df_or_path):
    """Public helper: prepare a DataFrame or CSV path into standardized `ds`/`y` DataFrame."""
    if isinstance(input_df_or_path, str):
        df = pd.read_csv(input_df_or_path)
    else:
        df = input_df_or_path.copy()
    df = df.copy()

    # if too few columns, create a simple index-based date
    if len(df.columns) < 1:
        raise ValueError('Input must contain at least one column')

    # Helper: find date-like column
    date_col = None
    date_keywords = ['date', 'ds', 'time', 'tanggal', 'tgl', 'year', 'month']
    for c in df.columns:
        if any(k in c.lower() for k in date_keywords):
            date_col = c
            break

    # If no obvious date column, try parsing each column but avoid treating integer ID-like
    # columns as datetimes. We require that parsed datetimes fall into a plausible year range
    # (1900-2100) for the column to be considered a date candidate.
    if date_col is None:
        def _plausible_datetime_series(s: pd.Series) -> int:
            try:
                parsed = pd.to_datetime(s, errors='coerce', utc=False)
            except Exception:
                return 0
            non_na = parsed.notna()
            if non_na.sum() == 0:
                return 0
            yrs = parsed.dt.year.fillna(0).astype(int)
            plausible = ((yrs >= 1900) & (yrs <= 2100)).sum()
            # Only count entries that were parsed and have plausible years
            return int(plausible)

        best = None
        best_count = -1
        for c in df.columns:
            # Skip pure numeric id columns unless they explicitly look like dates
            count = _plausible_datetime_series(df[c])
            if count > best_count:
                best_count = count
                best = c
        # choose best only if it has reasonable non-null plausible datetime fraction
        if best is not None and best_count >= max(2, int(0.1 * len(df))):
            date_col = best

    # Value column detection: prefer common names or any numeric column.
    # Use more robust detection by parsing columns when necessary.
    value_col = _detect_value_column(df)

    if value_col is None:
        raise ValueError('Could not detect a numeric/value column in the input dataframe')

    # Build standardized DataFrame
    if date_col is None:
        # create a simple date index if missing
        df_std = pd.DataFrame({'ds': pd.date_range(start=pd.Timestamp.today().normalize(), periods=len(df), freq='D')})
        df_std['y'] = _parse_numeric_series(df[value_col])
    else:
        df_std = pd.DataFrame({'ds': pd.to_datetime(df[date_col], errors='coerce'), 'y': _parse_numeric_series(df[value_col])})

    df_std = df_std.sort_values('ds').reset_index(drop=True)
    # drop rows where value couldn't be parsed
    non_na_before = len(df_std)
    df_std = df_std[df_std['y'].notna()].reset_index(drop=True)
    if len(df_std) < 2:
        raise ValueError(f'Prepared dataframe has insufficient non-NaN values ({len(df_std)}) after parsing numeric columns; original columns tried: date_col={date_col}, value_col={value_col}')

    # interpolate small gaps and forward/backfill
    df_std['y'] = df_std['y'].interpolate().bfill().ffill()
    return df_std


def _parse_numeric_series(series: pd.Series) -> pd.Series:
    """Parse a pandas Series of numbers that may include currency symbols, thousands separators,
    or localized decimal commas. Returns a float series with coercion to NaN on failure."""
    s = series.astype(str).str.strip()
    # remove currency symbols and letters
    s = s.str.replace(r"[A-Za-z¥$€£Rp.,]", lambda m: m.group(0) if m.group(0) in ['.', ','] else '', regex=True)
    # Heuristic: if comma used as decimal separator (e.g., '1.234,56'), convert '.' thousands to '' and ',' to '.'
    def fix_val(v: str) -> str:
        if v.count(',') == 1 and v.count('.') >= 1:
            # assume format 1.234,56
            v = v.replace('.', '').replace(',', '.')
        else:
            # remove thousands separators like ',' or '.' when they appear in >1 places
            if v.count(',') > 1 and '.' not in v:
                v = v.replace(',', '')
            elif v.count('.') > 1 and ',' not in v:
                v = v.replace('.', '')
            # replace comma decimal '123,45' -> '123.45'
            if v.count(',') == 1 and v.count('.') == 0:
                v = v.replace(',', '.')
        return v

    fixed = s.map(fix_val)
    numeric = pd.to_numeric(fixed, errors='coerce')
    return numeric


def _detect_value_column(df: pd.DataFrame):
    """Detect the best numeric/value column in a DataFrame using heuristics.

    Priority:
      1. column name matches value_keywords
      2. first numeric dtype column
      3. column with most parsable numeric entries using _parse_numeric_series
    Returns column name or None.
    """
    # Prefer explicit price/value columns by name (strong signal)
    value_keywords = ['price', 'price_euro', 'price_euros', 'harga', 'value', 'y', 'amount', 'qty', 'sales', 'penjualan']
    for c in df.columns:
        cn = c.lower()
        for k in value_keywords:
            if k in cn:
                return c

    # Next: try to find numeric columns that are not ID-like (avoid choosing primary keys)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    id_like = lambda name: any(x in name.lower() for x in ['id', 'index', 'no.']) or name.lower().endswith('_id')
    non_id_numeric = [c for c in numeric_cols if not id_like(c)]
    if len(non_id_numeric) > 0:
        return non_id_numeric[0]

    # Finally, parse all columns and pick the one with the most parsable numeric entries
    best_col = None
    best_count = -1
    for c in df.columns:
        try:
            parsed = _parse_numeric_series(df[c])
            cnt = int(parsed.notna().sum())
            if cnt > best_count:
                best_count = cnt
                best_col = c
        except Exception:
            continue

    # accept only if parsed non-nulls are a reasonable fraction of rows
    if best_col is not None and best_count >= max(2, int(0.05 * len(df))):
        # avoid choosing id-like columns even here
        if not id_like(best_col):
            return best_col
    return None


async def forecast_multi_series(df: pd.DataFrame, periods: int = 30, series_col: str | None = None) -> Dict:
    """Forecast multiple series in one DataFrame. If `series_col` is None the function will try to detect
    a categorical/grouping column automatically. Returns a dict mapping series_key -> forecast_result (same shape as forecast_timeseries output).
    """
    in_df = df.copy()
    # try to find series_col if not provided: common names
    if series_col is None:
        for c in in_df.columns:
            if c.lower() in ('series', 'id', 'category', 'model', 'type', 'brand', 'name'):
                series_col = c
                break
        # if still none, try to find a non-date non-numeric column
        if series_col is None:
            for c in in_df.columns:
                if c not in ('ds', 'y') and in_df[c].dtype == object:
                    series_col = c
                    break

    results = {}
    if series_col is None:
        # treat as single series
        res = await forecast_timeseries(in_df, periods=periods)
        results['__single__'] = res
        return results

    # group and forecast each series
    grouped = in_df.groupby(series_col)
    for key, grp in grouped:
        try:
            std = prepare_df(grp)
            res = await forecast_timeseries(std, periods=periods)
            results[str(key)] = res
        except Exception as exc:
            results[str(key)] = {'error': str(exc)}
    return results



def train_prophet_sync(df, periods: int = 30):
    """Synchronous wrapper for Prophet/baseline training."""
    import asyncio
    # use asyncio.run to avoid deprecated get_event_loop patterns
    return asyncio.run(_train_prophet_or_baseline(df, periods=periods))



def train_arima_sync(df, periods: int = 30):
    import asyncio
    return asyncio.run(_train_arima_or_baseline(df, periods=periods))



def train_exp_smoothing_sync(df, periods: int = 30):
    import asyncio
    return asyncio.run(_train_exp_smoothing_or_baseline(df, periods=periods))


def eval_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = np.where(np.abs(y_true) < 1e-8, 1e-8, y_true)
        mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)
    return {'mae': mae, 'rmse': rmse, 'mape': mape}


def select_best_model(candidates: list):
    """Select best candidate (name, forecast_df, metrics) by RMSE."""
    best = None
    for name, fcst, metrics in candidates:
        if best is None or metrics['rmse'] < best[2]['rmse']:
            best = (name, fcst, metrics)
    return best