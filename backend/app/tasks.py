import json
from io import BytesIO
import pandas as pd
import asyncio
from typing import Dict, Any
from app.core.forecasting import forecast_timeseries
from app.db import update_job_result, set_job_running


def forecast_job(job_id: int, file_bytes: bytes, periods: int = 30) -> Dict[str, Any]:
    """Job function executed by RQ worker or manual invoker.

    It runs forecasting synchronously and updates the SQLite job record.
    Returns the result dict (also stored in DB).
    """
    try:
        set_job_running(job_id)
        df = pd.read_csv(BytesIO(file_bytes))
        # run async forecasting in a blocking way
        result = asyncio.run(forecast_timeseries(df, periods=periods))
        update_job_result(job_id, result=result)
        return result
    except Exception as exc:
        update_job_result(job_id, error=str(exc))
        raise
