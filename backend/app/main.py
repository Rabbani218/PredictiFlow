from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from io import BytesIO
from typing import Any, Dict, Optional
import json

from app.db import create_job, set_job_running, update_job_result, get_job, list_jobs
try:
    # optional: prefer redis/rq when available
    from redis import Redis as _RedisClient
    from rq import Queue as _RQQueue
    _RQ_AVAILABLE = True
except Exception:
    _RQ_AVAILABLE = False

import os
REDIS_URL = os.environ.get('REDIS_URL')
import logging

from app.core.forecasting import forecast_timeseries
from app.models.schemas import ForecastResponse
from app.registry import register_model, list_models as registry_list, get_model, get_model_file_path
from fastapi.responses import FileResponse

# basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PredictiFlow API",
    description="Time Series Forecasting API",
    version="1.0.0"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to PredictiFlow API"}

@app.post("/predict", response_model=ForecastResponse)
async def predict(file: UploadFile = File(...), periods: int = 30):
    """Accept an uploaded CSV and return forecasting results.

    Query params:
      - periods: forecast horizon in periods (int)
    """
    # basic validation
    if periods <= 0 or periods > 365:
        raise HTTPException(status_code=400, detail="periods must be between 1 and 365")

    try:
        # Read raw bytes and parse with pandas to avoid file-pointer issues
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))
    except Exception as exc:
        logger.exception("Failed to read uploaded CSV")
        raise HTTPException(status_code=400, detail=f"Invalid CSV upload: {exc}")

    try:
        # Perform forecasting
        forecast_result = await forecast_timeseries(df, periods=periods)
    except Exception as exc:
        logger.exception("Forecasting failed")
        raise HTTPException(status_code=500, detail=f"Forecasting error: {exc}")

    return forecast_result

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


def _run_forecast_and_persist(job_id: int, file_bytes: bytes, periods: int):
    """Background worker function: runs forecasting and updates the job record."""
    try:
        set_job_running(job_id)
        df = pd.read_csv(BytesIO(file_bytes))
        # run forecasting synchronously in background thread
        import asyncio
        result = asyncio.run(forecast_timeseries(df, periods=periods))
        update_job_result(job_id, result=result)
    except Exception as exc:
        update_job_result(job_id, error=str(exc))


@app.post("/predict-async")
async def predict_async(background_tasks: BackgroundTasks, file: UploadFile = File(...), periods: int = 30) -> Dict[str, Any]:
    # validate
    if periods <= 0 or periods > 365:
        raise HTTPException(status_code=400, detail="periods must be between 1 and 365")

    contents = await file.read()
    payload = {"filename": file.filename, "periods": periods}
    job_id = create_job(payload)
    # If RQ is available and REDIS_URL configured, enqueue a background job
    if _RQ_AVAILABLE and REDIS_URL:
        try:
            redis_conn = _RedisClient.from_url(REDIS_URL)
            q = _RQQueue('default', connection=redis_conn)
            # enqueue the task (the callable path is app.tasks.forecast_job)
            q.enqueue('app.tasks.forecast_job', job_id, contents, periods)
            return {"job_id": job_id, "queued": True}
        except Exception:
            # fallback to background task
            background_tasks.add_task(_run_forecast_and_persist, job_id, contents, periods)
            return {"job_id": job_id, "queued": False}
    else:
        # schedule background task (runs in same process in threadpool)
        background_tasks.add_task(_run_forecast_and_persist, job_id, contents, periods)
        return {"job_id": job_id, "queued": False}


@app.get("/jobs")
async def api_list_jobs(limit: int = 50):
    return list_jobs(limit=limit)


@app.get("/jobs/{job_id}")
async def api_get_job(job_id: int):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.get("/models")
async def api_list_models(limit: int = 50):
    """Return registered models metadata."""
    return registry_list(limit=limit)


@app.post("/models")
async def api_upload_model(file: UploadFile = File(...), name: Optional[str] = None):
    """Upload a model artifact and register it in the registry."""
    contents = await file.read()
    model_name = name or file.filename
    try:
        model_id = register_model(model_name, file.filename, contents, metadata={})
    except Exception as exc:
        logger.exception("Failed to register model")
        raise HTTPException(status_code=500, detail=str(exc))
    return {"id": model_id, "name": model_name}


@app.get("/models/{model_id}/download")
async def api_download_model(model_id: int):
    m = get_model(model_id)
    if m is None:
        raise HTTPException(status_code=404, detail="model not found")
    p = get_model_file_path(m['filename'])
    if p is None:
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(path=str(p), filename=m['filename'], media_type='application/octet-stream')