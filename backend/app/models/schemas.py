from pydantic import BaseModel
from typing import List, Dict, Optional

class TimeSeriesData(BaseModel):
    timestamp: List[str]
    value: List[float]

class ModelMetrics(BaseModel):
    mae: float
    rmse: float
    mape: Optional[float]

class ForecastResponse(BaseModel):
    historical_data: TimeSeriesData
    forecast_data: TimeSeriesData
    best_model: str
    metrics: ModelMetrics
    decomposition: Dict[str, List[float]]