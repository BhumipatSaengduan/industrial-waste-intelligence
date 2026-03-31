from pydantic import BaseModel
from typing import Optional

class HealthResponse(BaseModel):
    status: str
    version: str
    model: str
    database: str

class CompositionResponse(BaseModel):
    filename: str
    instances: int
    composition: dict
    total_area_detected: float

class TrendResponse(BaseModel):
    date: str
    class_name: str
    annotation_count: int
    mean_bbox_area: float
    image_count: int

class ForecastResponse(BaseModel):
    forecast_date: str
    class_name: str
    predicted_count: float
    model_name: str

class InsightResponse(BaseModel):
    insight_date: str
    insight_type: str
    content: str

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list
