from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from .models import (
    HealthResponse, CompositionResponse,
    TrendResponse, ForecastResponse,
    InsightResponse, QueryRequest, QueryResponse
)
from .database import query_db

app = FastAPI(
    title='Industrial Waste Intelligence API',
    description='End-to-end AI platform for industrial waste composition analysis',
    version='1.0.0'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*']
)

# Health Check
@app.get('/', response_model=HealthResponse)
async def root():
    """Root health check — verify API and database connection."""
    try:
        query_db('SELECT 1')
        db_status = 'connected'
    except Exception:
        db_status = 'disconnected'

    return HealthResponse(
        status  = 'ok',
        version = '1.0.0',
        model   = 'Mask R-CNN ResNet-50 FPN (mAP@0.5: 48.95%)',
        database= db_status
    )

@app.get('/health')
async def health():
    """Lightweight health check for monitoring."""
    return {
        'status'   : 'ok',
        'timestamp': datetime.now().isoformat()
    }
