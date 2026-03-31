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

# GET /trend
@app.get('/trend')
async def get_trend(class_name: str = None, limit: int = 30):
    """
    Get daily waste composition trends from Cloud SQL.
    Optional filter by class_name.
    """
    if class_name:
        rows = query_db("""
            SELECT date, class_name, annotation_count,
                   mean_bbox_area, image_count
            FROM daily_waste_summary
            WHERE class_name = %s
            ORDER BY date DESC
            LIMIT %s
        """, (class_name, limit))
    else:
        rows = query_db("""
            SELECT date, class_name, annotation_count,
                   mean_bbox_area, image_count
            FROM daily_waste_summary
            ORDER BY date DESC, class_name
            LIMIT %s
        """, (limit,))

    for row in rows:
        row['date']           = str(row['date'])
        row['mean_bbox_area'] = float(row['mean_bbox_area']) if row['mean_bbox_area'] else 0.0

    return {'count': len(rows), 'data': rows}

# GET /forecast
@app.get('/forecast')
async def get_forecast(class_name: str = None):
    if class_name:
        rows = query_db("""
            SELECT forecast_date, class_name, predicted_count, model_name
            FROM forecast_results
            WHERE class_name = %s
            ORDER BY forecast_date, class_name
        """, (class_name,))
    else:
        rows = query_db("""
            SELECT forecast_date, class_name, predicted_count, model_name
            FROM forecast_results
            ORDER BY forecast_date, class_name
        """)

    if not rows:
        return {'count': 0, 'data': []}

    for row in rows:
        row['forecast_date']   = str(row['forecast_date'])
        row['predicted_count'] = float(row['predicted_count'])

    return {'count': len(rows), 'data': rows}

# GET /anomaly
@app.get('/anomaly')
async def get_anomaly():
    """
    Get anomaly detection results from Cloud SQL.
    Returns all flagged anomalous days with explanations.
    """
    rows = query_db("""
        SELECT date, is_anomaly, anomaly_score,
               annotation_count, explanation
        FROM anomaly_results
        WHERE is_anomaly = TRUE
        ORDER BY anomaly_score
    """)

    for row in rows:
        row['date']          = str(row['date'])
        row['anomaly_score'] = float(row['anomaly_score'])

    return {'count': len(rows), 'data': rows}

# GET /insights
@app.get('/insights')
async def get_insights(insight_type: str = None):
    """
    Get LLM-generated insights from Cloud SQL.
    Optional filter by insight_type: weekly_report or anomaly_explanation.
    """
    if insight_type:
        rows = query_db("""
            SELECT insight_date, insight_type, content
            FROM llm_insights
            WHERE insight_type = %s
            ORDER BY insight_date DESC
        """, (insight_type,))
    else:
        rows = query_db("""
            SELECT insight_date, insight_type, content
            FROM llm_insights
            ORDER BY insight_date DESC
        """)

    for row in rows:
        row['insight_date'] = str(row['insight_date'])

    return {'count': len(rows), 'data': rows}

# POST /query
@app.post('/query')
async def query_rag(request: QueryRequest):
    """
    Natural language query interface using RAG pipeline.
    Retrieves relevant documents from ChromaDB and generates answer via Ollama.
    """
    try:
        from src.llm.rag import get_collection, rag_query
        collection = get_collection()
        result     = rag_query(request.question, collection)
        return QueryResponse(
            question = result['question'],
            answer   = result['answer'],
            sources  = result['sources']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
