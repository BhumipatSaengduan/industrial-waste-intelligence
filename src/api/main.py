from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import os
import tempfile
import base64
import cv2

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

# Load model at startup
MODEL_PATH = os.getenv('MODEL_PATH', 'models/final/waste_maskrcnn_torchvision.pth')

try:
    from src.vision.inference import load_predictor, run_inference
    if os.path.exists(MODEL_PATH):
        model_tuple = load_predictor(MODEL_PATH, score_thresh=0.3)
        MODEL_READY = True
        print(f'Model loaded from {MODEL_PATH}')
    else:
        MODEL_READY = False
        print(f'Model not found at {MODEL_PATH} — /analyze will return 503')
except Exception as e:
    MODEL_READY = False
    print(f'Model load failed: {e}')

# Health Check
@app.get('/', response_model=HealthResponse)
async def root():
    try:
        query_db('SELECT 1')
        db_status = 'connected'
    except Exception:
        db_status = 'disconnected'

    return HealthResponse(
        status  = 'ok',
        version = '1.0.0',
        model   = f'Mask R-CNN ResNet-50 FPN (torchvision) — ready: {MODEL_READY}',
        database= db_status
    )

@app.get('/health')
async def health():
    return {
        'status'     : 'ok',
        'model_ready': MODEL_READY,
        'timestamp'  : datetime.now().isoformat()
    }

# POST /analyze
@app.post('/analyze')
async def analyze(file: UploadFile = File(...)):
    """
    Upload waste image and return segmentation results.
    composition sums to 100% of detected waste area.
    annotated_image is base64-encoded JPG with colored masks.
    """
    if not MODEL_READY:
        raise HTTPException(
            status_code=503,
            detail='Model not loaded — check MODEL_PATH in .env'
        )

    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail='Only JPG and PNG supported')

    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    instances, composition, annotated_img = run_inference(model_tuple, tmp_path)
    os.unlink(tmp_path)

    _, buffer  = cv2.imencode('.jpg', cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        'filename'           : file.filename,
        'instances'          : instances['count'],
        'composition'        : composition,
        'total_area_detected': round(sum(composition.values()), 2),
        'annotated_image'    : img_base64
    }

# GET /trend
@app.get('/trend')
async def get_trend(class_name: str = None, limit: int = 30):
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
