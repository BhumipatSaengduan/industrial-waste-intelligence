from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import os
import tempfile
import base64
import cv2
import threading
import time

from .models import (
    HealthResponse, CompositionResponse,
    TrendResponse, ForecastResponse,
    InsightResponse, QueryRequest, QueryResponse
)
from .database import query_db, execute_db

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

# ---------------------------------------------------------------------------
# Background scheduler — auto-generate AI Insights reports
# Daily report  : every day at 18:00
# Weekly report : every Monday at 18:00
# ---------------------------------------------------------------------------
def _run_insights_job(report_type: str):
    """Call the insights generation logic directly (no HTTP round-trip)."""
    import requests
    try:
        requests.post(
            f'http://localhost:8000/insights/generate',
            params={'report_type': report_type, 'force': False},
            timeout=120
        )
        print(f'[scheduler] {report_type} generated at {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    except Exception as e:
        print(f'[scheduler] {report_type} failed: {e}')


def _scheduler_loop():
    triggered_daily  = None   # tracks date string of last daily trigger
    triggered_weekly = None   # tracks ISO week string of last weekly trigger

    while True:
        now = datetime.now()
        today_str = now.strftime('%Y-%m-%d')
        week_str  = now.strftime('%Y-W%W')   # e.g. "2026-W14"

        # Daily report — every day at 18:00
        if now.hour == 18 and now.minute == 0 and triggered_daily != today_str:
            triggered_daily = today_str
            threading.Thread(target=_run_insights_job, args=('daily_report',), daemon=True).start()

        # Weekly report — Monday at 00:00 (covers previous Mon–Sun)
        if now.weekday() == 0 and now.hour == 0 and now.minute == 0 and triggered_weekly != week_str:
            triggered_weekly = week_str
            threading.Thread(target=_run_insights_job, args=('weekly_report',), daemon=True).start()

        time.sleep(60)   # check every minute


@app.on_event('startup')
def start_scheduler():
    def _catchup():
        time.sleep(5)  # wait for DB connection to be ready
        from datetime import timedelta
        now = datetime.now()  # uses TZ=Asia/Tokyo from docker-compose → JST

        # Daily catch-up: check last 7 days for any missed reports
        for days_ago in range(7):
            check_date = (now - timedelta(days=days_ago)).date()
            # Skip today if it's before 18:00 (report not due yet)
            if days_ago == 0 and now.hour < 18:
                continue
            existing = query_db(
                "SELECT id FROM llm_insights WHERE insight_type='daily_report' AND insight_date=%s LIMIT 1",
                (str(check_date),)
            )
            if not existing:
                print(f'[scheduler] Catch-up: generating missed daily report for {check_date}')
                _run_insights_job('daily_report')
                break  # generate only the most recent missed day

        # Weekly catch-up: check current week regardless of what day it is
        days_since_monday = now.weekday()
        monday_this_week = (now - timedelta(days=days_since_monday)).date()
        existing = query_db(
            "SELECT id FROM llm_insights WHERE insight_type='weekly_report' AND insight_date=%s LIMIT 1",
            (str(monday_this_week),)
        )
        if not existing:
            print(f'[scheduler] Catch-up: generating missed weekly report for week of {monday_this_week}')
            _run_insights_job('weekly_report')

    threading.Thread(target=_catchup, daemon=True).start()

    t = threading.Thread(target=_scheduler_loop, daemon=True)
    t.start()
    print('[scheduler] Background insight scheduler started')

# Ensure analysis_records table exists
try:
    execute_db("""
        CREATE TABLE IF NOT EXISTS analysis_records (
            id                  SERIAL PRIMARY KEY,
            filename            TEXT NOT NULL,
            analysis_date       DATE NOT NULL DEFAULT CURRENT_DATE,
            analyzed_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metal_pct           FLOAT NOT NULL DEFAULT 0,
            mixed_waste_pct     FLOAT NOT NULL DEFAULT 0,
            paper_cardboard_pct FLOAT NOT NULL DEFAULT 0,
            plastic_pct         FLOAT NOT NULL DEFAULT 0,
            wood_pct            FLOAT NOT NULL DEFAULT 0,
            total_area_detected FLOAT NOT NULL DEFAULT 0
        )
    """)
    execute_db(
        "CREATE INDEX IF NOT EXISTS idx_records_date ON analysis_records(analysis_date)"
    )
except Exception as e:
    print(f'analysis_records table setup failed: {e}')

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
async def get_trend(
    class_name: str = None,
    date_from:  str = None,
    date_to:    str = None,
    limit:      int = 30,
):
    # Build WHERE clauses
    conditions = []
    params     = []
    if class_name:
        conditions.append('class_name = %s')
        params.append(class_name)
    if date_from:
        conditions.append('date >= %s')
        params.append(date_from)
    if date_to:
        conditions.append('date <= %s')
        params.append(date_to)

    where = ('WHERE ' + ' AND '.join(conditions)) if conditions else ''

    # When date range given, return all rows in range; otherwise use limit
    if date_from or date_to:
        sql = f"""
            SELECT date, class_name, annotation_count,
                   mean_bbox_area, image_count
            FROM daily_waste_summary
            {where}
            ORDER BY date ASC, class_name
        """
        rows = query_db(sql, params if params else None)
    else:
        params.append(limit)
        sql = f"""
            SELECT date, class_name, annotation_count,
                   mean_bbox_area, image_count
            FROM daily_waste_summary
            {where}
            ORDER BY date DESC, class_name
            LIMIT %s
        """
        rows = query_db(sql, params if params else None)

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

# Mapping: waste class name → analysis_records column
_CLASS_COL = {
    'Metal':           'metal_pct',
    'Mixed Waste':     'mixed_waste_pct',
    'Paper-Cardboard': 'paper_cardboard_pct',
    'Plastic':         'plastic_pct',
    'Wood':            'wood_pct',
}


def _sync_daily_summary(date_str: str):
    """
    Re-aggregate daily_waste_summary for one date from analysis_records.
    If no records remain for that date, delete those summary rows
    (they were dashboard-created, not pipeline-created).
    """
    remaining = query_db("""
        SELECT metal_pct, mixed_waste_pct, paper_cardboard_pct,
               plastic_pct, wood_pct
        FROM analysis_records
        WHERE analysis_date = %s
    """, (date_str,))

    if not remaining:
        # No dashboard records left for this date — remove dashboard rows
        execute_db(
            "DELETE FROM daily_waste_summary WHERE date = %s", (date_str,)
        )
        return

    img_count = len(remaining)
    for class_name, col in _CLASS_COL.items():
        values     = [(r[col] or 0) for r in remaining]
        total_area = sum(values)               # sum of % across all images
        mean_area  = total_area / img_count if img_count else 0
        # Store total_area as annotation_count so compute_trend_pct
        # normalises correctly: each class gets its proportional share
        execute_db("""
            INSERT INTO daily_waste_summary
                (date, class_name, annotation_count, mean_bbox_area,
                 total_bbox_area, image_count)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (date, class_name) DO UPDATE SET
                annotation_count = EXCLUDED.annotation_count,
                mean_bbox_area   = EXCLUDED.mean_bbox_area,
                total_bbox_area  = EXCLUDED.total_bbox_area,
                image_count      = EXCLUDED.image_count
        """, (date_str, class_name, total_area, mean_area, total_area, img_count))


# POST /records/save
@app.post('/records/save')
async def save_record(payload: dict):
    """Save a dashboard analysis result and sync daily_waste_summary."""
    comp = payload.get('composition', {})
    from datetime import date as _date
    # Use client-provided local date (avoids UTC vs local timezone mismatch)
    analysis_date = payload.get('analysis_date') or str(_date.today())
    try:
        execute_db("""
            INSERT INTO analysis_records
                (filename, analysis_date, metal_pct, mixed_waste_pct,
                 paper_cardboard_pct, plastic_pct, wood_pct, total_area_detected)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            payload.get('filename', 'unknown'),
            analysis_date,
            comp.get('Metal', 0),
            comp.get('Mixed Waste', 0),
            comp.get('Paper-Cardboard', 0),
            comp.get('Plastic', 0),
            comp.get('Wood', 0),
            payload.get('total_area_detected', 0),
        ))
        _sync_daily_summary(analysis_date)
        return {'status': 'saved', 'daily_summary_updated': True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# DELETE /records/last  — undo the most recent save
@app.delete('/records/last')
async def delete_last_record():
    """Delete the most recently saved record and re-sync daily_waste_summary."""
    rows = query_db("""
        SELECT id, analysis_date FROM analysis_records
        ORDER BY analyzed_at DESC LIMIT 1
    """)
    if not rows:
        raise HTTPException(status_code=404, detail='No records to delete')
    record_id   = rows[0]['id']
    record_date = str(rows[0]['analysis_date'])
    execute_db("DELETE FROM analysis_records WHERE id = %s", (record_id,))
    _sync_daily_summary(record_date)
    return {'status': 'deleted', 'id': record_id, 'date': record_date}


# DELETE /records/all  — clear every dashboard-uploaded record
@app.delete('/records/all')
async def delete_all_records():
    """Delete all analysis_records and clean up daily_waste_summary."""
    # Dates present in analysis_records before deletion
    date_rows = query_db(
        "SELECT DISTINCT analysis_date FROM analysis_records"
    )
    dates_from_records = {str(r['analysis_date']) for r in date_rows}

    # Also find any current-year dates in daily_waste_summary that are
    # dashboard-originated (pipeline data is historical, not current-year)
    current_year_rows = query_db("""
        SELECT DISTINCT date FROM daily_waste_summary
        WHERE EXTRACT(YEAR FROM date) = EXTRACT(YEAR FROM CURRENT_DATE)
    """)
    dates_current_year = {str(r['date']) for r in current_year_rows}

    all_dates = dates_from_records | dates_current_year
    execute_db("DELETE FROM analysis_records")
    for d in all_dates:
        _sync_daily_summary(d)
    return {'status': 'deleted_all', 'dates_cleaned': len(all_dates)}


# DELETE /insights/one — delete a specific insight by date + type
@app.delete('/insights/one')
async def delete_one_insight(insight_date: str, insight_type: str):
    rows = query_db(
        "SELECT id FROM llm_insights WHERE insight_date = %s AND insight_type = %s LIMIT 1",
        (insight_date, insight_type)
    )
    if not rows:
        raise HTTPException(status_code=404, detail='Insight not found')
    execute_db(
        "DELETE FROM llm_insights WHERE insight_date = %s AND insight_type = %s",
        (insight_date, insight_type)
    )
    return {'status': 'deleted', 'date': insight_date, 'type': insight_type}


# DELETE /insights/all — clear all insight reports
@app.delete('/insights/all')
async def delete_all_insights():
    rows = query_db("SELECT COUNT(*) as n FROM llm_insights")
    count = rows[0]['n'] if rows else 0
    execute_db("DELETE FROM llm_insights")
    return {'status': 'deleted_all', 'count': count}


# GET /records
@app.get('/records')
async def get_records(
    date_from: str = None,
    date_to:   str = None,
    limit:     int = 500,
):
    """Return per-image analysis records with optional date range filter and limit."""
    _COLS = """
        SELECT id, filename, analysis_date, analyzed_at,
               metal_pct, mixed_waste_pct, paper_cardboard_pct,
               plastic_pct, wood_pct, total_area_detected
        FROM analysis_records
    """
    if date_from and date_to:
        rows = query_db(
            _COLS + "WHERE analysis_date BETWEEN %s AND %s ORDER BY analyzed_at DESC LIMIT %s",
            (date_from, date_to, limit),
        )
    elif date_from:
        rows = query_db(
            _COLS + "WHERE analysis_date >= %s ORDER BY analyzed_at DESC LIMIT %s",
            (date_from, limit),
        )
    elif date_to:
        rows = query_db(
            _COLS + "WHERE analysis_date <= %s ORDER BY analyzed_at DESC LIMIT %s",
            (date_to, limit),
        )
    else:
        rows = query_db(
            _COLS + "ORDER BY analyzed_at DESC LIMIT %s",
            (limit,),
        )

    for row in rows:
        row['analysis_date'] = str(row['analysis_date'])
        row['analyzed_at']   = str(row['analyzed_at'])

    return {'count': len(rows), 'data': rows}


# ── Helper: live DB context string for LLM enrichment ────────────────────────
def _get_live_context() -> str:
    """Build a real-time context string from DB for enriching LLM queries."""
    import pandas as pd
    from datetime import date, timedelta
    from collections import defaultdict
    today    = str(date.today())
    week_ago = str(date.today() - timedelta(days=7))

    lines = [f"=== LIVE DATA (as of {today}) ==="]

    # Today's per-image records
    today_recs = query_db("""
        SELECT metal_pct, mixed_waste_pct, paper_cardboard_pct, plastic_pct, wood_pct
        FROM analysis_records WHERE analysis_date = %s
    """, (today,))

    if today_recs:
        df = pd.DataFrame(today_recs)
        avgs = {
            'Metal':           float(df['metal_pct'].mean()),
            'Mixed Waste':     float(df['mixed_waste_pct'].mean()),
            'Paper-Cardboard': float(df['paper_cardboard_pct'].mean()),
            'Plastic':         float(df['plastic_pct'].mean()),
            'Wood':            float(df['wood_pct'].mean()),
        }
        total = sum(avgs.values())
        pcts = {k: round(v / total * 100, 1) if total > 0 else 0 for k, v in avgs.items()}
        lines.append(f"Today ({today}) — {len(today_recs)} image(s) analyzed:")
        for cls, pct in sorted(pcts.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  {cls}: {pct}%")
    else:
        lines.append(f"Today ({today}): No images analyzed yet.")

    # Last 7 days daily summaries
    recent = query_db("""
        SELECT date, class_name, annotation_count
        FROM daily_waste_summary WHERE date >= %s ORDER BY date DESC
    """, (week_ago,))

    if recent:
        by_date = defaultdict(dict)
        for r in recent:
            by_date[str(r['date'])][r['class_name']] = float(r['annotation_count'])
        lines.append("\nRecent daily data (last 7 days):")
        for d in sorted(by_date.keys(), reverse=True)[:5]:
            day = by_date[d]
            t   = sum(day.values())
            if t > 0:
                top = max(day, key=day.get)
                lines.append(f"  {d}: dominant = {top} ({round(day[top]/t*100,1)}%)")

    # Recent anomalies
    anomalies = query_db("""
        SELECT date, anomaly_score, explanation
        FROM anomaly_results
        WHERE is_anomaly = TRUE AND date >= %s ORDER BY date DESC LIMIT 3
    """, (week_ago,))

    if anomalies:
        lines.append("\nRecent anomalies:")
        for a in anomalies:
            lines.append(f"  {a['date']}: score={float(a['anomaly_score']):.3f} — {a.get('explanation','')}")
    else:
        lines.append("\nNo anomalies in the last 7 days.")

    return '\n'.join(lines)


# POST /forecast/regenerate
@app.post('/forecast/regenerate')
async def regenerate_forecast():
    """Generate fresh 30-day forecasts via trend extrapolation from daily_waste_summary."""
    import numpy as np
    import pandas as pd
    from datetime import date, timedelta

    FORECAST_CLASSES_LOCAL = ['Plastic', 'Paper-Cardboard', 'Mixed Waste', 'Wood', 'Metal']

    rows = query_db("""
        SELECT date, class_name, annotation_count
        FROM daily_waste_summary ORDER BY date ASC
    """)
    if not rows:
        raise HTTPException(status_code=404, detail='No historical data in daily_waste_summary')

    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    today          = date.today()
    forecast_dates = [today + timedelta(days=i) for i in range(1, 31)]

    # Remove only dashboard-generated forecasts (preserve original ML pipeline rows)
    execute_db("DELETE FROM forecast_results WHERE model_name = 'dashboard_trend'")

    total_created = 0
    for cls in FORECAST_CLASSES_LOCAL:
        cls_df = df[df['class_name'] == cls].copy().sort_values('date')
        if cls_df.empty:
            continue
        counts = cls_df['annotation_count'].values.astype(float)
        n      = len(counts)

        # Baseline: rolling mean of last 30 data points
        baseline = float(np.mean(counts[-min(30, n):]))
        floor    = baseline * 0.3   # never predict below 30% of baseline

        # Linear trend from last 14 points (damped)
        tail = min(14, n)
        slope = float(np.polyfit(np.arange(tail), counts[-tail:], 1)[0]) if tail >= 2 else 0.0

        for i, fd in enumerate(forecast_dates, 1):
            predicted = max(floor, baseline + slope * i * 0.5)
            execute_db("""
                INSERT INTO forecast_results
                    (forecast_date, class_name, predicted_count, model_name)
                VALUES (%s, %s, %s, 'dashboard_trend')
                ON CONFLICT (forecast_date, class_name, model_name) DO UPDATE
                    SET predicted_count = EXCLUDED.predicted_count
            """, (str(fd), cls, float(predicted)))
            total_created += 1

    return {'status': 'regenerated', 'forecasts_created': total_created, 'days_ahead': 30}


# POST /insights/generate
@app.post('/insights/generate')
async def generate_insights(report_type: str = 'daily_report', force: bool = False):
    """Generate a daily or weekly insight report and save to llm_insights."""
    import pandas as pd
    from datetime import date, timedelta
    today     = date.today()
    today_str = str(today)

    # Avoid duplicates unless force=True
    if not force:
        existing = query_db("""
            SELECT id FROM llm_insights
            WHERE insight_type = %s AND insight_date = %s LIMIT 1
        """, (report_type, today_str))
        if existing:
            return {'status': 'already_exists', 'type': report_type, 'date': today_str}

    period_start = str(today - timedelta(days=7)) if report_type == 'weekly_report' else today_str
    period_end   = today_str
    period_label = (
        f"Daily Report — {today_str}"
        if report_type == 'daily_report'
        else f"Weekly Report — {period_start} to {period_end}"
    )

    # Fetch data for the period
    records = query_db("""
        SELECT metal_pct, mixed_waste_pct, paper_cardboard_pct, plastic_pct, wood_pct
        FROM analysis_records WHERE analysis_date BETWEEN %s AND %s
    """, (period_start, period_end))

    summary_rows = query_db("""
        SELECT date, class_name, annotation_count, image_count
        FROM daily_waste_summary WHERE date BETWEEN %s AND %s
    """, (period_start, period_end))

    anomalies = query_db("""
        SELECT date, anomaly_score, explanation FROM anomaly_results
        WHERE is_anomaly = TRUE AND date BETWEEN %s AND %s ORDER BY date DESC
    """, (period_start, period_end))

    # Build context string
    if records:
        df  = pd.DataFrame(records)
        avgs = {
            'Metal':           float(df['metal_pct'].mean()),
            'Mixed Waste':     float(df['mixed_waste_pct'].mean()),
            'Paper-Cardboard': float(df['paper_cardboard_pct'].mean()),
            'Plastic':         float(df['plastic_pct'].mean()),
            'Wood':            float(df['wood_pct'].mean()),
        }
        tot  = sum(avgs.values())
        pcts = {k: round(v / tot * 100, 1) if tot > 0 else 0 for k, v in avgs.items()}
        dominant  = max(pcts, key=pcts.get)
        comp_text = '\n'.join(f'  {k}: {v}%' for k, v in sorted(pcts.items(), key=lambda x: x[1], reverse=True))
        data_context = (
            f"{period_label}\nImages analyzed: {len(records)}\n"
            f"Waste composition:\n{comp_text}\nDominant: {dominant} ({pcts[dominant]}%)"
        )
    elif summary_rows:
        df_s      = pd.DataFrame(summary_rows)
        by_cls    = df_s.groupby('class_name')['annotation_count'].sum()
        grand     = float(by_cls.sum())
        pct_s     = {k: round(float(v) / grand * 100, 1) for k, v in by_cls.items()} if grand > 0 else {}
        dominant  = max(pct_s, key=pct_s.get) if pct_s else 'Unknown'
        comp_text = '\n'.join(f'  {k}: {v}%' for k, v in sorted(pct_s.items(), key=lambda x: x[1], reverse=True))
        data_context = (
            f"{period_label}\nNo direct uploads; using pipeline summary data.\n"
            f"Waste composition:\n{comp_text}\nDominant: {dominant}"
        )
        pcts = pct_s
    else:
        data_context = f"{period_label}\nNo data available for this period."
        pcts = {}

    if anomalies:
        anom_lines = '\n'.join(f"  {a['date']}: {a.get('explanation','anomaly detected')}" for a in anomalies)
        data_context += f"\nAnomalies ({len(anomalies)}):\n{anom_lines}"
    else:
        data_context += "\nNo anomalies detected."

    # Try Ollama LLM first, fall back to template
    content = None
    try:
        from src.llm.insights import ollama_generate
        if report_type == 'daily_report':
            from src.llm.prompts import DAILY_REPORT_PROMPT
            prompt = DAILY_REPORT_PROMPT.format(context=data_context)
        else:
            from src.llm.prompts import WEEKLY_REPORT_PROMPT
            prompt = WEEKLY_REPORT_PROMPT.format(context=data_context)
        content = ollama_generate(prompt, max_tokens=300)
    except Exception:
        pass  # LLM unavailable — use template

    if not content:
        if pcts:
            ranked    = sorted(pcts.items(), key=lambda x: x[1], reverse=True)
            comp_str  = ', '.join(f'{k} {v}%' for k, v in ranked[:3])
            anom_note = (
                f"⚠️ {len(anomalies)} anomalous event(s) detected in this period."
                if anomalies else "No anomalies detected."
            )
            action = (
                f"Monitor {ranked[0][0]} levels closely — highest composition at {ranked[0][1]}%."
                if ranked[0][1] > 40
                else "Waste composition is balanced — continue standard operations."
            )
            content = f"TREND: Dominant waste type is {ranked[0][0]} ({ranked[0][1]}%) — composition: {comp_str}.\nANOMALY: {anom_note}\nACTION: {action}"
        else:
            period_word = 'today' if report_type == 'daily_report' else 'this week'
            content = (
                f"TREND: No images uploaded {period_word}.\n"
                f"ANOMALY: No data to evaluate.\n"
                f"ACTION: Upload waste facility images on the Image Analysis page to generate insights."
            )

    execute_db("""
        INSERT INTO llm_insights (insight_date, insight_type, content)
        VALUES (%s, %s, %s)
    """, (today_str, report_type, content))

    return {'status': 'generated', 'type': report_type, 'date': today_str}


# POST /anomaly/check
@app.post('/anomaly/check')
async def check_anomalies():
    """Re-run Isolation Forest anomaly detection on all daily_waste_summary data."""
    rows = query_db("""
        SELECT date, class_name, annotation_count, image_count
        FROM daily_waste_summary ORDER BY date ASC
    """)
    if not rows:
        return {'status': 'no_data', 'checked': 0, 'anomalies_found': 0}

    try:
        import pandas as pd
        from src.ml.anomaly import prepare_anomaly_features, detect_anomalies, explain_anomaly
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f'ML module unavailable: {e}')

    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])

    if df['date'].nunique() < 5:
        return {
            'status': 'insufficient_data',
            'message': f'Need ≥5 days of data, have {df["date"].nunique()}',
            'checked': 0, 'anomalies_found': 0,
        }

    df_pivot  = prepare_anomaly_features(df)
    df_result = detect_anomalies(df_pivot, contamination=0.1)
    df_mean   = df_pivot[['Metal', 'Mixed Waste', 'Paper-Cardboard', 'Plastic', 'Wood', 'total_count']].mean()

    execute_db("DELETE FROM anomaly_results")

    anomalies_found = 0
    for _, row in df_result.iterrows():
        is_anom     = bool(row['is_anomaly'])
        explanation = explain_anomaly(row, df_mean) if is_anom else None
        if is_anom:
            anomalies_found += 1
        execute_db("""
            INSERT INTO anomaly_results
                (date, class_name, anomaly_score, is_anomaly, annotation_count, explanation)
            VALUES (%s, NULL, %s, %s, %s, %s)
        """, (str(row['date'].date()), float(row['anomaly_score']), is_anom,
              int(row['total_count']), explanation))

    return {
        'status'          : 'checked',
        'total_days'      : len(df_result),
        'anomalies_found' : anomalies_found,
    }


# POST /query
@app.post('/query')
async def query_rag(request: QueryRequest):
    live_context = _get_live_context()
    try:
        from src.llm.rag import get_collection
        from src.llm.insights import ollama_generate
        from src.llm.prompts import RAG_PROMPT

        collection     = get_collection()
        results        = collection.query(query_texts=[request.question], n_results=3)
        retrieved_docs = results['documents'][0]
        retrieved_meta = results['metadatas'][0]

        # Live DB context prepended to ChromaDB knowledge base
        context_parts = [live_context]
        for doc, meta in zip(retrieved_docs, retrieved_meta):
            context_parts.append(
                f"[{meta['type']} — {meta.get('date', meta.get('week_start', ''))}]\n{doc}"
            )
        context = '\n\n'.join(context_parts)
        answer  = ollama_generate(RAG_PROMPT.format(context=context, question=request.question), max_tokens=400)
        return QueryResponse(
            question=request.question,
            answer=answer,
            sources=['live_database'] + [m['type'] for m in retrieved_meta],
        )
    except Exception:
        # ChromaDB/RAG unavailable — try Ollama with live context only
        try:
            from src.llm.insights import ollama_generate
            from src.llm.prompts import RAG_PROMPT
            answer = ollama_generate(
                RAG_PROMPT.format(context=live_context, question=request.question),
                max_tokens=400,
            )
            return QueryResponse(question=request.question, answer=answer, sources=['live_database'])
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
