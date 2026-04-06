import pandas as pd
import requests
from .prompts import WEEKLY_REPORT_PROMPT, ANOMALY_EXPLANATION_PROMPT

OLLAMA_URL   = 'http://localhost:11434'
OLLAMA_MODEL = 'llama3.2:3b'

def ollama_generate(prompt, model=OLLAMA_MODEL, max_tokens=300):
    """Send prompt to Ollama and return response text."""
    response = requests.post(
        f'{OLLAMA_URL}/api/generate',
        json={'model': model, 'prompt': prompt,
              'stream': False, 'options': {'num_predict': max_tokens}},
        timeout=60,
    )
    return response.json()['response']

def build_weekly_context(df_weekly, df_anomaly, week_start=None):
    """Format weekly data as text context for LLM."""
    if week_start is None:
        week_start = df_weekly['week_start'].max()
    df_week = df_weekly[df_weekly['week_start'] == week_start]

    lines = [f"Weekly Waste Report — Week of {str(week_start)[:10]}\n"]
    lines.append("Waste composition this week:")
    for _, row in df_week.iterrows():
        lines.append(f"  - {row['class_name']}: {int(row['annotation_count'])} annotations "
                    f"across {int(row['image_count'])} images")

    week_end       = pd.Timestamp(week_start) + pd.Timedelta(days=6)
    df_anom_week   = df_anomaly[
        (pd.to_datetime(df_anomaly['date']) >= pd.Timestamp(week_start)) &
        (pd.to_datetime(df_anomaly['date']) <= week_end)
    ]
    if len(df_anom_week) > 0:
        lines.append("\nAnomalous days detected this week:")
        for _, row in df_anom_week.iterrows():
            lines.append(f"  - {str(row['date'])[:10]}: {row['explanation']}")
    else:
        lines.append("\nNo anomalous days detected this week.")
    return '\n'.join(lines)

def generate_weekly_report(context):
    """Generate weekly operational report using LLM."""
    return ollama_generate(WEEKLY_REPORT_PROMPT.format(context=context), max_tokens=300)

def generate_anomaly_explanation(row):
    """Generate LLM explanation for a single anomaly day."""
    prompt = ANOMALY_EXPLANATION_PROMPT.format(
        date          = str(row['date'])[:10],
        total_count   = int(row['annotation_count']),
        anomaly_score = float(row['anomaly_score']),
        explanation   = row['explanation']
    )
    return ollama_generate(prompt, max_tokens=150)
