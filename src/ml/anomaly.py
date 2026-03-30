import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

FEATURE_COLS_AD = ['Metal', 'Mixed Waste', 'Paper-Cardboard', 'Plastic', 'Wood', 'total_count']

def prepare_anomaly_features(df_daily):
    """Pivot daily data for anomaly detection."""
    df_pivot = df_daily.pivot_table(
        index='date', columns='class_name',
        values='annotation_count', aggfunc='sum', fill_value=0
    ).reset_index()
    df_pivot['total_count'] = df_pivot[
        ['Metal', 'Mixed Waste', 'Paper-Cardboard', 'Plastic', 'Wood']
    ].sum(axis=1)
    return df_pivot

def detect_anomalies(df_pivot, contamination=0.1):
    """Run Isolation Forest and return DataFrame with anomaly flags."""
    X        = df_pivot[FEATURE_COLS_AD].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    model.fit(X_scaled)

    df_pivot['anomaly_score'] = model.decision_function(X_scaled)
    df_pivot['is_anomaly']    = model.predict(X_scaled) == -1
    return df_pivot

def explain_anomaly(row, df_mean):
    """Generate business explanation for an anomalous day."""
    explanations = []
    if row['total_count'] > df_mean['total_count'] * 2:
        explanations.append(
            f"Total waste volume was {row['total_count']:.0f} — "
            f"more than 2x the daily average ({df_mean['total_count']:.0f})")
    for cls in ['Plastic', 'Paper-Cardboard', 'Mixed Waste', 'Metal', 'Wood']:
        if row[cls] > df_mean[cls] * 3:
            explanations.append(f"{cls} was unusually high at {row[cls]:.0f} (avg: {df_mean[cls]:.0f})")
        elif row[cls] == 0 and df_mean[cls] > 10:
            explanations.append(f"{cls} was completely absent (avg: {df_mean[cls]:.0f})")
    if not explanations:
        explanations.append("Unusual combination of waste class proportions detected")
    return ' | '.join(explanations)
