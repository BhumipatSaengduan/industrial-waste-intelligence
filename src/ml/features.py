import pandas as pd
import numpy as np

FEATURE_COLS = [
    'lag_1', 'lag_2', 'lag_3', 'lag_7',
    'rolling_mean_3', 'rolling_mean_7',
    'rolling_std_3',  'rolling_std_7',
    'day_of_week', 'day_of_month',
    'week_of_year', 'month', 'is_weekend'
]
TARGET_COL    = 'annotation_count'
TEST_DAYS     = 7
FORECAST_CLASSES = ['Plastic', 'Paper-Cardboard', 'Mixed Waste', 'Wood']

def create_features(df, class_name):
    """Create time-series features for a single waste class."""
    df_cls = df[df['class_name'] == class_name].copy()
    df_cls = df_cls.sort_values('date').reset_index(drop=True)

    date_range = pd.date_range(df_cls['date'].min(), df_cls['date'].max(), freq='D')
    df_full    = pd.DataFrame({'date': date_range})
    df_cls     = df_full.merge(df_cls, on='date', how='left')
    df_cls[TARGET_COL]   = df_cls[TARGET_COL].fillna(0)
    df_cls['class_name'] = class_name

    for lag in [1, 2, 3, 7]:
        df_cls[f'lag_{lag}'] = df_cls[TARGET_COL].shift(lag)

    for window in [3, 7]:
        df_cls[f'rolling_mean_{window}'] = df_cls[TARGET_COL].shift(1).rolling(window).mean()
        df_cls[f'rolling_std_{window}']  = df_cls[TARGET_COL].shift(1).rolling(window).std()

    df_cls['day_of_week']  = df_cls['date'].dt.dayofweek
    df_cls['day_of_month'] = df_cls['date'].dt.day
    df_cls['week_of_year'] = df_cls['date'].dt.isocalendar().week.astype(int)
    df_cls['month']        = df_cls['date'].dt.month
    df_cls['is_weekend']   = (df_cls['day_of_week'] >= 5).astype(int)

    return df_cls.dropna().reset_index(drop=True)

def prepare_datasets(df_daily):
    """Prepare train/test splits for all forecast classes."""
    datasets = {}
    for cls in FORECAST_CLASSES:
        df_cls    = create_features(df_daily, cls)
        split_idx = len(df_cls) - TEST_DAYS
        train     = df_cls.iloc[:split_idx]
        test      = df_cls.iloc[split_idx:]
        datasets[cls] = {
            'X_train': train[FEATURE_COLS],
            'y_train': train[TARGET_COL],
            'X_test' : test[FEATURE_COLS],
            'y_test' : test[TARGET_COL],
            'df'     : df_cls
        }
    return datasets
