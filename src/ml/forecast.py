import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

from .features import FEATURE_COLS, FORECAST_CLASSES

def evaluate(y_test, y_pred):
    """Calculate MAE, RMSE, R²."""
    y_pred = np.maximum(y_pred, 0)
    return {
        'mae' : mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2'  : r2_score(y_test, y_pred)
    }

def train_linear_regression(datasets):
    """Train Linear Regression baseline for all classes."""
    results = {}
    for cls in FORECAST_CLASSES:
        X_train = datasets[cls]['X_train']
        y_train = datasets[cls]['y_train']
        X_test  = datasets[cls]['X_test']
        y_test  = datasets[cls]['y_test']
        with mlflow.start_run(run_name=f'LinearRegression_{cls}'):
            model  = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = np.maximum(model.predict(X_test), 0)
            metrics = evaluate(y_test, y_pred)
            mlflow.log_param('model', 'LinearRegression')
            mlflow.log_param('class_name', cls)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, 'model')
            results[cls] = {**metrics, 'y_test': y_test, 'y_pred': y_pred}
    return results

def train_xgboost(datasets):
    """Train XGBoost for all classes."""
    results = {}
    for cls in FORECAST_CLASSES:
        X_train = datasets[cls]['X_train']
        y_train = datasets[cls]['y_train']
        X_test  = datasets[cls]['X_test']
        y_test  = datasets[cls]['y_test']
        with mlflow.start_run(run_name=f'XGBoost_{cls}'):
            model = xgb.XGBRegressor(
                n_estimators=100, learning_rate=0.1,
                max_depth=3, random_state=42, verbosity=0)
            model.fit(X_train, y_train)
            y_pred  = np.maximum(model.predict(X_test), 0)
            metrics = evaluate(y_test, y_pred)
            mlflow.log_param('model', 'XGBoost')
            mlflow.log_param('class_name', cls)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, 'model')
            results[cls] = {**metrics, 'y_test': y_test, 'y_pred': y_pred}
    return results
