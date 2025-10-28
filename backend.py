# backend.py
# FastAPI backend + ML utilities (training, preprocessing, prediction).
# Artifacts saved to ./artifacts
# Endpoints:
#   - GET  /health
#   - GET  /columns
#   - GET  /metrics
#   - POST /train  {data_path or csv_b64}
#   - POST /predict {features: {...}}

import os
import io
import json
import base64
import warnings
from functools import lru_cache
from typing import Any, Dict, Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ========= CONFIG =========
TARGET_COL = "Parts_Per_Hour"
ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "linear_regression_model.pkl")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
FEATURE_COLS_PATH = os.path.join(ARTIFACTS_DIR, "feature_columns.json")
ENCODERS_PATH = os.path.join(ARTIFACTS_DIR, "encoders.json")
IMPUTERS_PATH = os.path.join(ARTIFACTS_DIR, "imputers.json")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")
PLOT_PATH = os.path.join(ARTIFACTS_DIR, "actual_vs_pred.png")

DEFAULT_DATA_PATH = "data/manufacturing_dataset.csv"


# ========= UTILITIES =========
def ensure_artifacts_dir() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_json(path: str) -> Any:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_df_from_path_or_b64(data_path: Optional[str], csv_b64: Optional[str]) -> pd.DataFrame:
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
    elif csv_b64:
        raw = base64.b64decode(csv_b64.encode())
        df = pd.read_csv(io.BytesIO(raw))
    else:
        raise ValueError("No valid data_path found and no csv_b64 provided.")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in the dataset.")
    return df

def compute_imputers(df: pd.DataFrame, target: str) -> Dict[str, Dict[str, Any]]:
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    num_cols = [c for c in num_cols if c != target]
    numeric_medians = {col: float(df[col].median()) if not df[col].dropna().empty else 0.0 for col in num_cols}
    categorical_modes = {}
    for col in cat_cols:
        if not df[col].dropna().empty:
            categorical_modes[col] = str(df[col].mode(dropna=True)[0])
        else:
            categorical_modes[col] = ""
    return {"numeric_medians": numeric_medians, "categorical_modes": categorical_modes}

def impute_df(df: pd.DataFrame, imputers: Dict[str, Dict[str, Any]], target: str) -> pd.DataFrame:
    df = df.copy()
    num_medians = imputers.get("numeric_medians", {})
    cat_modes = imputers.get("categorical_modes", {})
    for col, med in num_medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(med)
    for col, mode_val in cat_modes.items():
        if col in df.columns:
            df[col] = df[col].fillna(mode_val)
    if target in df.columns:
        df = df.dropna(subset=[target])
    return df

def build_label_encoders(df: pd.DataFrame) -> Dict[str, Any]:
    encoders = {}
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        classes = sorted([str(x) for x in df[col].astype(str).unique().tolist()])
        mapping = {cls: i for i, cls in enumerate(classes)}
        mode_val = str(df[col].mode(dropna=True)[0]) if not df[col].dropna().empty else (classes[0] if classes else "")
        encoders[col] = {"classes": classes, "mapping": mapping, "mode": mode_val}
    return encoders

def apply_encoders(df: pd.DataFrame, encoders: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy()
    for col, info in encoders.items():
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str)
        mapping = info["mapping"]
        mode_val = info["mode"]
        default_code = mapping.get(mode_val, 0)
        df[col] = df[col].map(lambda x: mapping.get(x, default_code))
    return df

def evaluate_and_plot(y_test: np.ndarray, y_pred: np.ndarray, plot_path: str) -> Dict[str, float]:
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color="blue", label="Predicted")
    line_min = min(y_test.min(), y_pred.min())
    line_max = max(y_test.max(), y_pred.max())
    plt.plot([line_min, line_max], [line_min, line_max], color="red", linewidth=2, label="Perfect Fit")
    plt.xlabel("Actual Parts Per Hour")
    plt.ylabel("Predicted Parts Per Hour")
    plt.title("Linear Regression: Actual vs Predicted")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()
    return {"mse": float(mse), "rmse": rmse, "r2": r2}

def train_model_from_df(df: pd.DataFrame) -> Dict[str, Any]:
    ensure_artifacts_dir()
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")
    imputers = compute_imputers(df, TARGET_COL)
    df_imp = impute_df(df, imputers, TARGET_COL)
    encoders = build_label_encoders(df_imp)
    df_enc = apply_encoders(df_imp, encoders)
    feature_cols = [c for c in df_enc.columns if c != TARGET_COL]
    X = df_enc[feature_cols].astype(float)
    y = df_enc[TARGET_COL].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    metrics = evaluate_and_plot(y_test.values, y_pred, PLOT_PATH)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    save_json(FEATURE_COLS_PATH, feature_cols)
    save_json(ENCODERS_PATH, encoders)
    save_json(IMPUTERS_PATH, imputers)
    save_json(METRICS_PATH, metrics)
    return {
        "n_rows": int(df.shape[0]),
        "n_features": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "metrics": metrics,
        "artifacts": {
            "model": MODEL_PATH,
            "scaler": SCALER_PATH,
            "feature_columns": FEATURE_COLS_PATH,
            "encoders": ENCODERS_PATH,
            "imputers": IMPUTERS_PATH,
            "metrics": METRICS_PATH,
            "plot": PLOT_PATH,
        },
    }

@lru_cache(maxsize=1)
def get_artifacts() -> Dict[str, Any]:
    model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    feature_cols = load_json(FEATURE_COLS_PATH)
    encoders = load_json(ENCODERS_PATH) or {}
    imputers = load_json(IMPUTERS_PATH) or {"numeric_medians": {}, "categorical_modes": {}}
    metrics = load_json(METRICS_PATH) or {}
    return {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "encoders": encoders,
        "imputers": imputers,
        "metrics": metrics,
    }

def clear_artifacts_cache():
    try:
        get_artifacts.cache_clear()
    except Exception:
        pass

def preprocess_for_inference(features: Dict[str, Any]) -> pd.DataFrame:
    art = get_artifacts()
    feature_cols = art["feature_cols"]
    encoders = art["encoders"]
    imputers = art["imputers"]
    if not feature_cols:
        raise ValueError("Model is not trained yet. Train the model to create feature_columns.json.")
    row = {}
    cat_cols = set(encoders.keys())
    num_medians = imputers.get("numeric_medians", {})
    cat_modes = imputers.get("categorical_modes", {})
    for col in feature_cols:
        if col in features:
            val = features[col]
        else:
            if col in cat_cols:
                val = cat_modes.get(col, "")
            else:
                val = num_medians.get(col, 0.0)
        row[col] = val
    df_row = pd.DataFrame([row])
    df_row = impute_df(df_row, imputers, target="___no_target___")
    df_row = apply_encoders(df_row, encoders)
    for col in feature_cols:
        if col not in df_row.columns:
            df_row[col] = 0.0
    return df_row[feature_cols].astype(float)


# ========= FASTAPI BACKEND =========
api = FastAPI(
    title="Manufacturing Parts Per Hour API",
    version="1.0.0",
    description="Train and predict Parts_Per_Hour using Linear Regression.",
)
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrainRequest(BaseModel):
    data_path: Optional[str] = None
    csv_b64: Optional[str] = None

class PredictRequest(BaseModel):
    features: Dict[str, Any]

@api.get("/health")
def health():
    return {"status": "ok"}

@api.get("/columns")
def get_columns():
    art = get_artifacts()
    feature_cols = art.get("feature_cols")
    encoders = art.get("encoders", {})
    if not feature_cols:
        raise HTTPException(status_code=400, detail="Model not trained yet. Please train first.")
    column_types = {c: ("categorical" if c in encoders else "numeric") for c in feature_cols}
    categorical_classes = {c: info.get("classes", []) for c, info in encoders.items()}
    imputers = art.get("imputers", {})
    return {
        "feature_columns": feature_cols,
        "column_types": column_types,
        "categorical_classes": categorical_classes,
        "imputers": imputers,
    }

@api.get("/metrics")
def metrics():
    art = get_artifacts()
    return {"metrics": art.get("metrics", {}), "plot_path": PLOT_PATH if os.path.exists(PLOT_PATH) else None}

@api.post("/train")
def train(req: TrainRequest):
    try:
        df = read_df_from_path_or_b64(req.data_path, req.csv_b64)
        info = train_model_from_df(df)
        clear_artifacts_cache()
        return {"message": "Model trained and artifacts saved.", "details": info}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api.post("/predict")
def predict(req: PredictRequest):
    art = get_artifacts()
    if not art.get("model") or not art.get("scaler") or not art.get("feature_cols"):
        raise HTTPException(status_code=400, detail="Model not trained yet. Please train first.")
    try:
        X = preprocess_for_inference(req.features)
        X_scaled = art["scaler"].transform(X)
        y_pred = float(art["model"].predict(X_scaled)[0])
        return {"prediction": y_pred}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))