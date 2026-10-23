"""Train and evaluate AYEN's baseline delay-risk model."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "ayen_synthetic_shipments.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "disruption_model.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.json"

TARGET = "is_delayed"
# Only fields known before a shipment's outcome are used as predictors.
NUMERIC_FEATURES = [
    "order_quantity", "planned_lead_time_days", "supplier_reliability",
    "port_congestion", "weather_severity", "customs_delay_risk",
    "inventory_days", "historical_delay_rate", "shipment_cost"
]
CATEGORICAL_FEATURES = [
    "supplier_country", "transport_mode", "product_category", "criticality"
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found. Run: python generate_data.py")

    df = pd.read_csv(DATA_PATH)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if df[TARGET].nunique() != 2:
        raise ValueError(f"{TARGET} must contain both 0 and 1 labels.")

    X = df[FEATURES]
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ("categorical", categorical_pipe, CATEGORICAL_FEATURES)
    ])
    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced",
            random_state=42, n_jobs=-1
        ))
    ])
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    metrics = {
        "dataset_type": "synthetic",
        "test_rows": int(len(y_test)),
        "delay_rate_test": round(float(y_test.mean()), 4),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "threshold": 0.50,
        "features": FEATURES,
        "target": TARGET,
        "warning": "Evaluation uses synthetic data and does not establish real-world performance."
    }
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Saved model:", MODEL_PATH)
    print("Saved metrics:", METRICS_PATH)
    print(json.dumps(metrics, indent=2))
    print("\nClassification report:\n", classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, predictions))

if __name__ == "__main__":
    main()
