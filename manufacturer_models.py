"""Manufacturer-specific model storage and training for AYEN's demo MVP."""
from pathlib import Path
import json
import re

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
MANUFACTURERS_DIR = ROOT / "data" / "manufacturers"
NUMERIC_FEATURES = [
    "order_quantity", "planned_lead_time_days", "supplier_reliability",
    "port_congestion", "weather_severity", "customs_delay_risk",
    "inventory_days", "historical_delay_rate", "shipment_cost",
]
CATEGORICAL_FEATURES = [
    "supplier_country", "transport_mode", "product_category", "criticality",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "is_delayed"
REQUIRED_COLUMNS = FEATURES + [TARGET]
MINIMUM_ROWS = 50
MINIMUM_ROWS_PER_CLASS = 10


def manufacturer_slug(name):
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(name).strip()).strip("_").lower()
    if not slug:
        raise ValueError("Enter a valid manufacturer name.")
    return slug[:60]


def manufacturer_directory(name):
    return MANUFACTURERS_DIR / manufacturer_slug(name)


def display_manufacturer_name(slug):
    return str(slug).replace("_", " ").replace("-", " ").title()


def save_history(name, dataframe):
    df = validate_history(dataframe)
    directory = manufacturer_directory(name)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "history.csv"
    df.to_csv(path, index=False)
    return path


def load_history(name):
    path = manufacturer_directory(name) / "history.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        return None


def list_manufacturers():
    if not MANUFACTURERS_DIR.exists():
        return []
    return sorted(
        p.name for p in MANUFACTURERS_DIR.iterdir()
        if p.is_dir() and (p / "history.csv").exists()
    )


def list_trained_manufacturers():
    return [slug for slug in list_manufacturers()
            if (manufacturer_directory(slug) / "model.joblib").exists()]


def validate_history(dataframe):
    missing = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if dataframe.empty:
        raise ValueError("The uploaded dataset contains no records.")

    df = dataframe[REQUIRED_COLUMNS].copy()
    for column in NUMERIC_FEATURES + [TARGET]:
        original = df[column]
        converted = pd.to_numeric(original, errors="coerce")
        if (original.notna() & converted.isna()).any():
            raise ValueError(f"Column '{column}' contains non-numeric values.")
        df[column] = converted

    if df[TARGET].isna().any():
        raise ValueError("Every shipment needs a known is_delayed outcome (0 or 1).")
    if not df[TARGET].isin([0, 1]).all():
        raise ValueError("'is_delayed' must contain only 0 and 1.")
    if df[FEATURES].isna().any().any():
        raise ValueError("Required feature columns contain missing values. Complete or remove those rows.")

    for column in CATEGORICAL_FEATURES:
        df[column] = df[column].astype(str).str.strip()
        if df[column].eq("").any():
            raise ValueError(f"Column '{column}' contains empty values.")

    probability_fields = [
        "supplier_reliability", "port_congestion", "weather_severity",
        "customs_delay_risk", "historical_delay_rate",
    ]
    for column in probability_fields:
        if not df[column].between(0, 1).all():
            raise ValueError(f"'{column}' must contain values between 0 and 1.")
    if (df["order_quantity"] <= 0).any():
        raise ValueError("'order_quantity' must be greater than zero.")
    if (df["planned_lead_time_days"] <= 0).any():
        raise ValueError("'planned_lead_time_days' must be greater than zero.")
    if (df["inventory_days"] < 0).any():
        raise ValueError("'inventory_days' cannot be negative.")
    if (df["shipment_cost"] < 0).any():
        raise ValueError("'shipment_cost' cannot be negative.")

    df[TARGET] = df[TARGET].astype(int)
    if df[TARGET].nunique() != 2:
        raise ValueError("Training data must contain both delayed (1) and not-delayed (0) outcomes.")
    return df


def _build_model():
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocess", preprocessing),
        ("classifier", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )),
    ])


def train_manufacturer_model(name, dataframe):
    df = validate_history(dataframe)
    counts = df[TARGET].value_counts()
    if len(df) < MINIMUM_ROWS:
        raise ValueError(f"At least {MINIMUM_ROWS} records are required; this dataset has {len(df)}.")
    if int(counts.min()) < MINIMUM_ROWS_PER_CLASS:
        raise ValueError(
            f"Each outcome class needs at least {MINIMUM_ROWS_PER_CLASS} records. "
            f"Current counts: {counts.to_dict()}"
        )

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y,
    )
    evaluation_model = _build_model()
    evaluation_model.fit(X_train, y_train)
    probabilities = evaluation_model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    metrics = {
        "manufacturer": str(name),
        "dataset_type": "manufacturer_uploaded_history",
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "total_rows": int(len(df)),
        "delay_rate": round(float(y.mean()), 4),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "threshold": 0.50,
        "features": FEATURES,
        "target": TARGET,
        "evaluation_note": "Held-out historical records; metrics do not guarantee future performance.",
    }

    # Refit on all saved historical records for use in future assessments.
    final_model = _build_model()
    final_model.fit(X, y)
    directory = manufacturer_directory(name)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, directory / "model.joblib")
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_manufacturer_model(name):
    path = manufacturer_directory(name) / "model.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def load_manufacturer_metrics(name):
    path = manufacturer_directory(name) / "metrics.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
