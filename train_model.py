"""Train and evaluate AYEN's baseline delay-risk model.

The model is trained on synthetic demonstration data.
Its evaluation metrics must not be interpreted as evidence
of real-world predictive performance.
"""

from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
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


NUMERIC_FEATURES = [
    "order_quantity",
    "planned_lead_time_days",
    "supplier_reliability",
    "port_congestion",
    "weather_severity",
    "customs_delay_risk",
    "inventory_days",
    "historical_delay_rate",
    "shipment_cost",
    "djibouti_port_dwell_days",
    "modjo_dwell_days",
    "fx_availability_index",
]

CATEGORICAL_FEATURES = [
    "supplier_country",
    "transport_mode",
    "product_category",
    "criticality",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run python3 generate_data.py first."
        )

    df = pd.read_csv(DATA_PATH)

    # Validate required columns.
    missing = [
        column
        for column in FEATURES + [TARGET]
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}\n"
            "Regenerate the dataset with python3 generate_data.py."
        )

    if df.empty:
        raise ValueError("The dataset is empty.")

    if df[TARGET].isna().any():
        raise ValueError(f"{TARGET} contains missing labels.")

    if df[TARGET].nunique() != 2:
        raise ValueError(
            f"{TARGET} must contain both 0 and 1 labels."
        )

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError(
            f"{TARGET} must contain only 0 and 1."
        )

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )


    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipe,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipe,
                CATEGORICAL_FEATURES,
            ),
        ]
    )


    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("=" * 60)
    print("AYEN MODEL TRAINING")
    print("=" * 60)

    print(f"Dataset: {DATA_PATH}")
    print(f"Total rows: {len(df):,}")
    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows: {len(X_test):,}")

    print(
        f"Overall simulated delay rate: {y.mean():.1%}"
    )

    print("\nTraining model...")
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    threshold = 0.50
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "dataset_type": "synthetic",
        "total_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(y_test)),
        "delay_rate_overall": round(float(y.mean()), 4),
        "delay_rate_test": round(float(y_test.mean()), 4),
        "accuracy": round(
            float(accuracy_score(y_test, predictions)), 4
        ),
        "precision": round(
            float(
                precision_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),
            4,
        ),
        "recall": round(
            float(
                recall_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),
            4,
        ),
        "f1": round(
            float(
                f1_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),
            4,
        ),
        "roc_auc": round(
            float(roc_auc_score(y_test, probabilities)), 4
        ),
        "threshold": threshold,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "features": FEATURES,
        "target": TARGET,
        "excluded_columns": [
            "shipment_id",
            "supplier_id",
            "actual_lead_time_days",
        ],
        "warning": (
            "Evaluation uses synthetic demonstration data. "
            "These metrics do not establish real-world "
            "performance on Ethiopian logistics data."
        ),
    }


    joblib.dump(model, MODEL_PATH)

    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )


    print("\nModel saved:")
    print(MODEL_PATH)

    print("\nMetrics saved:")
    print(METRICS_PATH)

    print("\nEvaluation metrics:")
    print(json.dumps(metrics, indent=2))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nFeatures used by the model:")
    for feature in FEATURES:
        print(f"  - {feature}")

    print("\nTraining complete.")
    print(
        "Reminder: results are based on synthetic data, "
        "not validated real-world performance."
    )


if __name__ == "__main__":
    main()

