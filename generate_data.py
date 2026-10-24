"""
Generate a reproducible, synthetic shipment dataset for AYEN.

The dataset is intentionally synthetic and is for development/demo use only.
It must not be represented as real Ethiopian logistics data.

"""

from pathlib import Path

import numpy as np
import pandas as pd



SEED = 42
N_ROWS = 6000

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

COUNTRIES = [
    "China",
    "India",
    "Turkey",
    "UAE",
    "Germany",
    "Italy",
    "Saudi Arabia",
    "Kenya",
]

MODES = ["Sea", "Air", "Road", "Rail"]

CATEGORIES = [
    "Raw materials",
    "Packaging",
    "Chemicals",
    "Spare parts",
    "Electronics",
    "Food ingredients",
]

SUPPLIERS = [f"SUP-{i:03d}" for i in range(1, 61)]


def main():
    rng = np.random.default_rng(SEED)
    n = N_ROWS


    supplier = rng.choice(SUPPLIERS, n)

    country = rng.choice(
        COUNTRIES,
        n,
        p=[0.28, 0.22, 0.12, 0.12, 0.09, 0.06, 0.06, 0.05],
    )

    mode = rng.choice(
        MODES,
        n,
        p=[0.55, 0.17, 0.20, 0.08],
    )

    category = rng.choice(
        CATEGORIES,
        n,
        p=[0.24, 0.19, 0.14, 0.17, 0.13, 0.13],
    )


    supplier_base = {
        s: float(rng.beta(8, 2))
        for s in SUPPLIERS
    }

    supplier_reliability = np.clip(
        np.array([supplier_base[s] for s in supplier])
        + rng.normal(0, 0.045, n),
        0.35,
        0.995,
    )



    planned_lead = np.select(
        [
            mode == "Sea",
            mode == "Air",
            mode == "Road",
            mode == "Rail",
        ],
        [
            rng.integers(24, 49, n),
            rng.integers(3, 12, n),
            rng.integers(8, 24, n),
            rng.integers(14, 33, n),
        ],
    ).astype(int)

    quantity = np.maximum(
        10,
        np.round(rng.lognormal(5.1, 0.75, n)),
    ).astype(int)


    port_congestion = np.clip(
        rng.beta(2.2, 2.0, n),
        0,
        1,
    )

    weather_severity = np.clip(
        rng.beta(1.6, 4.0, n),
        0,
        1,
    )

    customs_risk = np.clip(
        rng.beta(2.0, 3.3, n),
        0,
        1,
    )

    inventory_days = np.clip(
        np.round(rng.gamma(3.0, 5.0, n)),
        0,
        60,
    ).astype(int)

    criticality = rng.choice(
        ["Low", "Medium", "High", "Critical"],
        n,
        p=[0.15, 0.35, 0.33, 0.17],
    )

    historical_delay_rate = np.clip(
        1 - supplier_reliability
        + rng.normal(0, 0.07, n),
        0,
        1,
    )

   

    djibouti_base_dwell = rng.gamma(
        shape=2.0,
        scale=1.8,
        size=n,
    )

    djibouti_congestion_effect = (
        port_congestion * rng.uniform(1.0, 5.0, n)
    )

    djibouti_port_dwell_days = np.clip(
        np.round(
            djibouti_base_dwell
            + djibouti_congestion_effect
            + np.where(mode == "Sea", 1.5, 0.0)
        ),
        0,
        30,
    ).astype(int)

    modjo_base_dwell = rng.gamma(
        shape=1.8,
        scale=1.2,
        size=n,
    )

    modjo_customs_effect = customs_risk * rng.uniform(
        0.5,
        4.0,
        n,
    )

    modjo_port_effect = (
        djibouti_port_dwell_days
        * rng.uniform(0.03, 0.18, n)
    )

    modjo_dwell_days = np.clip(
        np.round(
            modjo_base_dwell
            + modjo_customs_effect
            + modjo_port_effect
        ),
        0,
        21,
    ).astype(int)


    fx_base = rng.beta(
        a=2.2,
        b=2.0,
        size=n,
    )

    fx_country_adjustment = pd.Series(country).map(
        {
            "China": -0.03,
            "India": 0.00,
            "Turkey": -0.02,
            "UAE": 0.04,
            "Germany": -0.03,
            "Italy": -0.03,
            "Saudi Arabia": 0.02,
            "Kenya": 0.05,
        }
    ).to_numpy()

    fx_category_adjustment = pd.Series(category).map(
        {
            "Raw materials": -0.04,
            "Packaging": 0.02,
            "Chemicals": -0.06,
            "Spare parts": -0.03,
            "Electronics": -0.05,
            "Food ingredients": 0.04,
        }
    ).to_numpy()

    fx_availability_index = np.clip(
        fx_base
        + fx_country_adjustment
        + fx_category_adjustment
        + rng.normal(0, 0.08, n),
        0.02,
        0.98,
    )

    mode_risk = (
        pd.Series(mode)
        .map(
            {
                "Sea": 0.18,
                "Air": -0.12,
                "Road": 0.05,
                "Rail": 0.08,
            }
        )
        .to_numpy()
    )

    country_risk = (
        pd.Series(country)
        .map(
            {
                "China": 0.05,
                "India": 0.03,
                "Turkey": 0.06,
                "UAE": -0.02,
                "Germany": -0.04,
                "Italy": -0.03,
                "Saudi Arabia": 0.00,
                "Kenya": -0.08,
            }
        )
        .to_numpy()
    )

    criticality_risk = (
        pd.Series(criticality)
        .map(
            {
                "Low": -0.12,
                "Medium": 0.00,
                "High": 0.08,
                "Critical": 0.14,
            }
        )
        .to_numpy()
    )


    djibouti_dwell_risk = (
        0.055 * djibouti_port_dwell_days
    )

    modjo_dwell_risk = (
        0.075 * modjo_dwell_days
    )

    fx_availability_risk = (
        0.90 * (1 - fx_availability_index)
    )

    logit = (
        -1.80
        + 2.45 * (1 - supplier_reliability)
        + 0.95 * port_congestion * (mode == "Sea")
        + 0.75 * weather_severity
        + 0.85 * customs_risk
        + 0.75 * historical_delay_rate
        + 0.45 * (planned_lead >= 30)
        + mode_risk
        + country_risk
        + criticality_risk
        - 0.012 * inventory_days
        + djibouti_dwell_risk
        + modjo_dwell_risk
        + fx_availability_risk
        + rng.normal(0, 0.38, n)
    )

    delay_probability = 1 / (1 + np.exp(-logit))

    delayed_event = rng.binomial(
        1,
        delay_probability,
    )

    extra_days = np.where(
        delayed_event == 1,
        np.maximum(
            1,
            np.round(rng.gamma(2.0, 3.2, n)),
        ).astype(int),
        np.maximum(
            0,
            np.round(rng.normal(0.3, 0.8, n)),
        ).astype(int),
    )

    actual_lead = planned_lead + extra_days

    # Shipment costs are synthetic and expressed in arbitrary
    # demonstration units, not verified market prices.
    shipment_cost = np.round(
        quantity
        * rng.uniform(1.5, 18.0, n)
        * np.where(mode == "Air", 2.5, 1.0),
        2,
    )


    df = pd.DataFrame(
        {
            "shipment_id": [
                f"AY-{i:06d}"
                for i in range(1, n + 1)
            ],
            "supplier_id": supplier,
            "supplier_country": country,
            "transport_mode": mode,
            "product_category": category,
            "order_quantity": quantity,
            "planned_lead_time_days": planned_lead,
            "supplier_reliability": np.round(
                supplier_reliability, 4
            ),
            "port_congestion": np.round(
                port_congestion, 4
            ),
            "weather_severity": np.round(
                weather_severity, 4
            ),
            "customs_delay_risk": np.round(
                customs_risk, 4
            ),
            "inventory_days": inventory_days,
            "criticality": criticality,
            "historical_delay_rate": np.round(
                historical_delay_rate, 4
            ),
            "shipment_cost": shipment_cost,

            # New corridor-specific features
            "djibouti_port_dwell_days": (
                djibouti_port_dwell_days
            ),
            "modjo_dwell_days": modjo_dwell_days,
            "fx_availability_index": np.round(
                fx_availability_index, 4
            ),

            # Outcome and simulated actual lead time
            "actual_lead_time_days": actual_lead,
            "is_delayed": (
                actual_lead > planned_lead
            ).astype(int),
        }
    )

    out = DATA_DIR / "ayen_synthetic_shipments.csv"

    df.to_csv(
        out,
        index=False,
    )

    assert len(df) == N_ROWS
    assert df["djibouti_port_dwell_days"].between(0, 30).all()
    assert df["modjo_dwell_days"].between(0, 21).all()
    assert df["fx_availability_index"].between(0, 1).all()
    assert df["is_delayed"].isin([0, 1]).all()

    print("=" * 60)
    print("AYEN SYNTHETIC SHIPMENT DATASET GENERATED")
    print("=" * 60)

    print(f"Output: {out}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Simulated delay rate: {df['is_delayed'].mean():.1%}")

    print("\nNew feature summaries:")

    print(
        "Djibouti port dwell (days): "
        f"mean={df['djibouti_port_dwell_days'].mean():.2f}, "
        f"median={df['djibouti_port_dwell_days'].median():.0f}"
    )

    print(
        "Modjo dwell (days): "
        f"mean={df['modjo_dwell_days'].mean():.2f}, "
        f"median={df['modjo_dwell_days'].median():.0f}"
    )

    print(
        "FX availability index: "
        f"mean={df['fx_availability_index'].mean():.2f}, "
        f"median={df['fx_availability_index'].median():.2f}"
    )

    print("\nDelay rate by FX availability group:")

    fx_groups = pd.cut(
        df["fx_availability_index"],
        bins=[0, 0.33, 0.66, 1.0],
        labels=["Low FX availability", "Medium", "High FX availability"],
        include_lowest=True,
    )

    print(
        df.groupby(
            fx_groups,
            observed=False,
        )["is_delayed"].mean().map(
            lambda value: f"{value:.1%}"
        ).to_string()
    )

    print("\nReminder:")
    print(
        "This dataset is synthetic demonstration data, "
        "not real-world Ethiopian logistics evidence."
    )


if __name__ == "__main__":
    main()
