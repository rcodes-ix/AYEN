"""Generate a reproducible, synthetic shipment dataset for AYEN.

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

COUNTRIES = ["China", "India", "Turkey", "UAE", "Germany", "Italy", "Saudi Arabia", "Kenya"]
MODES = ["Sea", "Air", "Road", "Rail"]
CATEGORIES = ["Raw materials", "Packaging", "Chemicals", "Spare parts", "Electronics", "Food ingredients"]
SUPPLIERS = [f"SUP-{i:03d}" for i in range(1, 61)]

def main():
    rng = np.random.default_rng(SEED)
    n = N_ROWS

    supplier = rng.choice(SUPPLIERS, n)
    country = rng.choice(COUNTRIES, n, p=[.28, .22, .12, .12, .09, .06, .06, .05])
    mode = rng.choice(MODES, n, p=[.55, .17, .20, .08])
    category = rng.choice(CATEGORIES, n, p=[.24, .19, .14, .17, .13, .13])

    # Supplier-level reliability is stable enough to learn from, but not deterministic.
    supplier_base = {s: float(rng.beta(8, 2)) for s in SUPPLIERS}
    supplier_reliability = np.clip(
        np.array([supplier_base[s] for s in supplier]) + rng.normal(0, .045, n), .35, .995
    )

    planned_lead = np.select(
        [mode == "Sea", mode == "Air", mode == "Road", mode == "Rail"],
        [rng.integers(24, 49, n), rng.integers(3, 12, n),
         rng.integers(8, 24, n), rng.integers(14, 33, n)]
    ).astype(int)
    quantity = np.maximum(10, np.round(rng.lognormal(5.1, .75, n))).astype(int)
    port_congestion = np.clip(rng.beta(2.2, 2.0, n), 0, 1)
    weather_severity = np.clip(rng.beta(1.6, 4.0, n), 0, 1)
    customs_risk = np.clip(rng.beta(2.0, 3.3, n), 0, 1)
    inventory_days = np.clip(np.round(rng.gamma(3.0, 5.0, n)), 0, 60).astype(int)
    criticality = rng.choice(["Low", "Medium", "High", "Critical"], n, p=[.15, .35, .33, .17])
    historical_delay_rate = np.clip(1 - supplier_reliability + rng.normal(0, .07, n), 0, 1)

    mode_risk = pd.Series(mode).map({"Sea": .18, "Air": -.12, "Road": .05, "Rail": .08}).to_numpy()
    country_risk = pd.Series(country).map({
        "China": .05, "India": .03, "Turkey": .06, "UAE": -.02,
        "Germany": -.04, "Italy": -.03, "Saudi Arabia": .00, "Kenya": -.08
    }).to_numpy()
    criticality_risk = pd.Series(criticality).map(
        {"Low": -.12, "Medium": 0, "High": .08, "Critical": .14}
    ).to_numpy()

    # Latent delay probability. The label is derived from simulated actual lead time.
    logit = (
        -1.55
        + 2.45 * (1 - supplier_reliability)
        + .95 * port_congestion * (mode == "Sea")
        + .75 * weather_severity
        + .85 * customs_risk
        + .75 * historical_delay_rate
        + .45 * (planned_lead >= 30)
        + mode_risk + country_risk + criticality_risk
        - .012 * inventory_days
        + rng.normal(0, .38, n)
    )
    delay_probability = 1 / (1 + np.exp(-logit))
    delayed_event = rng.binomial(1, delay_probability)
    extra_days = np.where(
        delayed_event == 1,
        np.maximum(1, np.round(rng.gamma(2.0, 3.2, n))).astype(int),
        np.maximum(0, np.round(rng.normal(.3, .8, n))).astype(int)
    )
    actual_lead = planned_lead + extra_days
    shipment_cost = np.round(
        quantity * rng.uniform(1.5, 18.0, n) * np.where(mode == "Air", 2.5, 1.0), 2
    )

    df = pd.DataFrame({
        "shipment_id": [f"AY-{i:06d}" for i in range(1, n + 1)],
        "supplier_id": supplier,
        "supplier_country": country,
        "transport_mode": mode,
        "product_category": category,
        "order_quantity": quantity,
        "planned_lead_time_days": planned_lead,
        "supplier_reliability": np.round(supplier_reliability, 4),
        "port_congestion": np.round(port_congestion, 4),
        "weather_severity": np.round(weather_severity, 4),
        "customs_delay_risk": np.round(customs_risk, 4),
        "inventory_days": inventory_days,
        "criticality": criticality,
        "historical_delay_rate": np.round(historical_delay_rate, 4),
        "shipment_cost": shipment_cost,
        "actual_lead_time_days": actual_lead,
        "is_delayed": (actual_lead > planned_lead).astype(int),
    })

    out = DATA_DIR / "ayen_synthetic_shipments.csv"
    df.to_csv(out, index=False)
    print(f"Created {out}")
    print(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
    print(f"Delay rate: {df['is_delayed'].mean():.1%}")
    print("Reminder: synthetic demo data only; not real-world evidence.")

if __name__ == "__main__":
    main()
