"""Generate a 5,000-row synthetic CSV for testing AYEN's manufacturer workflow.
This is fabricated demo data, not real company shipment history.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "data" / "manufacturer_demo.csv"
N = 5000


def main():
    rng = np.random.default_rng(42)
    order_quantity = rng.integers(50, 10001, size=N)
    planned_lead_time_days = rng.integers(3, 61, size=N)
    supplier_reliability = np.round(rng.beta(8, 2, size=N), 3)
    port_congestion = np.round(rng.beta(2, 2, size=N), 3)
    weather_severity = np.round(rng.beta(1.5, 4, size=N), 3)
    customs_delay_risk = np.round(rng.beta(2, 3, size=N), 3)
    inventory_days = rng.integers(0, 91, size=N)
    historical_delay_rate = np.round(rng.beta(2, 5, size=N), 3)
    shipment_cost = np.round(
        order_quantity * rng.uniform(15, 100, size=N) * rng.uniform(0.8, 1.5, size=N), 2
    )

    supplier_country = rng.choice(
        ["China", "India", "Turkey", "United Arab Emirates", "Germany", "Saudi Arabia", "Kenya", "Egypt"],
        size=N, p=[.30, .22, .14, .10, .08, .06, .05, .05],
    )
    transport_mode = rng.choice(["Sea", "Air", "Road", "Rail"], size=N, p=[.55, .12, .25, .08])
    product_category = rng.choice(
        ["Raw Materials", "Electronics", "Machinery", "Textiles", "Chemicals", "Packaging", "Spare Parts"],
        size=N,
    )
    criticality = rng.choice(["Low", "Medium", "High", "Critical"], size=N, p=[.15, .35, .35, .15])

    # Synthetic outcome rule for a demonstration only; not a real-world risk model.
    delay_score = (
        -1.65
        + 2.2 * (1 - supplier_reliability)
        + 1.25 * port_congestion
        + .85 * weather_severity
        + 1.15 * customs_delay_risk
        + 1.35 * historical_delay_rate
        - .012 * np.minimum(inventory_days, 60)
        - .012 * np.minimum(planned_lead_time_days, 45)
        + np.where(transport_mode == "Sea", .35, 0.0)
        + np.where(transport_mode == "Air", -.35, 0.0)
        + np.select(
            [criticality == "Low", criticality == "Medium", criticality == "High", criticality == "Critical"],
            [0.0, .05, .12, .18], default=0.0,
        )
    )
    delay_probability = np.clip(1 / (1 + np.exp(-delay_score)), .03, .95)
    is_delayed = rng.binomial(1, delay_probability)

    df = pd.DataFrame({
        "order_quantity": order_quantity,
        "planned_lead_time_days": planned_lead_time_days,
        "supplier_reliability": supplier_reliability,
        "port_congestion": port_congestion,
        "weather_severity": weather_severity,
        "customs_delay_risk": customs_delay_risk,
        "inventory_days": inventory_days,
        "historical_delay_rate": historical_delay_rate,
        "shipment_cost": shipment_cost,
        "supplier_country": supplier_country,
        "transport_mode": transport_mode,
        "product_category": product_category,
        "criticality": criticality,
        "is_delayed": is_delayed,
    })
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Created: {OUTPUT_PATH}")
    print(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
    print(f"Delayed: {int(df['is_delayed'].sum()):,} | Not delayed: {int((df['is_delayed'] == 0).sum()):,}")
    print(f"Delay rate: {df['is_delayed'].mean():.1%}")
    print("Reminder: this is synthetic demonstration data, not real company history.")


if __name__ == "__main__":
    main()
