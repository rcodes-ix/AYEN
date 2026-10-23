# AYEN — Manufacturer-aware Supply Chain Signals MVP

This version includes a general baseline model workflow and manufacturer-specific shipment histories and models. It is a demo MVP, not production-ready multi-company software.

## What's included

- `app.py`: five-page Streamlit app. Overview and Shipment Explorer can switch between the general dataset and each saved manufacturer's history. New Assessment can choose the general model or a trained manufacturer's model. Model Transparency can inspect baseline or manufacturer-specific metrics.
- `manufacturer_models.py`: validation, local history storage, model training/loading, and metrics helpers.
- `generate_data.py`: generates the 6,000-row general synthetic dataset.
- `train_model.py`: trains/evaluates the general baseline model.
- `generate_manufacturer_data.py`: generates a separate 5,000-row CSV for demonstrating the upload/train flow.
- `data/ayen_synthetic_shipments.csv`: generated general demo data.
- `data/manufacturer_demo.csv`: generated manufacturer upload demo data (14 columns: 13 model features plus `is_delayed`).

## Run locally (WSL/Linux)

From the extracted project directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

If you already have a working virtual environment for AYEN, activate it and run `python -m pip install -r requirements.txt`, then `python train_model.py` and `streamlit run app.py`.

The baseline training step creates `models/disruption_model.joblib` and `models/model_metrics.json`.

## Demonstrate manufacturer-specific training

1. Launch the app and open **Data import**.
2. Create a profile such as `Addis Manufacturing`.
3. Upload `data/manufacturer_demo.csv` from the extracted project folder.
4. Save the history.
5. In **Train a manufacturer-specific model**, select the profile and click **Train and evaluate model**.
6. Visit **New assessment** and choose **Manufacturer-Specific Model**. Select the trained manufacturer.
7. Check **Overview**, **Shipment explorer**, and **Model transparency**; each page has a source/model selector.

The manufacturer CSV is synthetic and exists only to demonstrate the workflow. Do not present it as real company data or use its metrics as evidence of real-world predictive accuracy.

## CSV schema

The upload must contain these columns:

`order_quantity, planned_lead_time_days, supplier_reliability, port_congestion, weather_severity, customs_delay_risk, inventory_days, historical_delay_rate, shipment_cost, supplier_country, transport_mode, product_category, criticality, is_delayed`

`is_delayed` must be 1 for a known delayed shipment and 0 for a known not-delayed shipment. Only upload records whose outcomes are known. The app requires at least 50 valid rows and at least 10 records in each outcome class before training.

## Important limitations

- Both included datasets are synthetic. Model metrics do not establish real-world predictive performance.
- Manufacturer histories and models are stored under `data/manufacturers/` on the local filesystem. This is suitable for a demo, not secure private multi-company storage.
- On Streamlit Cloud, local files may not persist reliably across restarts/redeployments. Production deployment needs durable storage, authentication, and strict company-level access controls.
- When deploying, ensure the baseline model artifact is available to the app. Run `python train_model.py` in the deployment/build workflow or deliberately manage the generated model artifact in your deployment process.
