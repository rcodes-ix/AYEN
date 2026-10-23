"""AYEN — Supply Chain Signals | Streamlit MVP."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "ayen_synthetic_shipments.csv"
MODEL_PATH = ROOT / "models" / "disruption_model.joblib"
METRICS_PATH = ROOT / "models" / "model_metrics.json"

NUMERIC_FEATURES = [
    "order_quantity", "planned_lead_time_days", "supplier_reliability",
    "port_congestion", "weather_severity", "customs_delay_risk",
    "inventory_days", "historical_delay_rate", "shipment_cost"
]
CATEGORICAL_FEATURES = [
    "supplier_country", "transport_mode", "product_category", "criticality"
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_UPLOAD_COLUMNS = FEATURES + ["is_delayed"]

st.set_page_config(
    page_title="AYEN | Supply Chain Signals",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --bg:#0b1020; --panel:#111a2e; --panel2:#151f35; --line:#26334d;
        --text:#eef3ff; --muted:#9aa9c4; --teal:#45e0c1; --blue:#7c9cff; }
.stApp { background: radial-gradient(ellipse at 15% 0%, #172642 0%, #0b1020 42%); color:var(--text); }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
h1,h2,h3 { font-family:'Manrope',sans-serif; letter-spacing:-.035em; }
.block-container { padding-top:1.6rem; padding-bottom:3rem; max-width:1500px; }
[data-testid="stSidebar"] { background:#0b1222; border-right:1px solid #26334d; }
[data-testid="stSidebar"] * { color:#eef3ff; }
.hero { padding:1.55rem 1.7rem; border:1px solid #2a3a56; border-radius:22px;
        background:linear-gradient(115deg,rgba(69,224,193,.12),rgba(124,156,255,.10) 55%,rgba(17,26,46,.9));
        margin-bottom:1.25rem; }
.eyebrow { color:#45e0c1; text-transform:uppercase; letter-spacing:.18em; font-size:.72rem; font-weight:700; }
.hero-title { font-family:Manrope,sans-serif; font-size:2.2rem; font-weight:800; margin:.35rem 0 .45rem; color:#f4f7ff; }
.hero-sub { color:#aab8d0; max-width:800px; line-height:1.55; }
.pill { display:inline-block; border:1px solid #34516a; border-radius:999px; padding:.32rem .65rem;
        font-size:.76rem; color:#9feedd; margin-right:.35rem; }
.metric { background:linear-gradient(145deg,#131e33,#10182a); border:1px solid #263650;
          border-radius:17px; padding:1rem 1.1rem; min-height:112px; }
.metric-label { color:#9aa9c4; font-size:.82rem; margin-bottom:.55rem; }
.metric-value { color:#f1f5ff; font-family:Manrope,sans-serif; font-size:1.8rem; font-weight:800; }
.metric-foot { color:#7185a5; font-size:.76rem; margin-top:.25rem; }
.section-card { border:1px solid #263650; background:rgba(17,26,46,.8); border-radius:18px; padding:1.1rem 1.25rem; }
.small-muted { color:#9aa9c4; font-size:.88rem; }
div[data-testid="stMetric"] { background:#111a2e; border:1px solid #263650; border-radius:15px; padding:14px 16px; }
div[data-testid="stMetricLabel"] { color:#9aa9c4; }
.stButton>button { border-radius:10px; font-weight:700; border:1px solid #3a5578; }
.stButton>button[kind="primary"] { background:linear-gradient(100deg,#35cbb0,#72a0ff); color:#081222; border:none; }
hr { border-color:#26334d; }
</style>
""", unsafe_allow_html=True)

def load_data():
    if DATA_PATH.exists():
        try:
            return pd.read_csv(DATA_PATH)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

def load_metrics():
    if METRICS_PATH.exists():
        try:
            return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def money(value):
    return f"{value:,.0f}"

def risk_band(probability):
    if probability >= .70:
        return "HIGH", "High disruption risk"
    if probability >= .40:
        return "MEDIUM", "Moderate disruption risk"
    return "LOW", "Lower disruption risk"

def recommendations(row, probability):
    actions = []
    inventory = float(row.get("inventory_days", 0))
    planned = float(row.get("planned_lead_time_days", 0))
    criticality = str(row.get("criticality", "Medium"))
    reliability = float(row.get("supplier_reliability", .7))
    congestion = float(row.get("port_congestion", 0))
    customs = float(row.get("customs_delay_risk", 0))
    weather = float(row.get("weather_severity", 0))

    if probability >= .70:
        actions.append(("Priority", "Contact the supplier and freight forwarder now; request a dated status update and escalation path."))
    elif probability >= .40:
        actions.append(("Monitor", "Ask the supplier for milestone confirmation and review the shipment again at the next checkpoint."))
    else:
        actions.append(("Routine", "Continue normal tracking and retain the current delivery plan unless new evidence appears."))

    if inventory <= max(5, planned * .25):
        actions.append(("Inventory", "Inventory cover appears tight relative to the planned lead time. Confirm usable stock and near-term production demand before committing to a response."))
    if criticality in ("High", "Critical"):
        actions.append(("Production continuity", "Because the input is marked high-criticality, prepare a continuity option and confirm whether another approved source or substitute is available."))
    if reliability < .65:
        actions.append(("Supplier", "Review recent supplier performance and request a recovery plan; consider an approved backup supplier if qualification and cost permit."))
    if congestion >= .65:
        actions.append(("Transport", "Check port/route status with the logistics provider and compare feasible rerouting or alternate-mode options, including total cost."))
    if customs >= .65:
        actions.append(("Documentation", "Pre-check customs paperwork, permits, tariff classification, and document completeness with the clearing agent."))
    if weather >= .65:
        actions.append(("Weather", "Confirm route-specific weather exposure and ask the carrier whether the current ETA needs adjustment."))
    actions.append(("Cost control", "Compare the expected cost of expediting or backup sourcing with the operational cost of a stockout before approving extra spend."))
    return actions

def make_chart(df, x, title, color=None):
    if df.empty or x not in df.columns:
        st.info("Not enough data to display this chart yet.")
        return
    if color and color in df.columns:
        fig = px.histogram(df, x=x, color=color, barmode="group", title=title)
    else:
        fig = px.histogram(df, x=x, title=title)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#dce6fa", title_font_size=15, margin=dict(l=10,r=10,t=48,b=10),
        legend_title_text=""
    )
    fig.update_xaxes(gridcolor="#25334b")
    fig.update_yaxes(gridcolor="#25334b")
    st.plotly_chart(fig, use_container_width=True)

df = load_data()
model = load_model()
metrics = load_metrics()

with st.sidebar:
    st.markdown("## ◆ AYEN")
    st.caption("SUPPLY CHAIN SIGNALS")
    st.markdown("---")
    page = st.radio(
        "WORKSPACE",
        ["Overview", "New assessment", "Shipment explorer", "Data import", "Model transparency"],
        label_visibility="visible"
    )
    st.markdown("---")
    st.markdown('<span class="pill">MVP</span><span class="pill">Decision support</span>', unsafe_allow_html=True)
    st.caption("Predictions are estimates, not guarantees. Verify with current supplier and logistics information.")
    if model is None:
        st.error("Model not found. Run the training steps in the setup instructions.")
    else:
        st.success("Baseline model loaded")

st.markdown("""
<div class="hero">
  <div class="eyebrow">AI-powered disruption intelligence</div>
  <div class="hero-title">See the risk before it reaches production.</div>
  <div class="hero-sub">AYEN helps manufacturers assess shipment delay risk, understand operational warning signs, and choose practical next steps for critical inputs.</div>
  <div style="margin-top:1rem"><span class="pill">Risk assessment</span><span class="pill">Operational context</span><span class="pill">Action guidance</span></div>
</div>
""", unsafe_allow_html=True)

if page == "Overview":
    if df.empty:
        st.warning("The sample dataset is missing. Run `python generate_data.py` from the AYEN project folder.")
    else:
        total = len(df)
        delay_rate = float(df["is_delayed"].mean()) if "is_delayed" in df else 0
        avg_lead = float(df["actual_lead_time_days"].mean()) if "actual_lead_time_days" in df else 0
        high_critical = int(df["criticality"].isin(["High", "Critical"]).sum()) if "criticality" in df else 0
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric"><div class="metric-label">Sample shipments</div><div class="metric-value">{total:,}</div><div class="metric-foot">Synthetic demonstration records</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric"><div class="metric-label">Observed delay rate</div><div class="metric-value">{delay_rate:.1%}</div><div class="metric-foot">Within this synthetic dataset only</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric"><div class="metric-label">Mean actual lead time</div><div class="metric-value">{avg_lead:.1f} d</div><div class="metric-foot">Historical simulated outcome</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric"><div class="metric-label">High-criticality records</div><div class="metric-value">{high_critical:,}</div><div class="metric-foot">Marked High or Critical</div></div>', unsafe_allow_html=True)
        st.markdown("### Risk landscape")
        left, right = st.columns(2)
        with left:
            make_chart(df, "transport_mode", "Shipments by transport mode", "is_delayed")
        with right:
            make_chart(df, "product_category", "Shipments by input category", "is_delayed")
        st.markdown("### Recent sample records")
        display_cols = [c for c in ["shipment_id", "supplier_country", "transport_mode", "product_category", "planned_lead_time_days", "inventory_days", "criticality", "is_delayed"] if c in df]
        st.dataframe(df[display_cols].head(12), use_container_width=True, hide_index=True)
        st.caption("All figures on this overview are derived from AYEN's generated synthetic dataset, not real company or national supply-chain records.")

elif page == "New assessment":
    st.markdown("## Assess a shipment")
    st.write("Enter the information available before the shipment arrives. AYEN estimates delay probability and turns risk signals into a practical checklist.")
    if model is None:
        st.error("The trained model is missing. Run `python train_model.py` after generating the dataset.")
    else:
        with st.form("assessment_form"):
            st.markdown("#### Shipment profile")
            a, b, c = st.columns(3)
            with a:
                supplier_country = st.selectbox("Supplier country", ["China", "India", "Turkey", "UAE", "Germany", "Italy", "Saudi Arabia", "Kenya"])
                transport_mode = st.selectbox("Transport mode", ["Sea", "Air", "Road", "Rail"])
                product_category = st.selectbox("Input category", ["Raw materials", "Packaging", "Chemicals", "Spare parts", "Electronics", "Food ingredients"])
                criticality = st.selectbox("Production criticality", ["Low", "Medium", "High", "Critical"], index=2)
            with b:
                order_quantity = st.number_input("Order quantity (units)", min_value=1, max_value=10_000_000, value=1000, step=100)
                planned_lead_time_days = st.number_input("Planned lead time (days)", min_value=1, max_value=365, value=30)
                supplier_reliability = st.slider("Supplier reliability (0–1)", 0.0, 1.0, 0.80, 0.01, help="Estimate from past on-time performance; 1 means very reliable.")
                historical_delay_rate = st.slider("Supplier's historical delay rate (0–1)", 0.0, 1.0, 0.20, 0.01)
            with c:
                inventory_days = st.number_input("Current inventory cover (days)", min_value=0, max_value=365, value=12)
                shipment_cost = st.number_input("Estimated shipment cost (ETB)", min_value=0.0, value=150000.0, step=5000.0)
                port_congestion = st.slider("Port / route congestion (0–1)", 0.0, 1.0, 0.35, 0.01)
                weather_severity = st.slider("Weather disruption severity (0–1)", 0.0, 1.0, 0.20, 0.01)
                customs_delay_risk = st.slider("Customs delay risk (0–1)", 0.0, 1.0, 0.25, 0.01)
            submitted = st.form_submit_button("Assess disruption risk", type="primary", use_container_width=True)

        if submitted:
            row = {
                "order_quantity": int(order_quantity),
                "planned_lead_time_days": int(planned_lead_time_days),
                "supplier_reliability": float(supplier_reliability),
                "port_congestion": float(port_congestion),
                "weather_severity": float(weather_severity),
                "customs_delay_risk": float(customs_delay_risk),
                "inventory_days": int(inventory_days),
                "historical_delay_rate": float(historical_delay_rate),
                "shipment_cost": float(shipment_cost),
                "supplier_country": supplier_country,
                "transport_mode": transport_mode,
                "product_category": product_category,
                "criticality": criticality,
            }
            input_df = pd.DataFrame([row], columns=FEATURES)
            probability = float(model.predict_proba(input_df)[0, 1])
            band, label = risk_band(probability)
            st.markdown("---")
            st.markdown("### Assessment result")
            r1, r2, r3 = st.columns([1, 1, 2])
            with r1:
                st.metric("Estimated delay probability", f"{probability:.0%}")
            with r2:
                st.metric("Risk band", band)
            with r3:
                st.markdown(f"**{label}**")
                st.progress(probability)
                st.caption("Model estimate based on entered inputs and a synthetic-data baseline. It is not a confirmed ETA.")
            likely_delay_days = max(1, round(planned_lead_time_days * 0.15 + 3))
            st.markdown("### What this means operationally")
            if inventory_days <= planned_lead_time_days * .25:
                st.warning("Inventory cover is low relative to planned lead time. Confirm stock, consumption, and replenishment timing.")
            elif criticality in ("High", "Critical"):
                st.info("This input is marked production-critical. A contingency plan may be worthwhile even if the estimated risk is not high.")
            else:
                st.info("Use the estimate as a prompt for verification. Confirm milestone dates with the supplier or carrier.")
            st.markdown("### Recommended next steps")
            for i, (tag, action) in enumerate(recommendations(row, probability), start=1):
                with st.container(border=True):
                    st.markdown(f"**{i}. {tag}**")
                    st.write(action)
            st.caption(f"Planning note: if a delay occurs, a scenario of about {likely_delay_days} extra days can be used for discussion, but this is a rough planning assumption, not a model prediction of delay duration.")

elif page == "Shipment explorer":
    st.markdown("## Shipment explorer")
    st.write("Explore the generated sample data or, after importing, review the structure of your own records.")
    if df.empty:
        st.warning("No sample dataset found. Run `python generate_data.py`.")
    else:
        filt1, filt2, filt3 = st.columns(3)
        with filt1:
            modes = sorted(df["transport_mode"].dropna().unique().tolist())
            selected_modes = st.multiselect("Transport mode", modes, default=modes)
        with filt2:
            crits = sorted(df["criticality"].dropna().unique().tolist())
            selected_crits = st.multiselect("Criticality", crits, default=crits)
        with filt3:
            delay_choice = st.selectbox("Delay status", ["All", "Delayed", "Not delayed"])
        view = df[df["transport_mode"].isin(selected_modes) & df["criticality"].isin(selected_crits)]
        if delay_choice == "Delayed":
            view = view[view["is_delayed"] == 1]
        elif delay_choice == "Not delayed":
            view = view[view["is_delayed"] == 0]
        st.caption(f"{len(view):,} records match the selected filters.")
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.download_button("Download filtered records", data=view.to_csv(index=False).encode("utf-8"), file_name="ayen_filtered_shipments.csv", mime="text/csv")

elif page == "Data import":
    st.markdown("## Import company shipment history")
    st.write("Upload historical shipment records to inspect data quality and calculate company-specific summaries. Uploading does **not** silently retrain the baseline model.")
    st.info("For this MVP, the CSV needs the model input columns plus `is_delayed` (1 = actual lead time exceeded planned lead time; 0 = it did not). Records need known outcomes. Do not label shipments still in transit as on-time.")
    with st.expander("Required CSV columns", expanded=False):
        st.code(", ".join(REQUIRED_UPLOAD_COLUMNS))
        st.write("Numeric risk signals should be numeric; reliability and risk fields use a 0–1 scale. Extra columns are allowed and ignored for model summaries.")
    uploaded = st.file_uploader("Choose a historical shipment CSV", type=["csv"])
    if uploaded is not None:
        try:
            company_df = pd.read_csv(uploaded)
            st.markdown("### File preview")
            st.dataframe(company_df.head(10), use_container_width=True, hide_index=True)
            missing_cols = [c for c in REQUIRED_UPLOAD_COLUMNS if c not in company_df.columns]
            if missing_cols:
                st.error("This file is missing required columns: " + ", ".join(missing_cols))
                st.write("Rename or add the required columns before using this MVP's data-quality summary.")
            else:
                st.success(f"Required columns found. {len(company_df):,} rows loaded.")
                invalid_labels = sorted(set(company_df["is_delayed"].dropna().unique()) - {0, 1})
                numeric_errors = []
                for col in NUMERIC_FEATURES + ["is_delayed"]:
                    converted = pd.to_numeric(company_df[col], errors="coerce")
                    if converted.isna().sum() > company_df[col].isna().sum():
                        numeric_errors.append(col)
                if invalid_labels:
                    st.error(f"`is_delayed` must use only 0 and 1. Unexpected values: {invalid_labels}")
                elif numeric_errors:
                    st.error("Some numeric columns contain non-numeric values: " + ", ".join(numeric_errors))
                else:
                    miss = company_df[REQUIRED_UPLOAD_COLUMNS].isna().sum().sum()
                    delay_rate = pd.to_numeric(company_df["is_delayed"], errors="coerce").mean()
                    x, y, z = st.columns(3)
                    x.metric("Uploaded records", f"{len(company_df):,}")
                    y.metric("Historical delay rate", f"{delay_rate:.1%}")
                    z.metric("Missing required cells", f"{int(miss):,}")
                    st.markdown("### Company-specific patterns")
                    left, right = st.columns(2)
                    with left:
                        make_chart(company_df, "transport_mode", "Company records by transport mode", "is_delayed")
                    with right:
                        make_chart(company_df, "supplier_country", "Company records by supplier country", "is_delayed")
                    st.caption("This analysis stays in the current app session. The uploaded file is not automatically saved to disk or used to retrain the model.")
        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")

elif page == "Model transparency":
    st.markdown("## Model transparency")
    st.write("AYEN currently uses a supervised baseline classifier trained on synthetic shipment examples. It provides a risk estimate from shipment and supplier signals.")
    st.warning("Important limitation: synthetic data is useful for testing the product workflow, but model scores on synthetic data do not prove accuracy on real shipments.")
    if metrics:
        st.markdown("### Held-out synthetic test results")
        cols = st.columns(5)
        for col, key, label in zip(cols, ["accuracy", "precision", "recall", "f1", "roc_auc"], ["Accuracy", "Precision", "Recall", "F1 score", "ROC-AUC"]):
            val = metrics.get(key)
            col.metric(label, f"{val:.3f}" if isinstance(val, (float, int)) else "N/A")
        st.caption(f"Evaluated on {metrics.get('test_rows', 'N/A')} held-out synthetic records. These are not real-world validation results.")
    else:
        st.info("No evaluation metrics found. Run `python train_model.py`.")
    st.markdown("### How the label works")
    st.code("is_delayed = 1 if actual_lead_time_days > planned_lead_time_days else 0")
    st.markdown("### Model input signals")
    st.write(", ".join(FEATURES))
    st.markdown("### What is not implemented yet")
    st.write("- Live weather/news feeds or verified external disruption alerts")
    st.write("- Automatic retraining from customer uploads")
    st.write("- Validated delay-duration prediction and financial-loss estimation")
    st.write("- Real supplier directory, live ETAs, or authenticated multi-company accounts")
    st.markdown("### Sensible next step before real deployment")
    st.write("Test on real historical shipments with known outcomes, validate labels and missing data, measure performance by route and product category, and get feedback from supply-chain practitioners before using the estimates for operational decisions.")
