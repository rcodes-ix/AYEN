from pathlib import Path
import json

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from manufacturer_models import (
    FEATURES as MANUFACTURER_FEATURES,
    REQUIRED_COLUMNS as MANUFACTURER_REQUIRED_COLUMNS,
    display_manufacturer_name,
    list_manufacturers,
    list_trained_manufacturers,
    load_history,
    load_manufacturer_metrics,
    load_manufacturer_model,
    save_history,
    train_manufacturer_model,
    validate_history,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "ayen_synthetic_shipments.csv"
MODEL_PATH = ROOT / "models" / "disruption_model.joblib"
METRICS_PATH = ROOT / "models" / "model_metrics.json"

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
REQUIRED_UPLOAD_COLUMNS = FEATURES + ["is_delayed"]

st.set_page_config(
    page_title="AYEN | Supply Chain Signals",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
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
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None


def load_metrics():
    if METRICS_PATH.exists():
        try:
            return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def risk_band(probability):
    if probability >= 0.70:
        return "HIGH", "High disruption risk"
    if probability >= 0.40:
        return "MEDIUM", "Moderate disruption risk"
    return "LOW", "Lower disruption risk"


def feature_label(feature):
    labels = {
        "order_quantity": "Order quantity",
        "planned_lead_time_days": "Planned lead time",
        "supplier_reliability": "Supplier reliability",
        "port_congestion": "Port / route congestion",
        "weather_severity": "Weather severity",
        "customs_delay_risk": "Customs delay risk",
        "inventory_days": "Inventory cover",
        "historical_delay_rate": "Historical supplier delay rate",
        "shipment_cost": "Shipment cost",
        "djibouti_port_dwell_days": "Djibouti port dwell time",
        "modjo_dwell_days": "Modjo dry port dwell time",
        "fx_availability_index": "FX availability",
        "supplier_country": "Supplier country",
        "transport_mode": "Transport mode",
        "product_category": "Input category",
        "criticality": "Production criticality",
    }
    return labels.get(feature, feature.replace("_", " ").title())


def format_feature_value(feature, value):
    if feature in {
        "supplier_reliability",
        "port_congestion",
        "weather_severity",
        "customs_delay_risk",
        "historical_delay_rate",
        "fx_availability_index",
    }:
        return f"{float(value):.0%}"
    if feature in {"djibouti_port_dwell_days", "modjo_dwell_days"}:
        return f"{float(value):.1f} days"
    if feature == "inventory_days":
        return f"{float(value):.0f} days"
    if feature == "planned_lead_time_days":
        return f"{float(value):.0f} days"
    if feature == "shipment_cost":
        return f"{float(value):,.0f} ETB"
    if feature == "order_quantity":
        return f"{float(value):,.0f} units"
    return str(value)


def make_reference_row(reference_df, features):
    reference = {}

    for feature in features:
        if feature not in reference_df.columns:
            continue

        series = reference_df[feature].dropna()
        if series.empty:
            continue

        if pd.api.types.is_numeric_dtype(series):
            reference[feature] = float(series.median())
        else:
            modes = series.mode()
            reference[feature] = str(modes.iloc[0]) if not modes.empty else str(series.iloc[0])

    return reference


def explain_prediction(selected_model, row, features, reference_df):
    available_features = [feature for feature in features if feature in row]
    if not available_features:
        return []

    reference = make_reference_row(reference_df, available_features)
    current_df = pd.DataFrame([{feature: row[feature] for feature in available_features}])
    current_probability = float(selected_model.predict_proba(current_df)[0, 1])

    explanations = []

    for feature in available_features:
        if feature not in reference:
            continue

        comparison_row = {
            item: row[item]
            for item in available_features
        }
        comparison_row[feature] = reference[feature]

        comparison_df = pd.DataFrame([comparison_row])
        comparison_probability = float(selected_model.predict_proba(comparison_df)[0, 1])
        difference = current_probability - comparison_probability

        explanations.append({
            "feature": feature,
            "value": row[feature],
            "difference": difference,
            "reference": reference[feature],
        })

    explanations.sort(key=lambda item: abs(item["difference"]), reverse=True)
    return explanations[:3]


def recommendations(row, probability):
    actions = []

    inventory = float(row.get("inventory_days", 0))
    planned = float(row.get("planned_lead_time_days", 0))
    criticality = str(row.get("criticality", "Medium"))
    reliability = float(row.get("supplier_reliability", 0.7))
    congestion = float(row.get("port_congestion", 0))
    customs = float(row.get("customs_delay_risk", 0))
    weather = float(row.get("weather_severity", 0))
    djibouti_dwell = float(row.get("djibouti_port_dwell_days", 0))
    modjo_dwell = float(row.get("modjo_dwell_days", 0))
    fx_availability = float(row.get("fx_availability_index", 1))

    if probability >= 0.70:
        actions.append((
            "Priority",
            "Contact the supplier and freight forwarder now. Request a dated status update, confirmation of the current shipment location, and a clear escalation path."
        ))
    elif probability >= 0.40:
        actions.append((
            "Monitor",
            "Ask the supplier for milestone confirmation and review the shipment again at the next checkpoint."
        ))
    else:
        actions.append((
            "Routine",
            "Continue normal tracking and retain the current delivery plan unless new evidence appears."
        ))

    if inventory <= max(5, planned * 0.25):
        actions.append((
            "Inventory",
            "Inventory cover appears tight relative to planned lead time. Confirm usable stock and near-term production demand before choosing a response."
        ))

    if criticality in ("High", "Critical"):
        actions.append((
            "Production continuity",
            "This input is marked high-criticality. Prepare a continuity option and confirm whether an approved alternative source or substitute is available."
        ))

    if reliability < 0.65:
        actions.append((
            "Supplier",
            "Review recent supplier performance and request a recovery plan. Consider an approved backup supplier if qualification and cost permit."
        ))

    if congestion >= 0.65:
        actions.append((
            "Transport",
            "Check port and route status with the logistics provider. Compare feasible rerouting or alternate-mode options, including total cost."
        ))

    if customs >= 0.65:
        actions.append((
            "Documentation",
            "Pre-check customs paperwork, permits, tariff classification, and document completeness with the clearing agent."
        ))

    if weather >= 0.65:
        actions.append((
            "Weather",
            "Confirm route-specific weather exposure and ask the carrier whether the current ETA needs adjustment."
        ))

    if djibouti_dwell >= 7:
        actions.append((
            "Djibouti port",
            "The entered Djibouti dwell time is elevated. Verify the latest port status, release requirements, container availability, and onward transport arrangements."
        ))

    if modjo_dwell >= 4:
        actions.append((
            "Modjo dry port",
            "The entered Modjo dwell time is elevated. Confirm customs-release status, outstanding documents, storage charges, and the planned dispatch date."
        ))

    if fx_availability <= 0.35:
        actions.append((
            "Foreign exchange",
            "The entered FX availability is low. Confirm payment readiness and supplier settlement requirements with the responsible finance team; do not assume this alone confirms a shipment delay."
        ))

    actions.append((
        "Cost control",
        "Compare the expected cost of expediting or backup sourcing with the operational cost of a stockout before approving extra spending."
    ))

    return actions


def make_chart(dataframe, column, title, color=None):
    if dataframe is None or dataframe.empty or column not in dataframe.columns:
        st.info("Not enough data to display this chart yet.")
        return

    if color and color in dataframe.columns:
        fig = px.histogram(dataframe, x=column, color=color, barmode="group", title=title)
    else:
        fig = px.histogram(dataframe, x=column, title=title)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#dce6fa",
        title_font_size=15,
        margin=dict(l=10, r=10, t=48, b=10),
        legend_title_text="",
    )
    fig.update_xaxes(gridcolor="#25334b")
    fig.update_yaxes(gridcolor="#25334b")
    st.plotly_chart(fig, use_container_width=True)


def choose_data_source(general_data, key):
    manufacturer_slugs = list_manufacturers()
    options = ["__general__"] + manufacturer_slugs

    selected = st.selectbox(
        "Data source",
        options,
        key=key,
        format_func=lambda value: (
            "General AYEN dataset" if value == "__general__"
            else display_manufacturer_name(value)
        ),
    )

    if selected == "__general__":
        return general_data, "General AYEN dataset", True

    history = load_history(selected)
    if history is None:
        st.error("This manufacturer's saved history could not be loaded.")
        return pd.DataFrame(), display_manufacturer_name(selected), False

    return history, display_manufacturer_name(selected), False


def show_model_metrics(metrics, title, baseline=False):
    if not metrics:
        st.info("No evaluation metrics are available for this model yet.")
        return

    st.markdown(f"### {title}")
    cols = st.columns(5)

    for col, key, label in zip(
        cols,
        ["accuracy", "precision", "recall", "f1", "roc_auc"],
        ["Accuracy", "Precision", "Recall", "F1 score", "ROC-AUC"],
    ):
        value = metrics.get(key)
        col.metric(label, f"{value:.3f}" if isinstance(value, (int, float)) else "N/A")

    st.caption(
        f"Evaluation records: {metrics.get('test_rows', 'N/A')}. "
        + (
            "These results use synthetic baseline data."
            if baseline
            else "These results use a held-out split of uploaded historical data."
        )
        + " Metrics do not guarantee future performance."
    )


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
        label_visibility="visible",
    )

    st.markdown("---")
    st.markdown(
        '<span class="pill">MVP</span><span class="pill">Decision support</span>',
        unsafe_allow_html=True,
    )
    st.caption("Predictions are estimates, not guarantees. Verify with current supplier and logistics information.")

    if model is None:
        st.warning("Baseline model not found. Run `python train_model.py` after generating the baseline dataset.")
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
    st.markdown("## Supply-chain overview")
    st.write("Choose whether to view the general demo dataset or one manufacturer's saved historical shipment records.")

    selected_df, source_label, is_general = choose_data_source(df, "overview_data_source")

    if selected_df.empty:
        if is_general:
            st.warning("The baseline dataset is missing. Run `python generate_data.py`.")
        else:
            st.info("This manufacturer has no readable saved history yet. Upload records in Data import.")
    else:
        total = len(selected_df)
        delay_rate = float(pd.to_numeric(selected_df["is_delayed"], errors="coerce").mean()) if "is_delayed" in selected_df else 0.0
        avg_planned = float(pd.to_numeric(selected_df["planned_lead_time_days"], errors="coerce").mean()) if "planned_lead_time_days" in selected_df else 0.0
        high_critical = int(selected_df["criticality"].isin(["High", "Critical"]).sum()) if "criticality" in selected_df else 0

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(
                f'<div class="metric"><div class="metric-label">Shipment records</div><div class="metric-value">{total:,}</div><div class="metric-foot">{source_label}</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="metric"><div class="metric-label">Observed delay rate</div><div class="metric-value">{delay_rate:.1%}</div><div class="metric-foot">Historical labels in selected data</div></div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f'<div class="metric"><div class="metric-label">Mean planned lead time</div><div class="metric-value">{avg_planned:.1f} d</div><div class="metric-foot">Planned duration, not actual ETA</div></div>',
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f'<div class="metric"><div class="metric-label">High-criticality records</div><div class="metric-value">{high_critical:,}</div><div class="metric-foot">Marked High or Critical</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("### Risk landscape")
        left, right = st.columns(2)

        with left:
            make_chart(selected_df, "transport_mode", "Shipments by transport mode", "is_delayed")
        with right:
            make_chart(selected_df, "product_category", "Shipments by input category", "is_delayed")

        st.markdown("### Recent records")
        display_cols = [
            column for column in [
                "shipment_id", "supplier_id", "supplier_country", "transport_mode",
                "product_category", "planned_lead_time_days", "inventory_days",
                "djibouti_port_dwell_days", "modjo_dwell_days",
                "fx_availability_index", "criticality", "is_delayed",
            ]
            if column in selected_df.columns
        ]
        st.dataframe(selected_df[display_cols].head(12), use_container_width=True, hide_index=True)

        if is_general:
            st.caption("General data is synthetic demonstration data, not real Ethiopian supply-chain records.")
        else:
            st.caption("These summaries describe the selected manufacturer's saved history. They are not predictions.")


elif page == "New assessment":
    st.markdown("## Assess a shipment")
    st.write("Choose the model first, then enter the information available before the shipment arrives.")

    model_type = st.radio(
        "Prediction model",
        ["General AYEN Model", "Manufacturer-Specific Model"],
        horizontal=True,
        key="assessment_model_type",
    )

    selected_model = None
    selected_model_name = None
    selected_metrics = {}
    selected_slug = None
    model_ready = False

    if model_type == "General AYEN Model":
        if model is not None:
            selected_model = model
            selected_model_name = "General AYEN Model"
            selected_metrics = metrics
            model_ready = True
            st.success("The general baseline model will be used.")
        else:
            st.error("The general model is unavailable. Generate the baseline data and run `python train_model.py`.")
    else:
        trained_slugs = list_trained_manufacturers()

        if not trained_slugs:
            st.warning("No trained manufacturer models are available. Use Data import to upload history and train a model.")
        else:
            selected_slug = st.selectbox(
                "Select manufacturer",
                trained_slugs,
                format_func=display_manufacturer_name,
                key="assessment_manufacturer",
            )

            try:
                selected_model = load_manufacturer_model(selected_slug)

                if selected_model is None:
                    st.error("This manufacturer's model is missing. Retrain it in Data import.")
                else:
                    selected_model_name = f"{display_manufacturer_name(selected_slug)} model"
                    selected_metrics = load_manufacturer_metrics(selected_slug)
                    model_ready = True
                    st.success(f"Predictions will use {selected_model_name}.")
            except Exception as exc:
                st.error(f"The selected model could not be loaded: {exc}")

    st.markdown("---")

    if model_ready:
        with st.form("assessment_form"):
            st.markdown("#### Shipment profile")

            a, b, c = st.columns(3)

            with a:
                supplier_country = st.selectbox(
                    "Supplier country",
                    ["China", "India", "Turkey", "UAE", "Germany", "Italy", "Saudi Arabia", "Kenya", "Egypt", "United Arab Emirates"],
                )
                transport_mode = st.selectbox("Transport mode", ["Sea", "Air", "Road", "Rail"])
                product_category = st.selectbox(
                    "Input category",
                    ["Raw materials", "Raw Materials", "Packaging", "Chemicals", "Spare parts", "Spare Parts", "Electronics", "Food ingredients", "Food Ingredients", "Machinery", "Textiles"],
                )
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

            st.markdown("#### Ethiopian corridor signals")
            st.caption("Enter the latest available operational estimates. These are user-provided inputs, not live port or FX feeds.")

            d1, d2, d3 = st.columns(3)

            with d1:
                djibouti_port_dwell_days = st.number_input(
                    "Djibouti port dwell (days)",
                    min_value=0.0,
                    max_value=90.0,
                    value=6.0,
                    step=1.0,
                    help="Estimated time cargo has spent at Djibouti port.",
                )

            with d2:
                modjo_dwell_days = st.number_input(
                    "Modjo dry port dwell (days)",
                    min_value=0.0,
                    max_value=90.0,
                    value=3.0,
                    step=1.0,
                    help="Estimated time cargo has spent at Modjo dry port.",
                )

            with d3:
                fx_availability_index = st.slider(
                    "FX availability (0–1)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.50,
                    step=0.05,
                    help="0 means unavailable; 1 means readily available for the relevant payment.",
                )

            submitted = st.form_submit_button(
                "Assess disruption risk",
                type="primary",
                use_container_width=True,
            )

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
                "djibouti_port_dwell_days": float(djibouti_port_dwell_days),
                "modjo_dwell_days": float(modjo_dwell_days),
                "fx_availability_index": float(fx_availability_index),
                "supplier_country": supplier_country,
                "transport_mode": transport_mode,
                "product_category": product_category,
                "criticality": criticality,
            }

            model_features = FEATURES if model_type == "General AYEN Model" else MANUFACTURER_FEATURES
            input_df = pd.DataFrame(
                [{feature: row[feature] for feature in model_features}],
                columns=model_features,
            )

            if model_type == "General AYEN Model":
                reference_df = df
            else:
                reference_df = load_history(selected_slug)
                if reference_df is None or reference_df.empty:
                    reference_df = pd.DataFrame(columns=model_features)

            try:
                probability = float(selected_model.predict_proba(input_df)[0, 1])
                band, label = risk_band(probability)

                st.markdown("### Assessment result")
                st.info(f"**Model used:** {selected_model_name}")

                r1, r2, r3 = st.columns([1, 1, 2])

                with r1:
                    st.metric("Estimated delay probability", f"{probability:.0%}")

                with r2:
                    st.metric("Risk band", band)

                with r3:
                    st.markdown(f"**{label}**")
                    st.progress(probability)
                    st.caption("Model estimate based on the entered inputs, not a confirmed ETA.")

                if selected_metrics:
                    with st.expander("Selected model evaluation metrics"):
                        metric_cols = st.columns(4)

                        for col, key, label_text in zip(
                            metric_cols,
                            ["accuracy", "precision", "recall", "f1"],
                            ["Accuracy", "Precision", "Recall", "F1 score"],
                        ):
                            value = selected_metrics.get(key)
                            col.metric(label_text, f"{value:.3f}" if isinstance(value, (int, float)) else "N/A")

                st.markdown("### Top 3 model signals")
                st.caption(
                    "Each signal shows how much the model's estimate changes when that one input is replaced "
                    "with its typical value in the selected dataset. This is a sensitivity check, not proof of causation."
                )

                explanations = explain_prediction(
                    selected_model,
                    row,
                    model_features,
                    reference_df,
                )

                if explanations:
                    for index, item in enumerate(explanations, start=1):
                        difference = item["difference"]
                        if difference > 0.005:
                            direction = f"Increases the estimate by about {difference:.1%}"
                        elif difference < -0.005:
                            direction = f"Decreases the estimate by about {abs(difference):.1%}"
                        else:
                            direction = "Little change compared with its typical value"

                        with st.container(border=True):
                            st.markdown(f"**{index}. {feature_label(item['feature'])}**")
                            st.write(f"Entered value: **{format_feature_value(item['feature'], item['value'])}**")
                            st.caption(direction)
                else:
                    st.info("The selected model's input sensitivity could not be calculated.")

                st.markdown("### What this means operationally")

                if inventory_days <= planned_lead_time_days * 0.25:
                    st.warning("Inventory cover is low relative to planned lead time. Confirm stock, consumption, and replenishment timing.")
                elif criticality in ("High", "Critical"):
                    st.info("This input is marked production-critical. A contingency plan may be worthwhile even if the estimated risk is not high.")
                else:
                    st.info("Use the estimate as a prompt for verification. Confirm milestone dates with the supplier or carrier.")

                st.markdown("### Recommended next steps")

                for index, (tag, action) in enumerate(recommendations(row, probability), start=1):
                    with st.container(border=True):
                        st.markdown(f"**{index}. {tag}**")
                        st.write(action)

                likely_delay_days = max(1, round(planned_lead_time_days * 0.15 + 3))
                st.caption(
                    f"Planning note: if a delay occurs, a scenario of about {likely_delay_days} extra days can be discussed. "
                    "This is a rough assumption, not a model prediction of delay duration."
                )

            except Exception as exc:
                st.error(
                    "The selected model could not generate a prediction. No other model was substituted. "
                    f"Details: {exc}"
                )


elif page == "Shipment explorer":
    st.markdown("## Shipment explorer")
    st.write("Choose which shipment history you want to inspect.")

    selected_df, source_label, is_general = choose_data_source(df, "explorer_data_source")

    if selected_df.empty:
        st.info("No records are available for this data source yet.")
    else:
        filt1, filt2, filt3 = st.columns(3)

        with filt1:
            modes = sorted(selected_df["transport_mode"].dropna().unique().tolist()) if "transport_mode" in selected_df else []
            selected_modes = st.multiselect("Transport mode", modes, default=modes)

        with filt2:
            crits = sorted(selected_df["criticality"].dropna().unique().tolist()) if "criticality" in selected_df else []
            selected_crits = st.multiselect("Criticality", crits, default=crits)

        with filt3:
            delay_choice = st.selectbox("Delay status", ["All", "Delayed", "Not delayed"])

        view = selected_df.copy()

        if "transport_mode" in view:
            view = view[view["transport_mode"].isin(selected_modes)]

        if "criticality" in view:
            view = view[view["criticality"].isin(selected_crits)]

        if delay_choice == "Delayed" and "is_delayed" in view:
            view = view[view["is_delayed"] == 1]
        elif delay_choice == "Not delayed" and "is_delayed" in view:
            view = view[view["is_delayed"] == 0]

        st.caption(f"{len(view):,} records from {source_label} match the selected filters.")
        st.dataframe(view, use_container_width=True, hide_index=True)

        st.download_button(
            "Download filtered records",
            data=view.to_csv(index=False).encode("utf-8"),
            file_name="ayen_filtered_shipments.csv",
            mime="text/csv",
        )

        if is_general:
            st.caption("The general dataset is synthetic demonstration data.")
        else:
            st.caption("Manufacturer records are loaded from the saved local history for this demo app.")


elif page == "Data import":
    st.markdown("## Manufacturer data & model training")
    st.write("Create a manufacturer profile, save historical shipment records, and train a model specifically for that manufacturer's data.")
    st.warning("Demo limitation: uploaded data and models are stored in the app's local filesystem. This is not secure, persistent multi-company storage for production use.")

    manufacturers = list_manufacturers()
    profile_options = ["__new__"] + manufacturers

    selected_profile = st.selectbox(
        "Manufacturer profile",
        profile_options,
        key="import_manufacturer_profile",
        format_func=lambda value: "Create a new profile" if value == "__new__" else display_manufacturer_name(value),
    )

    if selected_profile == "__new__":
        manufacturer_name = st.text_input(
            "New manufacturer name",
            placeholder="e.g. Addis Manufacturing",
            key="new_manufacturer_name",
        )
    else:
        manufacturer_name = selected_profile
        st.caption(f"Selected profile: {display_manufacturer_name(selected_profile)}")

    st.markdown("### Upload historical shipment data")
    st.info(
        "CSV must contain the manufacturer model's required input features plus `is_delayed`. "
        "Use 1 for a known delayed shipment and 0 for a known not-delayed shipment. "
        "Do not label shipments still in transit as on-time."
    )

    with st.expander("Required CSV columns"):
        st.code(", ".join(MANUFACTURER_REQUIRED_COLUMNS))

    uploaded = st.file_uploader(
        "Choose a historical shipment CSV",
        type=["csv"],
        key="manufacturer_csv_upload",
    )

    if uploaded is not None:
        try:
            raw_df = pd.read_csv(uploaded)
            st.markdown("#### File preview")
            st.dataframe(raw_df.head(10), use_container_width=True, hide_index=True)

            try:
                validated_df = validate_history(raw_df)
                delay_count = int(validated_df["is_delayed"].sum())
                on_time_count = int((validated_df["is_delayed"] == 0).sum())

                a, b, c = st.columns(3)
                a.metric("Historical shipments", f"{len(validated_df):,}")
                b.metric("Delayed", f"{delay_count:,}")
                c.metric("Not delayed", f"{on_time_count:,}")

                st.metric("Historical delay rate", f"{validated_df['is_delayed'].mean():.1%}")

                if st.button("Save manufacturer history", type="primary", use_container_width=True):
                    if not str(manufacturer_name).strip() or manufacturer_name == "__new__":
                        st.error("Enter a manufacturer name before saving.")
                    else:
                        saved_path = save_history(manufacturer_name, validated_df)
                        st.success(f"Saved {len(validated_df):,} records for {manufacturer_name}.")
                        st.caption(f"Saved locally to `{saved_path.relative_to(ROOT)}`")
                        st.info("The profile will be available in the selectors. Train its model in the section below or after this page refreshes.")

            except ValueError as exc:
                st.error(f"Data validation failed: {exc}")

        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")

    st.markdown("---")
    st.markdown("### Train a manufacturer-specific model")

    manufacturers = list_manufacturers()

    if not manufacturers:
        st.info("No saved manufacturer histories yet. Upload and save a CSV above.")
    else:
        train_slug = st.selectbox(
            "Manufacturer to train",
            manufacturers,
            format_func=display_manufacturer_name,
            key="training_manufacturer_profile",
        )

        saved_history = load_history(train_slug)

        if saved_history is None:
            st.error("The saved history could not be loaded.")
        else:
            st.write(f"Saved historical records: **{len(saved_history):,}**")
            class_counts = saved_history["is_delayed"].value_counts().to_dict() if "is_delayed" in saved_history else {}
            st.caption(f"Outcome counts (0 = not delayed, 1 = delayed): {class_counts}")

            if st.button("Train and evaluate model", type="primary", use_container_width=True):
                try:
                    result = train_manufacturer_model(train_slug, saved_history)
                    st.success(f"Successfully trained and saved the model for {display_manufacturer_name(train_slug)}.")
                    show_model_metrics(result, "Held-out evaluation results")
                    st.metric("Training records", result["training_rows"])
                    st.caption("Metrics use a held-out split of uploaded history. They do not guarantee future performance.")
                except (ValueError, OSError) as exc:
                    st.error(f"Model training was not completed: {exc}")


elif page == "Model transparency":
    st.markdown("## Model transparency")
    st.write("Inspect the evaluation metrics and training context for the general model or a trained manufacturer-specific model.")

    selected_type = st.radio(
        "Model to inspect",
        ["General AYEN Model", "Manufacturer-Specific Model"],
        horizontal=True,
        key="transparency_model_type",
    )

    if selected_type == "General AYEN Model":
        st.warning("The baseline model is trained on synthetic demonstration data. Its scores do not establish accuracy on real shipments.")
        show_model_metrics(metrics, "Baseline held-out test results", baseline=True)

        if metrics:
            st.caption(f"Dataset type: {metrics.get('dataset_type', 'not specified')}; threshold: {metrics.get('threshold', 'N/A')}")

    else:
        trained_slugs = list_trained_manufacturers()

        if not trained_slugs:
            st.info("No trained manufacturer-specific models are available. Upload historical data and train a model in Data import.")
        else:
            selected_slug = st.selectbox(
                "Select manufacturer model",
                trained_slugs,
                format_func=display_manufacturer_name,
                key="transparency_manufacturer",
            )

            manufacturer_metrics = load_manufacturer_metrics(selected_slug)
            st.markdown(f"### {display_manufacturer_name(selected_slug)} model")

            if manufacturer_metrics:
                show_model_metrics(manufacturer_metrics, "Held-out manufacturer-history results")

                a, b, c = st.columns(3)
                total_rows = manufacturer_metrics.get("total_rows")
                training_rows = manufacturer_metrics.get("training_rows")
                delay_rate = manufacturer_metrics.get("delay_rate")

                a.metric("Total historical records", f"{total_rows:,}" if isinstance(total_rows, int) else "N/A")
                b.metric("Training records", f"{training_rows:,}" if isinstance(training_rows, int) else "N/A")
                c.metric("Historical delay rate", f"{delay_rate:.1%}" if isinstance(delay_rate, (int, float)) else "N/A")

                st.caption("Evaluation is based on a held-out split of uploaded records. Small or unrepresentative histories may produce unstable metrics.")
            else:
                st.warning("The model file exists, but its metrics file is missing. Retrain this model in Data import.")

    st.markdown("### General model input signals")
    st.write(", ".join(FEATURES))

    st.markdown("### Manufacturer model input signals")
    st.write(", ".join(MANUFACTURER_FEATURES))

    st.markdown("### Important limitations")
    st.write(
        "- Synthetic demonstration results are not real-world validation.\n"
        "- Djibouti, Modjo, and FX values are manually entered or simulated inputs, not live feeds.\n"
        "- The top-three explanations are single-input sensitivity checks, not causal proof.\n"
        "- No validated prediction of delay duration or financial loss.\n"
        "- Uploaded histories and models use local filesystem storage in this MVP; authentication, secure tenant isolation, and durable shared storage are not implemented."
    )

    st.markdown("### Label definition")
    st.code("is_delayed = 1 if actual_lead_time_days > planned_lead_time_days else 0")

