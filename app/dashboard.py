import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from api.main import (
    TransactionInput,
    check_drift,
    compute_shap_values,
    health_check,
    predict_fraud,
)
from src.data.generate_data import FEATURES
from src.monitoring.drift import evaluate_data_drift

st.set_page_config(
    page_title="FraudLens | FinTech Risk Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom Enterprise FinTech Dark-Neon CSS ───
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #131129 0%, #090d16 55%, #05070e 100%);
        color: #f8fafc;
    }
    .fintech-banner {
        background: linear-gradient(135deg, rgba(19,17,41,0.95) 0%, rgba(46,16,101,0.82) 100%);
        border: 1px solid rgba(168, 85, 247, 0.35);
        border-left: 5px solid #a855f7;
        border-radius: 14px;
        padding: 22px 28px;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
    }
    .fintech-title {
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(90deg, #c084fc, #38bdf8, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .fintech-sub {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 6px;
    }
    .kpi-card {
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-top: 3px solid #a855f7;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.3);
    }
    .kpi-label {
        color: #94a3b8;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }
    .kpi-value {
        color: #f8fafc;
        font-size: 1.3rem;
        font-weight: 800;
        margin-top: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .kpi-sub {
        color: #34d399;
        font-size: 0.78rem;
        margin-top: 2px;
    }
    .panel-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 4px;
    }
    .panel-caption {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-bottom: 10px;
    }
    div[data-testid="stTabs"] button {
        font-weight: 700 !important;
        font-size: 0.95rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Auto-bootstrap data and model on cloud if missing ───
if not os.path.exists("models/fraud_pipeline.joblib") or not os.path.exists(
    "data/raw/transactions_reference.csv"
):
    with st.spinner("Initializing financial transaction dataset and training fraud models..."):
        from src.data.generate_data import generate_fraud_transactions
        from src.models.train import train_and_evaluate

        os.makedirs("data/raw", exist_ok=True)
        generate_fraud_transactions(n_samples=8000, drift=False, seed=42).to_csv(
            "data/raw/transactions_reference.csv", index=False
        )
        generate_fraud_transactions(n_samples=2000, drift=True, seed=99).to_csv(
            "data/raw/transactions_drifted.csv", index=False
        )
        train_and_evaluate()

with open("models/metadata.json", "r") as f:
    meta = json.load(f)
pipeline = joblib.load("models/fraud_pipeline.joblib")
winner_name = meta["winning_model"]
winner_stats = meta["leaderboard"][winner_name]

# ─── Sidebar Command Panel ───
with st.sidebar:
    st.markdown("### 💳 FraudLens Risk Engine")
    st.caption("Explainable Payment Fraud & MLOps")
    st.divider()
    st.markdown("**🟢 Engine Status:** `ONLINE`")
    st.markdown(f"**🏆 Champion Model:** `{winner_name}`")
    st.markdown(f"**🎯 Decision Threshold:** `{meta['optimal_threshold']}`")
    st.markdown(f"**📈 Validation PR-AUC:** `{winner_stats['pr_auc']:.4f}`")
    st.divider()
    st.markdown("#### 🔬 Monitored Features (8)")
    for f_name in FEATURES:
        st.markdown(f"- `{f_name}`")
    st.divider()
    st.caption("Built by Aditya Mitra • KIIT CSE")

# ─── Top Hero Banner ───
st.markdown(
    """
    <div class="fintech-banner">
        <div class="fintech-title">💳 FraudLens — Explainable Transaction Fraud & Risk Command Center</div>
        <div class="fintech-sub">
            Real-Time Payment Risk Scoring • Precision-Recall Threshold Calibration • Local SHAP Feature Attribution • KS-Test Drift Monitoring
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Top KPI Row ───
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-label">Active Production Model</div>
            <div class="kpi-value">{winner_name}</div>
            <div class="kpi-sub">✔ Leak-Free Pipeline</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f"""<div class="kpi-card" style="border-top-color:#34d399;">
            <div class="kpi-label">Held-Out PR-AUC Score</div>
            <div class="kpi-value">{winner_stats['pr_auc']*100:.2f}%</div>
            <div class="kpi-sub">▲ High Minority-Class Precision</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        f"""<div class="kpi-card" style="border-top-color:#38bdf8;">
            <div class="kpi-label">F1-Score / Precision</div>
            <div class="kpi-value">{winner_stats['f1_score']*100:.2f}%</div>
            <div class="kpi-sub">Precision: {winner_stats['precision']*100:.2f}%</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        f"""<div class="kpi-card" style="border-top-color:#fbbf24;">
            <div class="kpi-label">Calibrated Threshold</div>
            <div class="kpi-value">{meta['optimal_threshold']:.4f}</div>
            <div class="kpi-sub">Recall: {winner_stats['recall']*100:.2f}%</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.write("")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔍 Live XAI Transaction Inspector",
        "📊 Model Leaderboard & Diagnostics",
        "📉 Data Drift & MLOps Telemetry",
        "📡 Interactive REST API Playground",
    ]
)

# ═══════════════════════════════════════════════════════════
# TAB 1: LIVE XAI TRANSACTION INSPECTOR (Zero Text Overlap)
# ═══════════════════════════════════════════════════════════
with tab1:
    st.markdown("#### ⚡ Quick Payment Scenario Presets")
    p_col1, p_col2, p_col3 = st.columns(3)

    if "fraud_preset_vals" not in st.session_state:
        st.session_state.fraud_preset_vals = [850.0, 240.0, 2.0, 0.92, 6.0, 1.0, 2.0, 12.0]

    if p_col1.button("🚨 Load High-Risk Account Takeover Fraud", use_container_width=True):
        st.session_state.fraud_preset_vals = [1420.0, 380.0, 3.0, 0.95, 7.0, 1.0, 3.0, 8.0]
    if p_col2.button("🕵️ Load Card-Testing Velocity Spike", use_container_width=True):
        st.session_state.fraud_preset_vals = [490.0, 145.0, 23.0, 0.84, 5.0, 1.0, 1.0, 19.0]
    if p_col3.button("✅ Load Normal Domestic Retail Purchase", use_container_width=True):
        st.session_state.fraud_preset_vals = [38.5, 4.2, 14.0, 0.08, 1.0, 0.0, 0.0, 640.0]

    with st.expander("🎛️ Fine-Tune Transaction Telemetry Features", expanded=True):
        cols = st.columns(4)
        user_input = {}
        for idx, feat in enumerate(FEATURES):
            user_input[feat] = cols[idx % 4].number_input(
                feat,
                value=float(st.session_state.fraud_preset_vals[idx]),
                key=f"tx_{feat}_{st.session_state.fraud_preset_vals[idx]}",
            )

    df_in = pd.DataFrame([user_input])[FEATURES]
    prob = float(pipeline.predict_proba(df_in)[0, 1])
    thresh = float(meta["optimal_threshold"])
    is_fraud = prob >= thresh

    X_trans = pipeline.named_steps["preprocessor"].transform(df_in)
    vals = compute_shap_values(pipeline.named_steps["classifier"], X_trans)

    st.write("")
    g_col, s_col = st.columns([1, 1.4], gap="large")

    with g_col:
        st.markdown(
            '<div class="panel-header">🎯 Real-Time Fraud Risk Decision</div>'
            '<div class="panel-caption">Calibrated fraud probability vs optimal PR threshold</div>',
            unsafe_allow_html=True,
        )
        if is_fraud:
            st.error(
                f"🚨 **FRAUD DETECTED** — Risk Score: **{prob:.2%}** (Threshold: `{thresh}`)"
            )
        else:
            st.success(
                f"✅ **LEGITIMATE TRANSACTION** — Risk Score: **{prob:.2%}** (Threshold: `{thresh}`)"
            )

        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font": {"color": "#f8fafc", "size": 34}},
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickwidth": 1,
                        "tickcolor": "#94a3b8",
                        "tickfont": {"size": 12, "color": "#cbd5e1"},
                    },
                    "bar": {"color": "#f43f5e" if is_fraud else "#10b981"},
                    "bgcolor": "#0f172a",
                    "borderwidth": 1,
                    "bordercolor": "#334155",
                    "steps": [
                        {"range": [0, thresh * 100], "color": "rgba(16, 185, 129, 0.18)"},
                        {"range": [thresh * 100, 100], "color": "rgba(244, 63, 94, 0.22)"},
                    ],
                    "threshold": {
                        "line": {"color": "#fbbf24", "width": 4},
                        "thickness": 0.8,
                        "value": thresh * 100,
                    },
                },
            )
        )
        fig_gauge.update_layout(
            height=290,
            margin=dict(l=38, r=38, t=25, b=15),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#f8fafc"},
        )
        st.plotly_chart(
            fig_gauge, use_container_width=True, config={"displayModeBar": False}
        )

    with s_col:
        st.markdown(
            '<div class="panel-header">🧠 Local SHAP Feature Attribution</div>'
            '<div class="panel-caption">Exact feature-level impact driving this transaction decision</div>',
            unsafe_allow_html=True,
        )
        shap_df = pd.DataFrame({"Feature": FEATURES, "SHAP_Impact": vals}).sort_values(
            by="SHAP_Impact", ascending=True
        )
        shap_df["Impact Direction"] = np.where(
            shap_df["SHAP_Impact"] >= 0,
            "Pushes toward Fraud (+)",
            "Pushes toward Legitimate (-)",
        )
        fig_shap = px.bar(
            shap_df,
            x="SHAP_Impact",
            y="Feature",
            orientation="h",
            color="Impact Direction",
            color_discrete_map={
                "Pushes toward Fraud (+)": "#f43f5e",
                "Pushes toward Legitimate (-)": "#38bdf8",
            },
        )
        fig_shap.update_layout(
            height=365,
            margin=dict(l=10, r=20, t=15, b=55),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.5)",
            font={"color": "#f8fafc", "size": 12},
            xaxis_title="SHAP Value (Impact on Fraud Log-Odds / Probability)",
            yaxis_title=None,
            legend=dict(
                title=None,
                orientation="h",
                yanchor="top",
                y=-0.18,
                xanchor="center",
                x=0.5,
                font=dict(size=12, color="#e2e8f0"),
            ),
        )
        st.plotly_chart(
            fig_shap, use_container_width=True, config={"displayModeBar": False}
        )

# ═══════════════════════════════════════════════════════════
# TAB 2: MODEL LEADERBOARD & DIAGNOSTICS
# ═══════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### 🏆 Multi-Model Fraud Benchmark Comparison")
    lb_df = pd.DataFrame(meta["leaderboard"]).T.drop(columns=["confusion_matrix"])
    st.dataframe(lb_df.style.highlight_max(axis=0, color="#065f46"), use_container_width=True)

    c_left, c_right = st.columns([1.2, 1], gap="large")
    with c_left:
        st.markdown(
            '<div class="panel-header">📊 PR-AUC, F1, Precision & Recall Comparison</div>',
            unsafe_allow_html=True,
        )
        plot_df = (
            lb_df[["pr_auc", "f1_score", "precision", "recall"]]
            .reset_index()
            .melt(id_vars="index", var_name="Metric", value_name="Score")
        )
        fig_comp = px.bar(
            plot_df,
            x="Metric",
            y="Score",
            color="index",
            barmode="group",
            color_discrete_sequence=["#c084fc", "#38bdf8", "#34d399"],
        )
        fig_comp.update_layout(
            yaxis_range=[0.0, 1.05],
            margin=dict(l=10, r=10, t=15, b=50),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.5)",
            font={"color": "#f8fafc"},
            legend=dict(
                title=None, orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5
            ),
            height=360,
        )
        st.plotly_chart(fig_comp, use_container_width=True, config={"displayModeBar": False})

    with c_right:
        st.markdown(
            f'<div class="panel-header">🎯 Confusion Matrix — {winner_name}</div>',
            unsafe_allow_html=True,
        )
        cm = np.array(winner_stats["confusion_matrix"])
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Purples",
            x=["Pred Legit (0)", "Pred Fraud (1)"],
            y=["Actual Legit (0)", "Actual Fraud (1)"],
        )
        fig_cm.update_layout(
            margin=dict(l=10, r=10, t=15, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#f8fafc"},
            height=360,
        )
        st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})

# ═══════════════════════════════════════════════════════════
# TAB 3: DATA DRIFT & MLOPS TELEMETRY
# ═══════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### 📉 Kolmogorov-Smirnov (`ks_2samp`) Transaction Drift Monitor")
    col_r1, col_r2 = st.columns([1, 1])
    batch_choice = col_r1.radio(
        "Select Incoming Transaction Batch:",
        ["Drifted Holiday/Attack Surge Batch", "Stable Reference Batch"],
        horizontal=True,
    )
    inspected_feat = col_r2.selectbox("Inspect Feature Distribution Overlay:", FEATURES)

    ref_df = pd.read_csv("data/raw/transactions_reference.csv")
    cur_df = pd.read_csv(
        "data/raw/transactions_drifted.csv"
        if "Drifted" in batch_choice
        else "data/raw/transactions_reference.csv"
    )
    report = evaluate_data_drift(ref_df, cur_df)

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Drifted Features Flagged",
        f"{report['drifted_feature_count']} / {report['total_features']}",
    )
    m2.metric("Covariate Drift Share", f"{report['drift_share']:.1%}")
    m3.metric(
        "Pipeline Retraining Trigger",
        "🚨 RETRAIN REQUIRED" if report["retrain_triggered"] else "✅ MODEL STABLE",
    )

    dist_df = pd.concat(
        [
            pd.DataFrame(
                {inspected_feat: ref_df[inspected_feat].dropna(), "Dataset": "Reference Baseline"}
            ),
            pd.DataFrame(
                {inspected_feat: cur_df[inspected_feat].dropna(), "Dataset": "Incoming Batch"}
            ),
        ]
    )
    fig_dist = px.histogram(
        dist_df,
        x=inspected_feat,
        color="Dataset",
        barmode="overlay",
        nbins=45,
        histnorm="probability density",
        color_discrete_map={"Reference Baseline": "#38bdf8", "Incoming Batch": "#f43f5e"},
    )
    fig_dist.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.5)",
        font={"color": "#f8fafc"},
        legend=dict(title=None, orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})

    st.dataframe(pd.DataFrame(report["features"]).T, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 4: INTERACTIVE REST API PLAYGROUND
# ═══════════════════════════════════════════════════════════
with tab4:
    st.markdown("#### 📡 FastAPI Endpoint Simulator (Pydantic Schema Validation)")
    endpoint = st.selectbox(
        "Select FastAPI Endpoint:",
        [
            "POST /predict (Transaction Fraud & SHAP Attribution)",
            "GET /health (Service & Model Readiness)",
            "GET /drift-status (KS-Test Drift Telemetry JSON)",
        ],
    )

    if endpoint.startswith("POST /predict"):
        default_payload = {
            "transaction_amount": 850.0,
            "distance_from_home": 240.0,
            "hour_of_day": 2,
            "device_risk_score": 0.92,
            "velocity_last_1h": 6,
            "is_international": 1,
            "previous_fraud_count": 2,
            "account_age_days": 12,
        }
        raw_json = st.text_area(
            "Request JSON Body:", value=json.dumps(default_payload, indent=2), height=230
        )
        if st.button("🚀 Execute POST /predict"):
            try:
                parsed = json.loads(raw_json)
                validated_input = TransactionInput(**parsed)
                response = predict_fraud(validated_input)
                st.code(json.dumps(response, indent=2), language="json")
            except Exception as e:
                st.error(f"422 Unprocessable Entity / Validation Error: {e}")
    elif endpoint.startswith("GET /health"):
        if st.button("🚀 Execute GET /health"):
            st.code(json.dumps(health_check(), indent=2), language="json")
    else:
        if st.button("🚀 Execute GET /drift-status"):
            st.code(json.dumps(check_drift(use_drifted_batch=True), indent=2), language="json")