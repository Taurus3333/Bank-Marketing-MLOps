# app_streamlit.py
from __future__ import annotations
import pandas as pd
import streamlit as st
from pathlib import Path
from catboost import CatBoostClassifier # type: ignore
from bank_marketing.pipelines.transform import feature_engineer
from bank_marketing.services.inference import CAT_FEATURE_NAMES
from bank_marketing.config import MODEL_PATH, PRED_THRESHOLD

st.set_page_config(page_title="Bank Marketing Prediction", layout="wide")

# --- Load model ---
@st.cache_resource
def load_model(path: str):
    m = CatBoostClassifier()
    m.load_model(path)
    return m

model = load_model(MODEL_PATH)

# --- Hero Header ---
st.markdown(
    """
    <div style="background-color:#003366;padding:20px;border-radius:10px;margin-bottom:20px;">
        <h1 style="color:white;text-align:center;">🏦 Bank Marketing — Term Deposit Prediction</h1>
        <p style="color:white;text-align:center;">
        Estimate if a client will subscribe to a term deposit.  
        Powered by <b>CatBoost</b> with feature engineering and class imbalance handling.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Input Form ---
st.markdown("### 📋 Client Information")
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=40)
        balance = st.number_input("Account Balance (€)", value=500)
        campaign = st.number_input("Campaign Contacts (#)", min_value=0, value=2)
        pdays = st.number_input("Days Since Last Contact (−1 = never)", value=-1)

    with col2:
        previous = st.number_input("Previous Contacts (#)", min_value=0, value=0)
        day = st.number_input("Day of Month", min_value=1, max_value=31, value=5)
        default = st.radio("Default on Credit?", [0, 1], horizontal=True)
        housing = st.radio("Housing Loan?", [0, 1], horizontal=True)

    with col3:
        loan = st.radio("Personal Loan?", [0, 1], horizontal=True)
        job = st.selectbox("Job", [
            "admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
            "blue-collar","self-employed","retired","technician","services"
        ], index=3)
        marital = st.selectbox("Marital Status", ["married","divorced","single"], index=0)
        education = st.selectbox("Education", ["unknown","secondary","primary","tertiary"], index=1)

    contact = st.selectbox("Contact Method", ["unknown","telephone","cellular"], index=2)
    month = st.selectbox("Last Contact Month", [
        "jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"
    ], index=4)
    poutcome = st.selectbox("Previous Campaign Outcome", ["unknown","other","failure","success"], index=0)

    submitted = st.form_submit_button("🔮 Predict Likelihood")

# --- Prediction ---
if submitted:
    row = {
        "age": age, "balance": balance, "campaign": campaign, "pdays": pdays, "previous": previous,
        "day": day, "default": default, "housing": housing, "loan": loan,
        "job": job, "marital": marital, "education": education,
        "contact": contact, "month": month, "poutcome": poutcome,
    }
    df = pd.DataFrame([row])
    fe = feature_engineer(df).drop(columns=[c for c in ["y"] if c in df.columns])

    # Predict
    prob = float(model.predict_proba(fe)[:, 1][0])
    decision = int(prob >= PRED_THRESHOLD)

    # --- Display Prediction ---
    st.markdown("### 🧾 Prediction Result")
    colA, colB = st.columns([1, 2])

    with colA:
        st.metric("Subscription Probability", f"{prob:.3f}")
        st.metric("Decision", "✅ Subscribe" if decision else "❌ Do not subscribe")

    with colB:
        st.progress(prob)
        if decision:
            st.success("This client is **likely to subscribe** a term deposit.")
        else:
            st.warning("This client is **unlikely to subscribe** a term deposit.")

    # --- Explainability ---
    with st.expander("🔍 Show Engineered Features Used for Prediction"):
        st.dataframe(fe.T, use_container_width=True)
