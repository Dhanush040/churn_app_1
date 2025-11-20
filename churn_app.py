import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import matplotlib.pyplot as plt

# -------------------- Load Model --------------------
MODEL_FILE = 'random_forest_model.joblib'
model = load(MODEL_FILE)

# -------------------- App UI --------------------
st.title("Telco Customer Churn Prediction App")
st.write("Enter customer details below to predict whether the customer will churn.")

# -------------------- Input Fields --------------------
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)

InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0, step=0.1)
TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=100000.0, value=500.0, step=0.1)

SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
Partner = st.selectbox("Partner", ["No", "Yes"])
Dependents = st.selectbox("Dependents", ["No", "Yes"])
PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])

# -------------------- Encoding (manual mapping) --------------------
mapping_yes_no = {"No": 0, "Yes": 1}
InternetService_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
Contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}

input_data = pd.DataFrame({
    'tenure': [tenure],
    'InternetService': [InternetService_map[InternetService]],
    'Contract': [Contract_map[Contract]],
    'MonthlyCharges': [MonthlyCharges],
    'TotalCharges': [TotalCharges],
    'SeniorCitizen': [mapping_yes_no[SeniorCitizen]],
    'Partner': [mapping_yes_no[Partner]],
    'Dependents': [mapping_yes_no[Dependents]],
    'PhoneService': [mapping_yes_no[PhoneService]],
    'PaperlessBilling': [mapping_yes_no[PaperlessBilling]]
})

st.markdown("**Input summary**")
st.write(input_data.T)

# -------------------- Prediction + Visualization --------------------
if st.button("Predict Churn"):
    # Prediction (handle models without predict_proba)
    try:
        proba = model.predict_proba(input_data)[0]
        prob_churn = float(proba[1])  # probability of churn = class 1
    except Exception:
        # fallback: use decision_function or binary predict
        try:
            # decision_function -> convert with sigmoid approx (not exact)
            df_val = model.decision_function(input_data)[0]
            prob_churn = 1 / (1 + np.exp(-df_val))
        except Exception:
            prediction = model.predict(input_data)[0]
            prob_churn = 1.0 if prediction == 1 else 0.0

    # If we didn't compute prediction already, get it
    try:
        prediction = model.predict(input_data)[0]
    except Exception:
        # fallback to thresholding probability
        prediction = 1 if prob_churn >= 0.5 else 0

    # Display textual result
    if prediction == 1:
        st.error(f"Customer is **likely to churn** (Probability: {prob_churn:.2f})")
    else:
        st.success(f"Customer is **not likely to churn** (Probability: {prob_churn:.2f})")

    # -------------------- Matplotlib Visualization --------------------
    # 1) Probability bar (churn vs no-churn)
    labels = ['Not churn', 'Churn']
    probs = [1 - prob_churn, prob_churn]

    fig1, ax1 = plt.subplots(figsize=(6, 3))
    bars = ax1.bar(labels, probs)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Probability')
    ax1.set_title('Churn Probability')
    # Annotate bar values
    for bar, p in zip(bars, probs):
        height = bar.get_height()
        ax1.annotate(f'{p:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 6), textcoords="offset points", ha='center', va='bottom')
    st.pyplot(fig1)

    # 2) Optional: feature importances (if available)
    if hasattr(model, "feature_importances_"):
        try:
            feat_imps = model.feature_importances_
            feature_names = list(input_data.columns)  # ensure same order used during training
            # If model was trained with more features than we pass, attempt to plot only those present
            if len(feat_imps) == len(feature_names):
                fi_df = pd.DataFrame({'feature': feature_names, 'importance': feat_imps})
            else:
                # try to pair first N feature_importances with our features (best-effort)
                fi_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': feat_imps[:len(feature_names)]
                })
            fi_df = fi_df.sort_values('importance', ascending=True)

            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.barh(fi_df['feature'], fi_df['importance'])
            ax2.set_xlabel('Importance')
            ax2.set_title('Feature Importances (model-provided)')
            st.pyplot(fig2)
        except Exception as e:
            st.info("Feature importances exist but couldn't be plotted: " + str(e))
    else:
        st.info("Model does not expose `feature_importances_`. Skipping importance plot.")
