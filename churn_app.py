import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# Load Model
MODEL_FILE = 'random_forest_model.joblib'
model = load(MODEL_FILE)

st.title("Telco Customer Churn Prediction App")
st.write("Enter customer details below to predict whether the customer will churn.")

# ------------------------- Input Fields -------------------------

tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)

InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=500.0)

SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
Partner = st.selectbox("Partner", ["No", "Yes"])
Dependents = st.selectbox("Dependents", ["No", "Yes"])
PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])

# ----------------------- Encoding Section -----------------------

# Manual encoding matching your training code
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

# ------------------------- Prediction -------------------------

if st.button("Predict Churn"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f" Customer is **likely to churn** (Probability: {probability:.2f})")
    else:
        st.success(f"Customer is **not likely to churn** (Probability: {probability:.2f})")

