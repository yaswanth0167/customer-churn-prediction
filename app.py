import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load trained model (XGBoost)
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction App")
st.write("Enter customer details to predict churn")

# ---- INPUTS ---- #
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=500.0)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
security = st.selectbox("Online Security", ["Yes", "No"])
support = st.selectbox("Tech Support", ["Yes", "No"])

# ---- PREPROCESS INPUT ---- #
def preprocess():
    data = [tenure, monthly_charges, total_charges]

    # Contract encoding
    data += [
        1 if contract == "One year" else 0,
        1 if contract == "Two year" else 0
    ]

    # Internet encoding
    data += [
        1 if internet == "Fiber optic" else 0,
        1 if internet == "No" else 0
    ]

    # Binary features
    data.append(1 if security == "Yes" else 0)
    data.append(1 if support == "Yes" else 0)

    feature_names = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract_One year', 'Contract_Two year', 'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_Yes', 'TechSupport_Yes']
    return pd.DataFrame([data], columns=feature_names)

# ---- PREDICTION ---- #
if st.button("Predict"):

    input_data = preprocess()
    prediction = model.predict(input_data)

    st.subheader("Result:")

    if prediction[0] == 1:
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer will STAY")