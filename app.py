import streamlit as st
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("💳 Bank Customer Churn Prediction")

st.write("Enter customer details below to predict whether the customer will churn.")

# User Inputs
CreditScore = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
Age = st.number_input("Age", min_value=18, max_value=100, value=30)
Tenure = st.number_input("Tenure (Years with bank)", min_value=0, max_value=20, value=3)
Balance = st.number_input("Balance", min_value=0, value=50000)
NumOfProducts = st.selectbox("Number of Products", [1, 2, 3, 4])
HasCrCard = st.selectbox("Has Credit Card", [0, 1])
IsActiveMember = st.selectbox("Is Active Member?", [0, 1])
EstimatedSalary = st.number_input("Estimated Salary", min_value=0, value=50000)

# Country
Geography = st.selectbox("Country", ["France", "Germany", "Spain"])
France = 1 if Geography == "France" else 0
Germany = 1 if Geography == "Germany" else 0
Spain = 1 if Geography == "Spain" else 0

# Gender
Gender = st.selectbox("Gender", ["Male", "Female"])
Male = 1 if Gender == "Male" else 0
Female = 1 if Gender == "Female" else 0

# Prepare input
user_data = np.array([[CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard,
                       IsActiveMember, EstimatedSalary, France, Germany, Spain, Female, Male]])

# Scale Data
user_data_scaled = scaler.transform(user_data)

if st.button("Predict"):
    prediction = model.predict(user_data_scaled)
    
    if prediction == 1:
        st.error("🔴 The customer is likely to **CHURN** ❌")
    else:
        st.success("🟢 The customer is likely to **STAY** ✅")
