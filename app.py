import streamlit as st
import numpy as np
import joblib

# ---------------------------
# Load model and scaler
# ---------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("💳 Bank Customer Churn Prediction System")
st.write("Enter customer details to predict whether they will exit the bank.")

# ---------------------------
# Input Fields
# ---------------------------
CreditScore = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
Age = st.number_input("Age", min_value=18, max_value=100, value=40)
Tenure = st.number_input("Tenure (years with bank)", min_value=0, max_value=10, value=3)
Balance = st.number_input("Account Balance", value=50000.0)
NumOfProducts = st.selectbox("Number of Bank Products", [1, 2, 3, 4])
HasCrCard = st.selectbox("Has Credit Card?", [0, 1])
IsActiveMember = st.selectbox("Is Active Member?", [0, 1])
EstimatedSalary = st.number_input("Estimated Salary", value=60000.0)

# Gender and country inputs
Gender = st.selectbox("Gender", ["Male", "Female"])
Country = st.selectbox("Country", ["France", "Germany", "Spain"])

# ---------------------------
# Encoding Categorical Values
# ---------------------------
France = 1 if Country == "France" else 0
Germany = 1 if Country == "Germany" else 0
Spain = 1 if Country == "Spain" else 0

Female = 1 if Gender == "Female" else 0
Male = 1 if Gender == "Male" else 0

# ---------------------------
# Feature Engineering (Match Training Columns)
# ---------------------------
Mem_no_Products = NumOfProducts * 2  
Cred_Bal_Sal = CreditScore + Balance + EstimatedSalary
Bal_sal = Balance / (EstimatedSalary + 1)
Tenure_Age = Tenure * Age
Age_Tenure_product = Age * Tenure * NumOfProducts

# TF-IDF Surname placeholder values
surname_tfidf = [0, 0, 0, 0, 0]

# ---------------------------
# Final Feature Vector (ORDER MUST MATCH TRAINING)
# ---------------------------
features = np.array([[
    CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard,
    IsActiveMember, EstimatedSalary,
    Mem_no_Products, Cred_Bal_Sal, Bal_sal, Tenure_Age, Age_Tenure_product,
    *surname_tfidf,
    France, Germany, Spain, Female, Male
]])

# ---------------------------
# Prediction Button
# ---------------------------
if st.button("🔍 Predict Churn"):
    
    # Scale input
    scaled_features = scaler.transform(features)

    # Model Prediction
    result = model.predict(scaled_features)

    if result == 1:
        st.error("🔴 **Prediction: Customer is LIKELY TO LEAVE (Churn)** ❌")
    else:
        st.success("🟢 **Prediction: Customer Will Stay** ✅")
