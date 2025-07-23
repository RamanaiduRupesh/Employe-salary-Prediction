import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Employee Salary Prediction")

model = joblib.load("model.pkl")

st.header("Input Employee Information")

experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=1)
education = st.slider("Education Level (Numeric)", 0, 20, 10)

features = np.array([[experience, education]])

if st.button("Predict Salary"):
    prediction = model.predict(features)
    st.success(f"Predicted Salary: ₹{prediction[0]:,.2f}")
