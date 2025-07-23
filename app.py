import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Employee Salary Prediction")

model = joblib.load("model.pkl")

st.header("Input Employee Information")

experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=1)
education = st.slider("Education Level (Numeric)", 0, 20, 10)
role = st.selectbox("Job Role", ["Software Engineer", "Data Scientist", "Manager", "HR"])

# Encode role
role_map = {
    "Software Engineer": 0,
    "Data Scientist": 1,
    "Manager": 2,
    "HR": 3
}
role_encoded = role_map[role]

# Match input shape to training
features = np.array([[experience, education, role_encoded]])

if st.button("Predict Salary"):
    try:
        prediction = model.predict(features)
        st.success(f"Predicted Salary: ₹{prediction[0]:,.2f}")
    except ValueError as e:
        st.error(f"Prediction failed: {e}")
