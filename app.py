import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Employee Salary Prediction")

# Load the trained model
model = joblib.load("model.pkl")

st.header("Input Employee Information")

# Collect 5 input features from the user
experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=1)
education = st.slider("Education Level (Numeric)", 0, 20, 10)
role = st.selectbox("Job Role", ["Software Engineer", "Data Scientist", "Manager", "HR"])
age = st.number_input("Age", min_value=18, max_value=70, value=25)
hours = st.slider("Hours Worked per Week", 1, 80, 40)

# Encode categorical input
role_map = {
    "Software Engineer": 0,
    "Data Scientist": 1,
    "Manager": 2,
    "HR": 3
}
role_encoded = role_map[role]

# Create feature array with 5 inputs
features = np.array([[experience, education, role_encoded, age, hours]])

# Predict salary
if st.button("Predict Salary"):
    try:
        prediction = model.predict(features)
        st.success(f"Predicted Salary: ₹{prediction[0]:,.2f}")
    except ValueError as e:
        st.error(f"Prediction failed: {e}")
