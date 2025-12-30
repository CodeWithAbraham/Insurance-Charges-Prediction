import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ----------------- Load Models -----------------
full_model = pickle.load(open('full_model.pkl', 'rb'))
poly_model = pickle.load(open('poly_model.pkl', 'rb'))
poly_transformer = pickle.load(open('poly_transformer.pkl', 'rb'))

# ----------------- Streamlit App -----------------
st.title("Insurance Charges Prediction")
st.write("Enter your details to predict insurance charges.")

# ----------------- User Inputs -----------------
st.subheader("Basic Information")
age = st.number_input("Age", min_value=18, max_value=100, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# ----------------- Prepare Input Data -----------------
data = pd.DataFrame({
    'age': [age],
    'bmi': [bmi],
    'children': [children],
    'sex_male': [1 if sex == "male" else 0],
    'smoker_yes': [1 if smoker == "yes" else 0],
    'region_northwest': [1 if region == "northwest" else 0],
    'region_southeast': [1 if region == "southeast" else 0],
    'region_southwest': [1 if region == "southwest" else 0],
})

# ----------------- Feature Lists -----------------
full_features = [
    'age','bmi','children','sex_male','smoker_yes',
    'region_northwest','region_southeast','region_southwest'
]

numeric_features = ['age','bmi','children']
categorical_features = [
    'sex_male','smoker_yes','region_northwest','region_southeast','region_southwest'
]

# ----------------- Full Model Prediction -----------------
full_pred = full_model.predict(data[full_features])

# ----------------- Polynomial Model Prediction -----------------
X_numeric_poly = poly_transformer.transform(data[numeric_features])
X_poly_final = np.hstack([X_numeric_poly, data[categorical_features].values])
poly_pred = poly_model.predict(X_poly_final)

# ----------------- Show Predictions -----------------
st.subheader("Predicted Insurance Charges")
st.success(f"Full Linear Regression: ${full_pred[0]:.2f}")
st.success(f"Polynomial Regression: ${poly_pred[0]:.2f}") 

st.subheader("Future Features")
st.info("""
        I will be working on adding more features to improve prediction accuracy, such as:
- Detailed smoking history
- Physical activity level
- Dietary habits
- Other health indicators
Stay tuned for upcoming updates!
""")
