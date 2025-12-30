import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D
import pickle

# ----------------- Load dataset -----------------
insurance_data = pd.read_csv('insurance.csv')

# ----------------- One-Hot Encode categorical features -----------------
categorical_features = ['sex', 'smoker', 'region']
insurance_data = pd.get_dummies(insurance_data, columns=categorical_features, drop_first=True)

# ----------------- Remove outliers using IQR -----------------
Q1 = insurance_data['charges'].quantile(0.25)
Q3 = insurance_data['charges'].quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
insurance_data = insurance_data[(insurance_data['charges'] >= lower_limit) & 
                                (insurance_data['charges'] <= upper_limit)]

# ----------------- Target variable -----------------
y_charges = insurance_data['charges']

# ----------------- Simple Linear Regression (BMI) -----------------
X_bmi = insurance_data[['bmi']]
bmi_model = LinearRegression()
bmi_model.fit(X_bmi, y_charges)
bmi_pred = bmi_model.predict(X_bmi)

# ----------------- 2-Feature Multivariate Regression (Age + BMI) -----------------
X_age_bmi = insurance_data[['age', 'bmi']]
age_bmi_model = LinearRegression()
age_bmi_model.fit(X_age_bmi, y_charges)
age_bmi_pred = age_bmi_model.predict(X_age_bmi)

# ----------------- Full Multivariate Regression (All numeric features) -----------------
X_all_features = insurance_data.drop(columns=['charges'])  # Now all numeric
full_model = LinearRegression()
full_model.fit(X_all_features, y_charges)
full_pred = full_model.predict(X_all_features)

# ----------------- Polynomial Regression -----------------
numeric_features = ['age', 'bmi', 'children']
X_numeric = insurance_data[numeric_features]
X_categorical = insurance_data.drop(columns=numeric_features + ['charges'])

poly_transformer = PolynomialFeatures(degree=2)
X_numeric_poly = poly_transformer.fit_transform(X_numeric)
X_poly_final = np.hstack([X_numeric_poly, X_categorical.values])

poly_model = LinearRegression()
poly_model.fit(X_poly_final, y_charges)
poly_pred = poly_model.predict(X_poly_final)

# ----------------- Save Models -----------------
with open('full_model.pkl', 'wb') as f:
    pickle.dump(full_model, f)

with open('poly_model.pkl', 'wb') as f:
    pickle.dump(poly_model, f)

with open('poly_transformer.pkl', 'wb') as f:
    pickle.dump(poly_transformer, f)

print("Models saved successfully! ✅")

# ----------------- Evaluation Function -----------------
def evaluate_model(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n{name} Metrics:")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAE     : {mae:.2f}")
    print(f"RMSE    : {rmse:.2f}")
    return r2, mae, rmse

r2_bmi, mae_bmi, rmse_bmi = evaluate_model("Simple LR", y_charges, bmi_pred)
r2_age_bmi, mae_age_bmi, rmse_age_bmi = evaluate_model("2-Feature LR", y_charges, age_bmi_pred)
r2_full, mae_full, rmse_full = evaluate_model("Full LR", y_charges, full_pred)
r2_poly, mae_poly, rmse_poly = evaluate_model("Polynomial LR", y_charges, poly_pred)

# ----------------- Results Summary -----------------
results_df = pd.DataFrame({
    "Model": ["Simple LR", "2-Feature LR", "Full LR", "Polynomial LR"],
    "R2 Score": [r2_bmi, r2_age_bmi, r2_full, r2_poly],
    "MAE": [mae_bmi, mae_age_bmi, mae_full, mae_poly],
    "RMSE": [rmse_bmi, rmse_age_bmi, rmse_full, rmse_poly]
})

print("\nModel Comparison (R2, MAE, RMSE):")
print(results_df)
