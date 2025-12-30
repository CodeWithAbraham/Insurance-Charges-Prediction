# 🧾 Insurance Charges Prediction (ML + Streamlit Deployment)

An end-to-end Machine Learning project that predicts individual medical insurance charges based on demographic and health-related inputs. The project includes data preprocessing, regression model training and comparison, and deployment via a real-time interactive Streamlit web application.
---

## 📊 Dataset
- Medical insurance cost dataset containing features such as `age`, `bmi`, `children`, `sex`, `smoker`, and `region`
- Categorical variables encoded using **One-Hot Encoding**
- Outliers removed using the **Interquartile Range (IQR)** method for improved model robustness

---

## 🧠 Machine Learning Models
The following regression models were implemented and evaluated:

| Model Type | Purpose |
|----------|---------|
| Simple Linear Regression | Analyze impact of BMI on insurance charges |
| 2-Feature Linear Regression | Capture combined effect of Age + BMI |
| Full Linear Regression | Train on all available numeric features |
| Polynomial Regression (Degree 2) | Model non-linear relationships and feature interactions |

### Evaluation Metrics:
- **R² Score**
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**

Model comparison is included to determine the most effective approach.
---

## 💾 Model Persistence
Trained models are saved using `pickle`:
- `full_model.pkl`
- `poly_model.pkl`
- `poly_transformer.pkl`

---

## 🌐 Deployment
A user-friendly **Streamlit web app** was developed to:

- Collect real-time user inputs
- Transform features using pre-trained transformers
- Generate predictions from both **Linear** and **Polynomial** models
- Display predicted insurance charges instantly

### Planned Enhancements:
Future updates aim to include lifestyle and medical history features such as:
- Detailed smoking history
- Activity levels
- Diet patterns
- Additional health indicators

---

## 🛠️ Tech Stack
- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **scikit-learn**
- **Streamlit**
- **Pickle**

---
