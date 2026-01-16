import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# -----------------------------
# PAGE TITLE
# -----------------------------
st.title("OBP Concentrate Silica Prediction App (Linear Regression)")

# -----------------------------
# LOAD DATA FUNCTION
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_excel(
        "OBP_Silica_Cleaned_Data.xlsx",
        sheet_name="Sheet1"
    )
    return df

# -----------------------------
# LOAD DATA
# -----------------------------
df = load_data()

# -----------------------------
# DEFINE FEATURES & TARGET
# -----------------------------
features = [
    'feed_fe_pct',
    'feed_sio2_pct',
    'feed_al2o3_pct',
    'feed_loi_pct',
    'minus_10micron_pct',
    'minus_45micron_pct',
    'minus_150micron_pct'
]

target = 'concentrate_sio2_pct'

# -----------------------------
# DATA CLEANING
# -----------------------------
df = df.replace('-', pd.NA)

for col in features + [target]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=[target])

for col in features:
    df[col] = df[col].fillna(df[col].mean())

# -----------------------------
# MODEL PREPARATION
# -----------------------------
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# MODEL PERFORMANCE
# -----------------------------
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.subheader("Model Performance")
st.write(f"R² Score: {r2:.3f}")
st.write(f"RMSE: {rmse:.3f}")
st.subheader("Predicted vs Actual Concentrate SiO₂")

fig1, ax1 = plt.subplots()
ax1.scatter(y_test, y_pred)
ax1.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle='--'
)

ax1.set_xlabel("Actual Concentrate SiO₂ (%)")
ax1.set_ylabel("Predicted Concentrate SiO₂ (%)")
ax1.set_title("Variance Explained by Feed Chemistry & PSD")

st.pyplot(fig1)
st.subheader("Residual Analysis (Beneficiation Stability)")

residuals = y_test - y_pred

fig2, ax2 = plt.subplots()
ax2.scatter(y_pred, residuals)
ax2.axhline(0, linestyle='--')

ax2.set_xlabel("Predicted Concentrate SiO₂ (%)")
ax2.set_ylabel("Residual (Actual − Predicted)")
ax2.set_title("Residual Distribution – Classification & Liberation")

st.pyplot(fig2)
st.subheader("Influence of Process Variables on Concentrate SiO₂")

coef_df = pd.DataFrame({
    "Parameter": features,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient")

fig3, ax3 = plt.subplots()
ax3.barh(coef_df["Parameter"], coef_df["Coefficient"])
ax3.set_xlabel("Regression Coefficient")
ax3.set_title("Beneficiation Parameter Influence (Linear Regression)")

st.pyplot(fig3)


# -----------------------------
# USER INPUT
# -----------------------------
st.subheader("Enter Feed & PSD Values")

feed_fe = st.number_input("Feed Fe (%)", 50.0, 65.0, 55.0)
feed_sio2 = st.number_input("Feed SiO₂ (%)", 5.0, 25.0, 10.0)
feed_al2o3 = st.number_input("Feed Al₂O₃ (%)", 2.0, 8.0, 4.5)
feed_loi = st.number_input("Feed LOI (%)", 2.0, 8.0, 4.5)

minus_10 = st.number_input("-10 micron (%)", 0.0, 20.0, 8.0)
minus_45 = st.number_input("-45 micron (%)", 20.0, 60.0, 40.0)
minus_150 = st.number_input("-150 micron (%)", 80.0, 100.0, 93.0)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Concentrate SiO₂"):
    input_df = pd.DataFrame([[ 
        feed_fe,
        feed_sio2,
        feed_al2o3,
        feed_loi,
        minus_10,
        minus_45,
        minus_150
    ]], columns=features)

    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Concentrate SiO₂: {prediction:.2f} %")

