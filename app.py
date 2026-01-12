import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# -----------------------------
# PAGE TITLE
# -----------------------------
st.title("OBP Concentrate Silica Prediction App")

# -----------------------------
# LOAD DATA FUNCTION
# -----------------------------
@st.cache
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
# DEFINE FEATURES & TARGET  ✅ VERY IMPORTANT
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
# DATA CLEANING (STEP 1 & STEP 2)
# -----------------------------
# Replace plant symbols like '-' with NaN
df = df.replace('-', pd.NA)

# Convert columns to numeric
for col in features + [target]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows without target
df = df.dropna(subset=[target])
# -----------------------------
# FINAL STEP: FILL NaN IN INPUT FEATURES
# -----------------------------
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

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    min_samples_split=5,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# MODEL PERFORMANCE
# -----------------------------
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

st.subheader("Model Performance")
st.write(f"R² Score: {r2:.3f}")
st.write(f"RMSE: {rmse:.3f}")

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
