# streamlit_app.py

import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ==============================
# Load and Train Model
# ==============================
df = pd.read_csv("Crop_recommendation.csv")

X = df.iloc[:, :-1]   # First 7 numerical inputs
y = df.iloc[:, -1]    # Categorical output (crop)

# Normalize input
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train Random Forest
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# ==============================
# Streamlit Web UI
# ==============================
st.title("🌱 Crop Recommendation System")
st.write("Enter soil & weather parameters to get the best crop recommendation.")

N = st.number_input("Nitrogen (N)", 0, 200, 50)
P = st.number_input("Phosphorus (P)", 0, 200, 50)
K = st.number_input("Potassium (K)", 0, 200, 50)
temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0, 80.0)
ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 200.0)

if st.button("🌾 Recommend Crop"):
    # Prepare input
    sample = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=X.columns)
    sample_scaled = scaler.transform(sample)

    # Predict
    prediction = rf.predict(sample_scaled)[0]
    st.success(f"✅ Recommended Crop: **{prediction}**")
