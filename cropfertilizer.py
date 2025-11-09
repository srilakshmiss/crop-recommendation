# streamlit_app.py

import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ==============================
# Load Dataset & Train Model
# ==============================
df = pd.read_csv("Crop_recommendation.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# ==============================
# Fertilizer Recommendation Mapping
# ==============================
fertilizer_dict = {
    'apple': 'NPK 10:10:10 or urea + compost mixture for tree fruiting',
    'banana': 'Compost + MOP + Urea (N:P:K = 200:50:200 kg/ha)',
    'barley': 'DAP + Urea in split doses, 60:40:40 NPK ratio',
    'beetroot': 'Balanced fertilizer with more phosphorus (NPK 5:10:10)',
    'blackgram': 'DAP + Rhizobium inoculation (20:40:40 NPK)',
    'blackpepper': 'Farmyard manure + NPK 50:50:150 g per vine yearly',
    'broccoli': 'Urea + Superphosphate + Potash (150:100:100 NPK)',
    'cabbage': 'NPK 120:80:60 + organic compost',
    'carrot': 'Superphosphate + MOP + Urea (40:60:60 NPK)',
    'chickpea': 'NPK 20:40:40 + DAP basal application',
    'chilli': 'FYM + Urea + SSP + MOP (100:50:50 NPK)',
    'chrysanthemum': 'Compost + NPK 75:75:75',
    'coconut': 'Organic manure + Urea + SSP + MOP (500:320:1200 g/tree)',
    'coffee': 'NPK 120:90:120 + compost + lime annually',
    'cotton': 'Urea + SSP + MOP (60:30:30 NPK)',
    'cucumber': 'NPK 90:60:60 + compost + micronutrients',
    'daisy': 'Organic compost + NPK 60:60:40',
    'garlic': 'NPK 100:50:50 + FYM',
    'ginger': 'Compost + Urea + SSP (75:50:50 NPK)',
    'grapes': 'FYM + NPK 500:200:700 g/vine yearly',
    'greengram': 'NPK 20:40:40 + DAP + Rhizobium inoculation',
    'groundnut': 'Gypsum + SSP + Urea (25:50:75 NPK)',
    'guava': 'FYM + NPK 600:400:600 g/tree yearly',
    'hibiscus': 'Organic compost + NPK 50:30:30',
    'horsegram': 'DAP basal dose + Rhizobium culture',
    'jackfruit': 'FYM + NPK 600:300:600 g/tree yearly',
    'jasmine': 'Compost + NPK 40:40:40 + micronutrients',
    'jowar': 'Urea + SSP (80:40:40 NPK)',
    'jute': 'NPK 80:40:40 + FYM 5t/ha',
    'kidneybeans': 'NPK 30:60:30 + Rhizobium inoculation',
    'lemon': 'FYM + NPK 500:250:500 g/tree',
    'lentil': 'NPK 20:40:40 + DAP basal + biofertilizer',
    'maize': 'Urea + DAP + MOP (120:60:40 NPK)',
    'mango': 'FYM + NPK 1000:500:1000 g/tree annually',
    'mothbeans': 'NPK 20:40:40 + DAP basal',
    'mungbean': 'NPK 20:40:40 + Rhizobium inoculation',
    'muskmelon': 'FYM + NPK 100:60:60 + micronutrients',
    'mustard': 'NPK 80:40:40 + Sulphur',
    'onion': 'NPK 100:50:50 + FYM',
    'orange': 'FYM + NPK 600:400:600 g/tree yearly',
    'papaya': 'FYM + NPK 250:250:500 g/plant + micronutrients',
    'peas': 'NPK 20:60:60 + DAP + Rhizobium culture',
    'pigeonpeas': 'NPK 25:50:25 + DAP basal + FYM',
    'pomegranate': 'FYM + NPK 500:200:500 g/tree yearly',
    'potato': 'Urea + SSP + MOP (150:100:100 NPK)',
    'radish': 'NPK 40:60:40 + FYM 25 t/ha',
    'ragi': 'NPK 60:30:30 + compost basal dose',
    'rice': 'Urea + DAP + MOP (100:40:40 NPK)',
    'rose': 'FYM + NPK 60:60:40 + micronutrients',
    'sugarcane': 'NPK 340:170:170 + pressmud compost',
    'sunflower': 'NPK 60:60:40 + FYM',
    'tomato': 'Urea + DAP + MOP (100:50:50 NPK)',
    'watermelon': 'NPK 80:60:60 + FYM + micronutrients',
    'wheat': 'NPK 120:60:40 + DAP basal + Urea topdress'
}

# ==============================
# Streamlit UI
# ==============================
st.title("🌱 Crop Recommendation System")
st.write("Enter soil and weather parameters to get the best crop and fertilizer suggestion.")

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

    # Predict crop
    crop_prediction = rf.predict(sample_scaled)[0]

    # Fertilizer suggestion
    fertilizer = fertilizer_dict.get(crop_prediction, "General NPK 20:20:20 or organic compost")

    st.success(f"✅ Recommended Crop: **{crop_prediction.capitalize()}**")
    st.info(f"💡 Suggested Fertilizer: {fertilizer}")
