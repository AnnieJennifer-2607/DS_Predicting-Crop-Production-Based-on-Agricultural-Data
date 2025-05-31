import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import joblib
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go

# Page config
st.set_page_config(page_title="🌾 Crop Production Predictor", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #f9f9f9;}
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 0.5em 1em;
    }
    .stDownloadButton>button {
        background-color: #2C9CDB;
        color: white;
        font-size: 16px;
        padding: 0.5em 1em;
    }
    </style>
""", unsafe_allow_html=True)

# Load data and model
df_cleaned = pd.read_csv("Cleaned_CropProduction.csv")
model = joblib.load("crop_production_rf_model.pkl", mmap_mode=None)

try:
    model_features = joblib.load("model_features.pkl")
except FileNotFoundError:
    st.error("🔍Feature mapping file not found. Please check your model setup.")
    model_features = []

# Sidebar - Inputs
st.sidebar.image("crop_logo.jpeg", use_container_width=True)
st.sidebar.title("⚙️ Input Parameters")

area = st.sidebar.selectbox("📍 Select Region", df_cleaned["Area"].unique())
crop = st.sidebar.selectbox("🌾 Select Crop", df_cleaned["Item"].unique())
year = st.sidebar.slider("🗓️ Select Year", int(df_cleaned["Year"].min()), int(df_cleaned["Year"].max()), 2025)
area_harvested = st.sidebar.number_input("🌿 Area Harvested (in hectares)", min_value=0.0, value=1000.0)
yield_amount = st.sidebar.number_input("🌱 Yield (tons per hectare)", min_value=0.0, value=2.5)

input_data = pd.DataFrame([[area, crop, year, area_harvested, yield_amount]], columns=["Area", "Item", "Year", "Area harvested", "Yield"])
input_encoded = pd.get_dummies(input_data, columns=["Area", "Item"])
input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)
input_encoded = input_encoded.astype(float)

# Main title
st.title("📊 Crop Production Prediction Dashboard")
st.markdown("Predict agricultural output and explore crop production trends across regions.")

# Prediction section
with st.expander("🤖 Predict Crop Production", expanded=True):
    if st.sidebar.button("🔮 Predict Production"):
        prediction = model.predict(input_encoded)
        st.session_state.prediction = np.round(prediction[0], 2)
        st.success(f"🔮 **Predicted Crop Production:** {st.session_state.prediction} tons")
    elif "prediction" in st.session_state:
        st.info(f"🔮 Last Prediction: {st.session_state.prediction} tons")

# Tabs for EDA
tab1, tab2, tab3 = st.tabs(["🗒️ Summary", "📈 Trends", "🔍 Regional Analysis"])

with tab1:
    st.subheader("🗒️ Dataset Summary")
    st.dataframe(df_cleaned.describe(), use_container_width=True)

with tab2:
    st.subheader("📈 Crop Production Trends Over the Years")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x='Year', y='Production', data=df_cleaned, hue='Item', legend=False, ax=ax)
    ax.set_title("Yearly Trend in Crop Production")
    st.pyplot(fig)


with tab3:
    st.subheader("🗺️ Region-wise Production Distribution")
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.barplot(data=df_cleaned, x="Area", y="Production", ax=ax, palette="Set3")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    st.pyplot(fig)

# Download prediction report
if "prediction" in st.session_state:
    result = pd.DataFrame({
        "Region": [area],
        "Crop": [crop],
        "Year": [year],
        "Area harvested (ha)": [area_harvested],
        "Yield (tons/ha)": [yield_amount],
        "Predicted Production": [st.session_state.prediction]
    })
    csv = result.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Prediction Report", data=csv,
                       file_name="Crop_Prediction_Report.csv", mime="text/csv")
#.\venv\Scripts\Activate.ps1
#streamlit run CropPredicationApp.py
# Ctrl+C to stop the server