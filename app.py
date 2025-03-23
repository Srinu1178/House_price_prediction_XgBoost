import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import asyncio
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import LabelEncoder

# ----------------------- Page Config ----------------------- #
st.set_page_config(page_title="House Price Predictor", layout="wide")

# ----------------------- Load Models ----------------------- #
@st.cache_resource
def load_xgb_model():
    return joblib.load('xgb_model.pkl')

@st.cache_resource
def load_scaler():
    return joblib.load('scaler.pkl')

async def load_models():
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        scaler, xgb_model = await asyncio.gather(
            loop.run_in_executor(executor, load_scaler),
            loop.run_in_executor(executor, load_xgb_model)
        )
    return scaler, xgb_model

with st.spinner("Loading models..."):
    scaler, xgb_model = asyncio.run(load_models())

# ----------------------- Load Data & Preprocess ----------------------- #
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("Bengaluru_House_Data.csv")
    df['price'] *= 100000

    df.drop(['society', 'availability', 'balcony'], axis=1, inplace=True)
    df['total_sqft'] = df['total_sqft'].apply(lambda x: str(x).split('-')[0] if '-' in str(x) else x)
    df['total_sqft'] = pd.to_numeric(df['total_sqft'], errors='coerce')

    df['total_sqft'].fillna(df['total_sqft'].median(), inplace=True)
    df['bath'].fillna(df['bath'].median(), inplace=True)
    df['location'].fillna(df['location'].mode()[0], inplace=True)
    df['area_type'].fillna(df['area_type'].mode()[0], inplace=True)
    df['size'].fillna(df['size'].mode()[0], inplace=True)

    df['bhk'] = df['size'].apply(lambda x: int(str(x).split(' ')[0]) if pd.notnull(x) else 1)
    df.drop('size', axis=1, inplace=True)
    df['bhk'] = df['bhk'].replace(0, 1)
    df['bath_per_bhk'] = df['bath'] / df['bhk']
    df['price_per_sqft'] = df['price'] / df['total_sqft']

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)

    df = df[df['total_sqft'] / df['bhk'] >= 300]
    df = df[df['price'] < df['price'].quantile(0.99)]

    return df

df = load_and_preprocess_data()
st.subheader("🏡 Bangalore Houses Data")
st.dataframe(df[['location', 'area_type', 'total_sqft', 'bhk', 'bath']])
# ----------------------- Label Encoding ----------------------- #
label_encoder_location = LabelEncoder()
df['location'] = label_encoder_location.fit_transform(df['location'].astype(str))

label_encoder_area = LabelEncoder()
df['area_type'] = label_encoder_area.fit_transform(df['area_type'].astype(str))

# ----------------------- Define Features ----------------------- #
X = df.drop(['price'], axis=1)
y = df['price']

# ----------------------- Streamlit UI ----------------------- #
st.title("🏡 House Price Prediction App")
st.markdown("A robust ML-based house price prediction using **XGBoost** and dynamic filters.")

st.subheader("📊 Sample Preprocessed Dataset")
st.dataframe(df.head())

# ----------------------- Sidebar Inputs ----------------------- #
st.sidebar.header("🔍 Enter House Details")

location_options = label_encoder_location.inverse_transform(np.unique(df['location']))
area_type_options = label_encoder_area.inverse_transform(np.unique(df['area_type']))

location = st.sidebar.selectbox("📍 Location", location_options)
area_type = st.sidebar.selectbox("🏘 Area Type", area_type_options)
total_sqft = st.sidebar.number_input("📐 Total Square Feet", min_value=300, value=1000, step=50)

# Conditional BHK based on area_type and total_sqft
def suggest_bhk_options(area_type, sqft):
    if area_type == "Super built-up  Area":
        return list(range(1, 6)) if sqft >= 600 else [1, 2]
    elif area_type == "Built-up  Area":
        return list(range(1, 5)) if sqft >= 800 else [1, 2, 3]
    elif area_type == "Plot  Area":
        return list(range(2, 6)) if sqft >= 1200 else [1, 2, 3]
    elif area_type == "Carpet  Area":
        return list(range(1, 4)) if sqft < 700 else list(range(2, 5))
    else:
        return [1, 2, 3]

bhk_options = suggest_bhk_options(area_type, total_sqft)
bhk = st.sidebar.selectbox("🛏 BHK (Bedrooms)", bhk_options)
def suggest_bathroom_options(area_type, sqft, bhk):
    if area_type == "Plot  Area":
        if sqft < 1000:
            return list(range(1, min(3, bhk+2)))
        elif sqft < 1500:
            return list(range(2, min(4, bhk+2)))
        else:
            return list(range(2, min(5, bhk+3)))
    elif area_type == "Super built-up  Area":
        if bhk == 1:
            return [1, 2]
        elif bhk == 2:
            return [2, 3]
        elif bhk == 3:
            return [2, 3, 4]
        elif bhk >= 4:
            return [3, 4, 5]
    elif area_type == "Built-up  Area":
        if bhk == 1:
            return [1]
        elif bhk == 2:
            return [1, 2]
        elif bhk == 3:
            return [2, 3]
        else:
            return [3, 4]
    elif area_type == "Carpet  Area":
        return list(range(1, min(4, bhk + 2)))
    else:
        return list(range(1, bhk + 2))

bath_options = suggest_bathroom_options(area_type, total_sqft, bhk)
bath = st.sidebar.selectbox("🚽 Bathrooms", bath_options)

# ----------------------- Prediction Section ----------------------- #
if st.button("💡 Predict House Price"):
    try:
        # Dynamic price_per_sqft based on average of selected filters
        area_encoded = label_encoder_area.transform([area_type])[0]
        location_encoded = label_encoder_location.transform([location])[0]

        filtered_df = df[(df['location'] == location_encoded) & (df['area_type'] == area_encoded)]
        if not filtered_df.empty:
            avg_pps = filtered_df['price_per_sqft'].mean()
        else:
            avg_pps = df['price_per_sqft'].mean()

        # Check if exact house exists in dataset
        matching_row = df[
            (df['area_type'] == area_type) &
            (df['location'] == location) &
            (df['total_sqft'] == total_sqft) &
            (df['bath'] == bath) &
            (df['bhk'] == bhk)
        ]

        if not matching_row.empty:
            matched_price = matching_row['price'].mean()
            st.info("📍 This house already exists in the dataset.")
            st.success(f"🏠 **Actual House Price from Dataset: ₹ {int(matched_price):,}/-**")
        else:
            input_df = pd.DataFrame({
                'area_type': [area_encoded],
                'location': [location_encoded],
                'total_sqft': [total_sqft],
                'bath': [bath],
                'bhk': [bhk],
                'price_per_sqft': [avg_pps],
                'bath_per_bhk': [bath / bhk]
            })

        input_scaled = scaler.transform(input_df)
        y_pred = xgb_model.predict(input_scaled)
        price = y_pred[0]

        st.success(f"🏠 **Estimated House Price: ₹ {int(price):,}/-**")
    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")

# ----------------------- Footer ----------------------- #
st.markdown("---")
st.markdown("📌 Developed by **Srinu** | House Price Prediction App using Streamlit + XGBoost")
