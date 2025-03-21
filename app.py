import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load models and scaler
scaler = joblib.load("D:/ProjectResearch/scaler.pkl")
xgb_model = joblib.load("D:/ProjectResearch/xgb_model.pkl")
ann_model = load_model("D:/ProjectResearch/ann_model.keras")
lstm_model = load_model("D:/ProjectResearch/lstm_model.keras")

# Page configuration
st.set_page_config(page_title="🏠 House Price Prediction", layout="wide")
st.title("🏡 Bengaluru House Price Prediction using XGBoost + ANN + LSTM (Hybrid Model)")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully.")
else:
    data = pd.read_csv("Bengaluru_House_Data.csv")
    st.info("ℹ️ Using default dataset (Bengaluru_House_Data.csv)")

# Preprocessing
def preprocess_data(df):
    df = df.copy()
    df.drop(['society', 'availability', 'balcony'], axis=1, errors='ignore', inplace=True)
    df['total_sqft'] = df['total_sqft'].apply(lambda x: str(x).split('-')[0] if '-' in str(x) else x)
    df['total_sqft'] = pd.to_numeric(df['total_sqft'], errors='coerce')
    df['total_sqft'].fillna(df['total_sqft'].median(), inplace=True)
    df['bath'].fillna(df['bath'].median(), inplace=True)
    df['location'].fillna(df['location'].mode()[0], inplace=True)
    df['area_type'].fillna(df['area_type'].mode()[0], inplace=True)
    df['size'].fillna(df['size'].mode()[0], inplace=True)

    if 'size' in df.columns:
        df['bhk'] = df['size'].apply(lambda x: int(str(x).split(' ')[0]) if pd.notnull(x) else 0)
        df.drop(['size'], axis=1, inplace=True)
    else:
        df['bhk'] = 0

    df['bhk'] = df['bhk'].replace(0, 1)
    df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']
    df['bath_per_bhk'] = df['bath'] / df['bhk']
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)

    df = df[df['total_sqft'] / df['bhk'] >= 300]
    df = df[df['price_per_sqft'] < df['price_per_sqft'].quantile(0.99)]

    from sklearn.preprocessing import LabelEncoder
    df['location'] = LabelEncoder().fit_transform(df['location'].astype(str))
    df['area_type'] = LabelEncoder().fit_transform(df['area_type'].astype(str))

    return df

# Preprocessing
data_cleaned = preprocess_data(data)
X = data_cleaned.drop(['price'], axis=1)
y = data_cleaned['price']
X_scaled = scaler.transform(X)

# Predictions
xgb_pred = xgb_model.predict(X_scaled)
ann_pred = ann_model.predict(X_scaled).flatten()
lstm_input = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
lstm_pred = lstm_model.predict(lstm_input).flatten()
hybrid_pred = (0.4 * xgb_pred + 0.35 * ann_pred + 0.25 * lstm_pred)

# Evaluation
def evaluate_model(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return round(r2, 4), round(mae, 2), round(rmse, 2)

xgb_r2, xgb_mae, xgb_rmse = evaluate_model(y, xgb_pred)
ann_r2, ann_mae, ann_rmse = evaluate_model(y, ann_pred)
lstm_r2, lstm_mae, lstm_rmse = evaluate_model(y, lstm_pred)
hybrid_r2, hybrid_mae, hybrid_rmse = evaluate_model(y, hybrid_pred)

comparison_df = pd.DataFrame({
    "Model": ["XGBoost", "ANN", "LSTM", "Hybrid"],
    "R² Score": [xgb_r2, ann_r2, lstm_r2, hybrid_r2],
    "MAE": [xgb_mae, ann_mae, lstm_mae, hybrid_mae],
    "RMSE": [xgb_rmse, ann_rmse, lstm_rmse, hybrid_rmse]
})

st.markdown("### 🔘 Select what you want to view:")

# Define layout with 3 columns
colA1, colA2, colA3 = st.columns(3)

with colA1:
    show_data = st.button("📊 Show Preprocessed Data")
with colA2:
    show_predictions = st.button("📈 Show Model Predictions")
with colA3:
    show_visuals = st.button("📉 Show Data Visualizations")

# Display Data
if show_data or show_predictions:
    colD1, colD2 = st.columns(2)

    if show_data:
        with colD1:
            st.subheader("🧼 Preprocessed Data")
            st.dataframe(data_cleaned.head(20))

    if show_predictions:
        with colD2:
            st.subheader("📈 Model Predictions")
            results = pd.DataFrame({
                "Actual Price (Lakhs)": y,
                "XGBoost Prediction": xgb_pred.round(2),
                "ANN Prediction": ann_pred.round(2),
                "LSTM Prediction": lstm_pred.round(2),
                "Hybrid Model Prediction": hybrid_pred.round(2)
            })
            st.dataframe(results.head(20))

# Data Visualizations
if show_visuals:
    st.subheader("📉 Data Visualizations")
    colV1, colV2 = st.columns(2)

    with colV1:
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        sns.heatmap(data_cleaned.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax1)
        ax1.set_title("Correlation Heatmap")
        st.pyplot(fig1)

    with colV2:
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        sns.histplot(data_cleaned['price'], bins=40, kde=True, ax=ax2)
        ax2.set_title("Price Distribution")
        st.pyplot(fig2)

    # One more row (next two columns)
    colV3, colV4 = st.columns(2)

    with colV3:
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        sns.boxplot(x='bhk', y='price', data=data_cleaned, ax=ax3)
        ax3.set_title("BHK vs Price Boxplot")
        st.pyplot(fig3)

# SHAP Explainability
if st.button("🔍 Show SHAP Explainability (XGBoost)"):
    st.warning("⏳ Generating SHAP summary... This might take a few seconds.")
    explainer_xgb = shap.Explainer(xgb_model, X_scaled)
    shap_values = explainer_xgb(X_scaled)
    st.subheader("📢 SHAP Summary Plot (XGBoost)")
    shap.summary_plot(shap_values, X_scaled, feature_names=X.columns)
    fig4 = plt.gcf()
    plt.tight_layout()
    st.pyplot(fig4)

# Model Comparison
if st.button("📊 Show Model Performance Comparison"):
    st.subheader("📊 Model Evaluation Metrics")
    colM1, colM2 = st.columns(2)

    with colM1:
        st.dataframe(comparison_df)

    with colM2:
        fig6, ax6 = plt.subplots(figsize=(7, 5))
        sns.barplot(data=comparison_df, x='Model', y='R² Score', palette='viridis', ax=ax6)
        ax6.set_ylim(0, 1)
        ax6.set_title("Model R² Score Comparison")
        st.pyplot(fig6)

# Footer
st.markdown("---")
st.markdown("👨‍💻 Developed by Srinu | Hybrid ML + DL Model | Streamlit Deployment")

