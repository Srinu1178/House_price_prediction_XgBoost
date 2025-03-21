
# 🏠 Bangalore House Price Prediction (Hybrid Model: XGBoost + ANN + LSTM)

This project aims to predict house prices in Bangalore using a hybrid ensemble of Machine Learning (XGBoost) and Deep Learning models (ANN and LSTM). The model pipeline includes feature engineering, preprocessing, training, explainability using SHAP, and data visualizations.

---

## 📁 Project Structure
```
model_training.py                # Main script for training and evaluation
scaler.pkl                       # Saved StandardScaler
xgb_model.pkl                    # Trained XGBoost model
ann_model.h5                     # Trained ANN model
lstm_model.h5                    # Trained LSTM model
shap_summary_plot_xgb.png       # SHAP summary for XGBoost
shap_summary_plot_ann.png       # SHAP summary for ANN
shap_summary_plot_lstm.png      # SHAP summary for LSTM
correlation_heatmap.png         # Correlation heatmap
price_distribution.png          # Price distribution histogram
```

---

## 📊 Dataset
- **File:** `Bengaluru_House_Data.csv`
- **Source:** Kaggle
- Contains features like: area_type, size, total_sqft, bath, location, price, etc.

---

## ⚙️ Features Implemented
- Missing value imputation
- Feature engineering (BHK, price_per_sqft, bath_per_bhk)
- Outlier removal
- Label Encoding
- Data Scaling (StandardScaler)
- Model Training:
  - ✅ XGBoost Regressor
  - ✅ Artificial Neural Network (ANN)
  - ✅ Long Short-Term Memory (LSTM)
- Weighted Hybrid Ensemble Model
- Model Evaluation using R² score
- SHAP-based Explainability
- Data Visualization

---

## 📈 Model Accuracies
The script outputs individual model accuracies and weighted hybrid model performance (based on R² score):
- **XGBoost Accuracy**
- **ANN Accuracy**
- **LSTM Accuracy**
- **Hybrid Accuracy (Weighted Ensemble)**

---

## 🔍 SHAP Explainability
- **XGBoost:** SHAP TreeExplainer
- **ANN:** SHAP DeepExplainer
- **LSTM:** SHAP KernelExplainer (Experimental workaround for 3D input)

---

## 📊 Visualizations
- Correlation Heatmap
- Price Distribution
- SHAP Summary Plots

---

## 🚀 How to Run
1. Make sure required libraries are installed:
```bash
pip install pandas numpy scikit-learn xgboost tensorflow shap seaborn matplotlib joblib
```

2. Update the path in `model_training.py` if necessary:
```python
df = pd.read_csv("path_to/Bengaluru_House_Data.csv")
```

3. Run the script:
```bash
python model_training.py
```

---

## 💾 Model Saving
- Models are saved as `.pkl` and `.keras` files for future prediction or deployment.

---

## 📂 Future Enhancements
- Streamlit Web App Deployment
- Meta-Model Stacking instead of Weighted Averaging
- Automated Hyperparameter Tuning
- Outlier detection using IQR or Isolation Forest
- Integration of Explainable AI (XAI) dashboards

---

## 📬 Contact
**Author:** Srinu  
**LinkedIn:** (www.linkedin.com/in/saga-naga-venkata-srinivasu-950570251)
