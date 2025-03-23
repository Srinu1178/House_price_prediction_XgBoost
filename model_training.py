# -------------------- Imports --------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow.keras.backend as K
import shap
import joblib
import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("Bengaluru_House_Data.csv")
df['price'] = df['price'] * 100000
df.head()

# Data Preprocessing
df.drop(['society', 'availability', 'balcony'], axis=1, inplace=True)
df['total_sqft'] = df['total_sqft'].apply(lambda x: str(x).split('-')[0] if '-' in str(x) else x)
df['total_sqft'] = pd.to_numeric(df['total_sqft'], errors='coerce')

# Impute missing values
df['total_sqft'].fillna(df['total_sqft'].median(), inplace=True)
df['bath'].fillna(df['bath'].median(), inplace=True)
df['location'].fillna(df['location'].mode()[0], inplace=True)
df['area_type'].fillna(df['area_type'].mode()[0], inplace=True)
df['size'].fillna(df['size'].mode()[0], inplace=True)

# Feature Engineering
if 'size' in df.columns:
    df['bhk'] = df['size'].apply(lambda x: int(str(x).split(' ')[0]) if pd.notnull(x) else 0)
    df.drop(['size'], axis=1, inplace=True)
else:
    df['bhk'] = 0  # fallback if 'size' column is missing

# Handle potential division by zero and missing values
# Replace 0 bhk to avoid division error
df['bhk'] = df['bhk'].replace(0, 1)

# Calculate price per sqft and bath per bhk safely
df['price_per_sqft'] = df['price']/ df['total_sqft']
df['bath_per_bhk'] = df['bath'] / df['bhk']

# Handle infinities and missing values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)

# Outlier Removal
df = df[df['total_sqft'] / df['bhk'] >= 300]
df = df[df['price_per_sqft'] < df['price_per_sqft'].quantile(0.99)]

# Encode categorical features
label_encoder_location = LabelEncoder()
df['location'] = label_encoder_location.fit_transform(df['location'].astype(str))
label_encoder_area = LabelEncoder()
df['area_type'] = label_encoder_area.fit_transform(df['area_type'].astype(str))

# Define features and target
X = df.drop(['price'], axis=1)
y = df['price']

# Check data types before scaling
print("X dtypes before scaling:\n", X.dtypes)

# Train-test split and scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# XGBoost model (updated parameters for improved accuracy)
xgb_model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=7, subsample=0.9, colsample_bytree=0.8, random_state=42)
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)
acc_xgb = r2_score(y_test, y_pred_xgb) * 100
print(f"XGBoost Accuracy: {acc_xgb:.2f}%")

# Deep ANN model with regularization and callbacks
ann_model = Sequential()
ann_model.add(Dense(512, activation='relu', input_dim=X_train_scaled.shape[1]))
ann_model.add(Dropout(0.3))
ann_model.add(Dense(256, activation='relu'))
ann_model.add(Dropout(0.3))
ann_model.add(Dense(128, activation='relu'))
ann_model.add(Dense(64, activation='relu'))
ann_model.add(Dense(32, activation='relu'))
ann_model.add(Dense(1))  # Output layer

# Compile the model
optimizer = Adam(learning_rate=0.001)
ann_model.compile(optimizer=optimizer, loss='mse')

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)

# Fit the model
history = ann_model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=300,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=0
)

# Prediction and accuracy
y_pred_ann = ann_model.predict(X_test_scaled).flatten()
acc_ann = r2_score(y_test, y_pred_ann) * 100
print(f"Improved ANN Accuracy: {acc_ann:.2f}%")

# CrossLayer definition
class CrossLayer(Layer):
    def __init__(self, num_layers, **kwargs):
        super(CrossLayer, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.cross_weights = []
        self.cross_biases = []

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        for i in range(self.num_layers):
            self.cross_weights.append(self.add_weight(name=f'weight_{i}',
                                                      shape=(input_dim, 1),
                                                      initializer='glorot_uniform',
                                                      trainable=True))
            self.cross_biases.append(self.add_weight(name=f'bias_{i}',
                                                     shape=(input_dim, ),
                                                     initializer='zeros',
                                                     trainable=True))
        super(CrossLayer, self).build(input_shape)

    def call(self, inputs):
        x0 = inputs
        x = x0
        for i in range(self.num_layers):
            xw = K.dot(x, self.cross_weights[i])
            xw = xw * x0
            x = xw + self.cross_biases[i] + x
        return x

# DCN Architecture with regularization
inputs = Input(shape=(X_train_scaled.shape[1],))

# Cross Layer Output
cross_output = CrossLayer(num_layers=4)(inputs)  # Increased cross layers

# Deep Part of the Model
deep = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(inputs)  # Increased depth and L2 regularization
deep = Dropout(0.3)(deep)  # Dropout for regularization
deep = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(deep)
deep = Dropout(0.3)(deep)  # Dropout layer
deep = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(deep)

# Combine Cross and Deep Outputs
combined = Concatenate()([cross_output, deep])

# Output Layer
output = Dense(1)(combined)

# Compile the Model
dcn_model = Model(inputs=inputs, outputs=output)
dcn_model.compile(optimizer='adam', loss='mse')

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)

# Train the Model
history = dcn_model.fit(X_train_scaled, y_train,
                        validation_split=0.2,
                        epochs=500,  # Increased number of epochs for better convergence
                        batch_size=32,
                        callbacks=[early_stop, reduce_lr],
                        verbose=1)

# Evaluate the Model
y_pred_dcn = dcn_model.predict(X_test_scaled).flatten()
acc_dcn = r2_score(y_test, y_pred_dcn) * 100
print(f"DCN Accuracy: {acc_dcn:.2f}%")


y_pred_xgb = xgb_model.predict(X_test_scaled)
y_pred_ann = ann_model.predict(X_test_scaled)
y_pred_dcn = dcn_model.predict(X_test_scaled)

# Combine predictions (e.g., weighted average)
# Here we are using equal weight for all models
y_pred_hybrid = (y_pred_xgb + y_pred_ann.flatten() + y_pred_dcn.flatten()) / 3

# Evaluate hybrid model
hybrid_acc = r2_score(y_test, y_pred_hybrid) * 100
print(f"Hybrid Model Accuracy: {hybrid_acc:.2f}%")


joblib.dump(scaler, 'scaler.pkl')
joblib.dump(xgb_model, 'xgb_model.pkl')
ann_model.save('ann_model.keras')
dcn_model.save('dcn_model.keras')

# Evaluate individual models
r2_xgb = r2_score(y_test, y_pred_xgb)
r2_ann = r2_score(y_test, y_pred_ann)
r2_dcn = r2_score(y_test, y_pred_dcn)

# Evaluate hybrid models
r2_hybrid_simple = r2_score(y_test, y_pred_hybrid)

# Create a comparison table
comparison_df = pd.DataFrame({
    'Model': ['XGBoost', 'ANN', 'DCN', ' Hybrid'],
    'R2 Score': [r2_xgb, r2_ann, r2_dcn, r2_hybrid_simple]
})

print(comparison_df)
# Create a comparison table
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='R2 Score', data=comparison_df, palette='viridis')
plt.title('Model Comparison: R2 Scores')
plt.ylabel('R2 Score')
plt.xlabel('Model')
plt.savefig('model_comparison_plot.png', bbox_inches='tight', dpi=300)  
plt.show()
# Predictions with hybrid model
y_pred_hybrid = (y_pred_xgb + y_pred_ann.flatten() + y_pred_dcn.flatten()) / 3

# Scatter plot for Hybrid Model
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_hybrid, color='purple', alpha=0.6, label='Hybrid Model')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
plt.title('Hybrid Model: Predicted vs Actual')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.savefig('Hybrid_model_prediction.png', bbox_inches='tight', dpi=300)  
plt.show()

# R2 Score for Hybrid Model
r2_hybrid = r2_score(y_test, y_pred_hybrid)
print(f"Hybrid Model R2 Score: {r2_hybrid:.2f}")
