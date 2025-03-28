import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, precision_recall_fscore_support

# ✅ Configure Plot Style
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 18

# ==============================
# 📌 Load & Preprocess Dataset
# ==============================
file_path = r"D:\MAHESH\Paleoclimatology\paleoclimate_data.csv"
df = pd.read_csv(file_path)

# ✅ Standardize Column Names
df.columns = df.columns.str.strip().str.lower()
df.rename(columns={'iceage': 'age', 'co2wet': 'co2'}, inplace=True)

# ✅ Check Required Columns
required_columns = {'depth', 'age', 'agedif'}
missing_cols = required_columns - set(df.columns)
if missing_cols:
    raise KeyError(f"Missing required columns: {missing_cols}")

# ✅ Handle Missing Values
df.dropna(subset=['depth', 'age', 'agedif'], inplace=True)

# ==============================
# 📌 Feature Extraction
# ==============================
X_data = df[['depth', 'age']].values
Y_data = df[['agedif']].values  # Predicting δ18O (Age Difference)

# ✅ Normalize Input Features (Min-Max Scaling)
X_min, X_max = X_data.min(axis=0), X_data.max(axis=0)
X_data = (X_data - X_min) / (X_max - X_min)

# ✅ Split Dataset for Training & Testing
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)

# ✅ Create Time-Series Sequences (TFT requires 3D input)
time_steps = 10

def create_sequences(X, Y, time_steps):
    X_seq, Y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i : i + time_steps])
        Y_seq.append(Y[i + time_steps])
    return np.array(X_seq), np.array(Y_seq)

X_train_seq, Y_train_seq = create_sequences(X_train, Y_train, time_steps)
X_test_seq, Y_test_seq = create_sequences(X_test, Y_test, time_steps)

# ==============================
# 📌 Build & Train TFT Model
# ==============================
def build_tft(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.LSTM(64)(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# ✅ Train the TFT Model
tft_model = build_tft((time_steps, X_train.shape[1]))
tft_model.fit(X_train_seq, Y_train_seq, epochs=200, batch_size=32, verbose=1)

# ✅ Evaluate Model Performance
Y_pred = tft_model.predict(X_test_seq)
rmse = np.sqrt(mean_squared_error(Y_test_seq, Y_pred))
mse = mean_squared_error(Y_test_seq, Y_pred)
mae = mean_absolute_error(Y_test_seq, Y_pred)

# Convert to binary classification for F1-score evaluation
threshold = np.median(Y_test_seq)
Y_test_binary = (Y_test_seq >= threshold).astype(int)
Y_pred_binary = (Y_pred >= threshold).astype(int)

precision, recall, f1, _ = precision_recall_fscore_support(Y_test_binary, Y_pred_binary, average='binary')

print(f"RMSE: {rmse:.4f}")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"F1-Score: {f1:.4f}")

# ==============================
# 📌 Predict δ18O for Entire Dataset
# ==============================
def predict_agedif(X):
    X_norm = (X - X_min) / (X_max - X_min)
    X_seq, _ = create_sequences(X_norm, np.zeros(len(X_norm)), time_steps)
    return tft_model.predict(X_seq).flatten()

df['predicted_agedif'] = np.nan
df.loc[time_steps:, 'predicted_agedif'] = predict_agedif(df[['depth', 'age']].values)

# ✅ Save Updated Dataset with Predictions
output_path = r"D:\MAHESH\Paleoclimatology\paleoclimate_predictions.csv"
df.to_csv(output_path, index=False)
print(f"Predictions saved to: {output_path}")

# ✅ Print Sample Predictions
print(df[['depth', 'age', 'agedif', 'predicted_agedif']].head())

# ==============================
# 📌 δ18O Signal Graph
# ==============================
plt.figure(figsize=(12, 6))
plt.plot(df['age'], df['agedif'], label="Actual δ18O", color="blue", linewidth=2, alpha=0.7)
plt.xlabel("Years Before Present")
plt.ylabel("δ18O Value")
plt.grid(True)
plt.savefig("agedif_signal_graph.png", dpi=300)
plt.show()

# ==============================
# 📌 Fine-Tuned Forecast Error Signal Graph
# ==============================
df = df.sort_values("age", ascending=False)
window_size = max(5, len(df) // 50)
df["forecast_error"] = np.abs(df["age"] - df["age"].rolling(window=window_size, min_periods=1).median())
df["forecast_error"] = (df["forecast_error"] - df["forecast_error"].min()) / (df["forecast_error"].max() - df["forecast_error"].min())

plt.figure(figsize=(14, 6))
plt.plot(df["age"], df["forecast_error"], color='black', linewidth=2.5, linestyle="solid", label="Forecast Error")
plt.xlabel("Years Before Present", fontsize=23, fontweight='bold')
plt.ylabel("Forecast Error", fontsize=23, fontweight='bold')
plt.title("Forecast Error Signal Over Time", fontsize=23, fontweight='bold')
plt.gca().invert_xaxis()
plt.legend(fontsize=18)
plt.tight_layout()
plt.savefig("optimized_forecast_error_signal.png", dpi=300)
plt.show()
