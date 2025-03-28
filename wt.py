import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# ✅ Load datasets
df_pinn = pd.read_csv(r"D:\MAHESH\Paleoclimatology\paleoclimate_predictions_pinn1.csv")
df_tft = pd.read_csv(r"D:\MAHESH\Paleoclimatology\paleoclimate_predictionstft.csv")
df_bnn = pd.read_csv(r"D:\MAHESH\Paleoclimatology\predicted_paleoclimate_data.csv")

# ✅ Standardize column names
df_pinn.rename(columns={"depth": "Depth", "predicted_δ18o": "PINN_Predicted_d18O"}, inplace=True)
df_tft.rename(columns={"depth": "Depth", "predicted_δ18o": "TFT_Predicted_d18O"}, inplace=True)
df_bnn.rename(columns={"Predicted d18O": "BNN_Predicted_d18O"}, inplace=True)

# ✅ Merge datasets on Depth (ensure "Depth" is present in all)
df = df_pinn.merge(df_tft, on="Depth", how="inner").merge(df_bnn, on="Depth", how="inner")

# ✅ Debugging step: Check for NaN values after merging
print("\n🔍 Checking for NaN values:")
print(df.isnull().sum())

# ✅ Drop or fill missing values
if df.isnull().values.any():
    print("⚠️ NaN values found! Handling missing data...")
    df.dropna(inplace=True)  # Drop rows with NaN values
    # Alternatively, you can fill NaN values: df.fillna(df.mean(), inplace=True)

# ✅ Ensure the true δ18O column exists
if "δ18o" in df.columns:
    df.rename(columns={"δ18o": "True_d18O"}, inplace=True)
elif "d18O" in df.columns:
    df.rename(columns={"d18O": "True_d18O"}, inplace=True)
else:
    raise KeyError("⚠️ Could not find the true δ18O column!")

# ✅ Extract true and predicted values
true_values = df["True_d18O"].values.flatten()
pinn_pred = df["PINN_Predicted_d18O"].values.flatten()
tft_pred = df["TFT_Predicted_d18O"].values.flatten()
bnn_pred = df["BNN_Predicted_d18O"].values.flatten()

# ✅ Check array shapes before computing MSE
print("\n✅ Shape Debugging:")
print(f"True δ18O shape: {true_values.shape}")
print(f"PINN Prediction shape: {pinn_pred.shape}")
print(f"TFT Prediction shape: {tft_pred.shape}")
print(f"BNN Prediction shape: {bnn_pred.shape}")

# ✅ Ensure shapes match before computing errors
assert true_values.shape == pinn_pred.shape, "Mismatch: true_values & PINN predictions"
assert true_values.shape == tft_pred.shape, "Mismatch: true_values & TFT predictions"
assert true_values.shape == bnn_pred.shape, "Mismatch: true_values & BNN predictions"

# ✅ Compute Evaluation Metrics
pinn_mse = mean_squared_error(true_values, pinn_pred)
tft_mse = mean_squared_error(true_values, tft_pred)
bnn_mse = mean_squared_error(true_values, bnn_pred)

pinn_rmse = np.sqrt(pinn_mse)
tft_rmse = np.sqrt(tft_mse)
bnn_rmse = np.sqrt(bnn_mse)

pinn_mae = np.mean(np.abs(true_values - pinn_pred))
tft_mae = np.mean(np.abs(true_values - tft_pred))
bnn_mae = np.mean(np.abs(true_values - bnn_pred))

# ✅ Compute inverse-error-based weights
error_inv = np.array([1/pinn_mse, 1/tft_mse, 1/bnn_mse])
weights = error_inv / np.sum(error_inv)  # Normalize weights

# ✅ Compute final weighted δ18O prediction
df["Weighted_d18O"] = (weights[0] * pinn_pred) + (weights[1] * tft_pred) + (weights[2] * bnn_pred)

# ✅ Save final weighted predictions
df.to_csv("final_weighted_d18O_predictions.csv", index=False)
print("\n✅ Final weighted δ18O predictions saved to 'final_weighted_d18O_predictions.csv'.")

# ✅ Print weights and errors for reference
print("\n📌 Model Weights:")
print(f"PINN Weight: {weights[0]:.4f}, MSE: {pinn_mse:.6f}, RMSE: {pinn_rmse:.6f}, MAE: {pinn_mae:.6f}")
print(f"TFT Weight: {weights[1]:.4f}, MSE: {tft_mse:.6f}, RMSE: {tft_rmse:.6f}, MAE: {tft_mae:.6f}")
print(f"BNN Weight: {weights[2]:.4f}, MSE: {bnn_mse:.6f}, RMSE: {bnn_rmse:.6f}, MAE: {bnn_mae:.6f}")