import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchbnn as bnn  # Bayesian Neural Network library
from sklearn.metrics import mean_absolute_error

# ✅ Set plot style
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 23

# ✅ Load dataset
df = pd.read_csv(r"D:\MAHESH\Paleoclimatology\paleoclimate_data.csv", encoding="utf-8")  
df = df.dropna()  # Remove missing values
print(f"Total data points: {len(df)}")  

# ✅ Ensure required columns exist
features = ["Depth", "IceAge", "AgeDif", "GasAge", "CO2wet"]
if not all(col in df.columns for col in features):
    raise KeyError("Some required feature columns are missing!")

# ✅ Check if δ18O column exists, else generate synthetic values
if "d18O" not in df.columns:
    print("⚠️ 'd18O' column missing! Generating synthetic values for training.")
    df["d18O"] = np.random.normal(loc=-5, scale=1, size=len(df))  # Generate synthetic δ18O values

# ✅ Extract input features and target variable (δ18O)
X = df[features].values  
y = df["d18O"].values  # Target variable (δ18O)

# ✅ Normalize input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ✅ Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# ✅ Create dataset & DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# ✅ Define Bayesian Neural Network Model
class BayesianNN(nn.Module):
    def __init__(self, input_dim):
        super(BayesianNN, self).__init__()
        self.fc1 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=input_dim, out_features=128)
        self.relu = nn.ReLU()
        self.fc2 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=128, out_features=64)
        self.fc3 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=64, out_features=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# ✅ Initialize model, loss function, and optimizer
model = BayesianNN(X.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ Train the Bayesian Neural Network
def train_model(model, data_loader, criterion, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in data_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        epoch_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}")

train_model(model, data_loader, criterion, optimizer)

# ✅ Predict function with Uncertainty Estimation
def predict_with_uncertainty(f_model, X, n_samples=100):
    f_model.train()  # Keep model in training mode for Bayesian uncertainty estimation
    predictions = np.array([f_model(X).detach().numpy().flatten() for _ in range(n_samples)])
    mean_pred = predictions.mean(axis=0)
    std_pred = predictions.std(axis=0)  # Uncertainty estimation
    return mean_pred, std_pred

# ✅ Run predictions with uncertainty estimation
mean_pred, std_pred = predict_with_uncertainty(model, X_tensor)
df["Predicted d18O"] = mean_pred
df["Uncertainty"] = std_pred  # Store uncertainty in DataFrame

# ✅ Save predictions to CSV
df.to_csv("predicted_paleoclimate_data.csv", index=False)
print("\n✅ Predictions saved to 'predicted_paleoclimate_data.csv'.")

# ✅ Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(df["d18O"], df["Predicted d18O"])
print(f"📉 Mean Absolute Error (MAE): {mae:.4f}")

# ✅ Plot Actual vs Predicted δ18O with Uncertainty
plt.figure(figsize=(12, 6))
plt.plot(df["Depth"], df["Predicted d18O"], label="Predicted δ18O", color="blue", linewidth=2)
plt.fill_between(df["Depth"], df["Predicted d18O"] - df["Uncertainty"], df["Predicted d18O"] + df["Uncertainty"], 
                 color="blue", alpha=0.3, label="Uncertainty")
plt.xlabel("Depth (m)")
plt.ylabel("δ18O (‰)")
plt.grid(True)
plt.legend()
plt.show()

# ✅ Compute Forecast Error directly
df = df.sort_values("IceAge", ascending=False)
df["Forecast_Error"] = np.abs(df["IceAge"] - df["IceAge"].rolling(window=10, min_periods=1).mean())

# ✅ Plot Forecast Error Signal
plt.figure(figsize=(12, 5))
plt.plot(df["IceAge"], df["Forecast_Error"], color='black', linewidth=1.5, label="Forecast Error Signal")
plt.xlabel("Years Before Present")
plt.ylabel("Forecast Error")
plt.gca().invert_xaxis()
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()

# ✅ Plot δ18O Signal Graph
plt.figure(figsize=(12, 6))
plt.plot(df["Depth"], df["d18O"], label="Actual δ18O", color="blue", linewidth=2, alpha=0.8)
plt.xlabel("Years", fontsize=23, fontweight='bold')
plt.xlim(1625, 2025)
plt.xticks(range(1625, 2026, 25))
plt.ylabel("δ18O (‰)", fontsize=23, fontweight='bold')
plt.savefig("d18O_signal_graph.png", dpi=300)
plt.show()
