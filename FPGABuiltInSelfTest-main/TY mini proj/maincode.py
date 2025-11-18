# ====================================================
# LSTM MODEL FOR SOLDER JOINT REMAINING USEFUL LIFE (RUL) PREDICTION
# ====================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import random

# -------------------------------
# 0. Reproducibility
# -------------------------------
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

# -------------------------------
# 1. Generate realistic dummy data
# -------------------------------
num_samples = 5000
initial_rul = 50.0  # hours
sequence_length = 20  # window length
time = np.arange(0, num_samples) * 0.1  # time steps

# Degradation patterns
fault_count = np.clip(np.random.poisson(0.1 * time), 0, None)
fault_duration_avg = np.clip(np.random.normal(0.05 * time, 0.3), 0, None)
fault_resistance = np.clip(np.random.normal(0.1 * time, 0.2), 0, None)
temperature = np.random.normal(75, 5, num_samples)

# RUL decreases with minor noise
RUL = np.clip(initial_rul - 0.01 * (time ** 1.1) + np.random.normal(0, 0.2, num_samples), 0, None)

data = pd.DataFrame({
    'fault_count': fault_count,
    'fault_duration_avg': fault_duration_avg,
    'fault_resistance': fault_resistance,
    'time': time,
    'temperature': temperature,
    'RUL': RUL
})

# -------------------------------
# 2. Normalize features and RUL
# -------------------------------
feature_scaler = MinMaxScaler()
X_scaled = feature_scaler.fit_transform(data[['fault_count', 'fault_duration_avg', 'fault_resistance', 'time', 'temperature']])

rul_scaler = MinMaxScaler()
y_scaled = rul_scaler.fit_transform(data['RUL'].values.reshape(-1, 1))

# Combine for sequence creation
combined = np.hstack([X_scaled, y_scaled])  # shape (num_samples, 6)
combined_tensor = torch.tensor(combined, dtype=torch.float32)

# -------------------------------
# 3. Create sequences for LSTM
# -------------------------------
def create_sequences(tensor_data, seq_length=20):
    Xs = []
    ys = []
    n = tensor_data.shape[0]
    for i in range(n - seq_length):
        seq_X = tensor_data[i:i + seq_length, :-1]  # features
        seq_y = tensor_data[i + seq_length, -1]     # target
        Xs.append(seq_X)
        ys.append(seq_y)
    Xs = torch.stack(Xs)
    ys = torch.stack(ys).unsqueeze(1)
    return Xs, ys

X_seq, y_seq = create_sequences(combined_tensor, seq_length=sequence_length)
print("Created sequences:", X_seq.shape, y_seq.shape)

# -------------------------------
# 4. Train/Val/Test split
# -------------------------------
num_sequences = X_seq.shape[0]
train_size = int(0.7 * num_sequences)
val_size = int(0.15 * num_sequences)
test_size = num_sequences - train_size - val_size

X_train = X_seq[:train_size]; y_train = y_seq[:train_size]
X_val = X_seq[train_size:train_size + val_size]; y_val = y_seq[train_size:train_size + val_size]
X_test = X_seq[train_size + val_size:]; y_test = y_seq[train_size + val_size:]

print("Shapes - train/val/test:", X_train.shape, X_val.shape, X_test.shape)

# -------------------------------
# 5. DataLoaders
# -------------------------------
batch_size = 64
train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)
test_ds = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# -------------------------------
# 6. Define LSTM model for regression
# -------------------------------
class SolderLSTM_RUL(nn.Module):
    def _init_(self, input_size=5, hidden_size=64, num_layers=2, output_size=1, bidirectional=False, dropout=0.0):
        super(SolderLSTM_RUL, self)._init_()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        fc_in = hidden_size * self.num_directions
        self.fc = nn.Linear(fc_in, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time step
        out = self.fc(out)
        return out

# -------------------------------
# 7. Setup device, model, loss, optimizer, scheduler
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SolderLSTM_RUL(input_size=5, hidden_size=64, num_layers=2, output_size=1).to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)

# -------------------------------
# 8. Training loop with validation
# -------------------------------
epochs = 100
best_val_loss = float('inf')
train_losses = []
val_losses = []

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        running_loss += loss.item() * xb.size(0)

    epoch_train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    # Validation
    model.eval()
    val_running = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            vloss = loss_fn(preds, yb)
            val_running += vloss.item() * xb.size(0)
    epoch_val_loss = val_running / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)
    scheduler.step(epoch_val_loss)

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), "best_solder_lstm.pth")

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Train MSE: {epoch_train_loss:.6f} | Val MSE: {epoch_val_loss:.6f}")

# Load best model
model.load_state_dict(torch.load("best_solder_lstm.pth"))

# -------------------------------
# 9. Evaluate on test set
# -------------------------------
model.eval()
y_preds = []
y_trues = []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        preds = model(xb).cpu().numpy().reshape(-1, 1)
        y_true = yb.cpu().numpy().reshape(-1, 1)
        y_preds.append(preds)
        y_trues.append(y_true)

y_preds = np.vstack(y_preds)
y_trues = np.vstack(y_trues)

# Inverse scaling
y_preds_orig = rul_scaler.inverse_transform(y_preds).squeeze()
y_trues_orig = rul_scaler.inverse_transform(y_trues).squeeze()

# Metrics
mae = mean_absolute_error(y_trues_orig, y_preds_orig)
rmse = math.sqrt(mean_squared_error(y_trues_orig, y_preds_orig))
print(f"\nTest MAE: {mae:.4f} | Test RMSE: {rmse:.4f}")

# -------------------------------
# 10. Plot results
# -------------------------------
plt.figure(figsize=(9, 4))
plt.plot(train_losses, label='Train MSE')
plt.plot(val_losses, label='Val MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(y_trues_orig[:400], label='Actual RUL')
plt.plot(y_preds_orig[:400], label='Predicted RUL')
plt.xlabel('Sequence Index (test)')
plt.ylabel('RUL (hours)')
plt.legend()
plt.title('LSTM Predicted vs Actual RUL (sample)')
plt.show()