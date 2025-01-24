import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime
from Misc_Functions import *
from tqdm import tqdm
import logging
import csv
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Mixed precision (if GPU is available)
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

# File paths and directories
# data_path = find_file("Deuteron_2_100_No_Noise_500K.csv")  # Type in file name here
data_path = r"J:\Users\Devin\Desktop\Spin Physics Work\ANN Github\NMR-Fermilab\Big_Data\Deuteron_2_100_No_Noise_500K.csv"
version = 'Deuteron_2_100_V5'  # Rename for each run
performance_dir = f"Model Performance/{version}"  # Directory for saving performance metrics
model_dir = f"Models/{version}"  # Directory for saving model weights
os.makedirs(performance_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

### Add a customized weight to bias lower values of polarization
class WeightedMSELoss(nn.Module):
    def __init__(self, threshold=0.005, weight_near_zero=2.0, weight_else=1.0):
        super(WeightedMSELoss, self).__init__()
        self.threshold = threshold
        self.weight_near_zero = weight_near_zero
        self.weight_else = weight_else

    def forward(self, y_pred, y_true):
        # Ensure y_pred and y_true have the same shape
        y_pred = y_pred.squeeze()  # Remove extra dimensions (e.g., [32, 1] -> [32])
        y_true = y_true.squeeze()  # Ensure y_true is also squeezed (if necessary)

        # Calculate squared errors
        errors = (y_pred - y_true) ** 2

        # Assign weights based on true values
        weights = torch.where(y_true < self.threshold, self.weight_near_zero, self.weight_else)

        # Apply weights to errors
        weighted_errors = errors * weights

        # Return mean weighted error
        return torch.mean(weighted_errors)

### Use evaluation metric that biases near zero
def evaluate_near_zero(model, test_loader, threshold=0.1):
    model.eval()
    near_zero_loss = 0
    near_zero_count = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch).squeeze()
            mask = y_batch < threshold  # Only consider values near 0
            if mask.any():
                near_zero_loss += torch.mean((y_pred[mask] - y_batch[mask]) ** 2).item()
                near_zero_count += mask.sum().item()

    if near_zero_count > 0:
        near_zero_loss /= near_zero_count
    else:
        near_zero_loss = float('nan')

    return near_zero_loss

# Define the model
class Polarization(nn.Module):
    def __init__(self, input_dim):
        super(Polarization, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 1),
            # nn.Linear()
        )


    def forward(self, x):
        return self.layers(x)

# Load and preprocess data
print("Getting data...")
data = pd.read_csv(data_path)
print(f"Data found at: {data_path}")

val_fraction = 0.2
test_fraction = 0.1

train_split_index = int(len(data) * (1 - val_fraction - test_fraction))
val_split_index = int(len(data) * (1 - test_fraction))

train_data = data.iloc[:train_split_index]
val_data = data.iloc[train_split_index:val_split_index]
test_data = data.iloc[val_split_index:]

target_variable = "P"
X_train, y_train = train_data.drop([target_variable, 'SNR'], axis=1).values, train_data[target_variable].values
X_val, y_val = val_data.drop([target_variable, 'SNR'], axis=1).values, val_data[target_variable].values
X_test, y_test = test_data.drop([target_variable, 'SNR'], axis=1).values, test_data[target_variable].values

# Normalize X values to [0,1]
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Add Gaussian noise
noise_std = 0.05
X_train += np.random.normal(0, noise_std, X_train.shape)
X_val += np.random.normal(0, noise_std, X_val.shape)
X_test += np.random.normal(0, noise_std, X_test.shape)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

BATCH_SIZE = 32

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize model, optimizer, and loss function
model = Polarization(X_train.shape[1]).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=5e-3)
criterion = nn.MSELoss()
# criterion = WeightedMSELoss(threshold=0.1, weight_near_zero=2.0, weight_else=1.0)

# Create CSV file for logging
csv_file = os.path.join(performance_dir, f'training_metrics_{version}.csv')
with open(csv_file, mode='w', newline='') as file:
    writer_csv = csv.writer(file)
    writer_csv.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Learning Rate'])

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=200, patience=10):
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    epochs_no_improve = 0  # Counter for epochs without improvement

    # Initialize learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)  # Shape: [batch_size, 1]
            loss = criterion(y_pred.squeeze(), y_batch)  # Fix shape mismatch
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        train_losses.append(epoch_train_loss / len(train_loader))

        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)  # Shape: [batch_size, 1]
                loss = criterion(y_pred.squeeze(), y_batch)  # Fix shape mismatch
                epoch_val_loss += loss.item()
        val_losses.append(epoch_val_loss / len(val_loader))

        # Step the learning rate scheduler
        scheduler.step(val_losses[-1])

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Save metrics to CSV
        with open(csv_file, mode='a', newline='') as file:
            writer_csv = csv.writer(file)
            writer_csv.writerow([epoch+1, train_losses[-1], val_losses[-1], current_lr])

        # Print progress
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {train_losses[-1]:.4f}, "
              f"Val Loss: {val_losses[-1]:.4f}, "
              f"Learning Rate: {current_lr:.6f}")

        # Check for improvement in validation loss
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            epochs_no_improve = 0  # Reset counter
            # Save the best model
            torch.save(model.state_dict(), os.path.join(model_dir, f'best_model_{version}.pt'))
        else:
            epochs_no_improve += 1  # Increment counter

        # Early stopping check
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}. No improvement for {patience} epochs.")
            break

    return train_losses, val_losses

# Train the model
print("Starting training...")
train_losses, val_losses = train_model(model, train_loader, val_loader, optimizer, criterion)

near_zero_mse = evaluate_near_zero(model, test_loader, threshold=0.1)
print(f"MSE for values near 0: {near_zero_mse:.4f}")

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid()

loss_plot_path = os.path.join(performance_dir, f'{version}_Loss_Plot.png')
plt.savefig(loss_plot_path, dpi=600)
print(f"Loss plot saved to {loss_plot_path}")

# Save model summary
model_summary_path = os.path.join(performance_dir, 'model_summary.txt')
with open(model_summary_path, 'w') as f:
    f.write(str(model))
print(f"Model summary saved to {model_summary_path}")

# Save the final model
final_model_path = os.path.join(model_dir, f'final_model_{version}.pt')
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved to {final_model_path}")

# Evaluate on test data
model.eval()
test_loss = 0
y_test_pred = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        y_pred = model(X_batch).squeeze()
        loss = criterion(y_pred, y_batch)
        test_loss += loss.item()
        y_test_pred.extend(y_pred.cpu().numpy())

test_loss /= len(test_loader)
y_test_pred = np.array(y_test_pred)
residuals = y_test.cpu().numpy() - y_test_pred

# Save test results
test_results_df = pd.DataFrame({
    'Actual': y_test.cpu().numpy(),
    'Predicted': y_test_pred,
    'Residuals': residuals
})

event_results_file = os.path.join(performance_dir, f'test_event_results_{version}.csv')
test_results_df.to_csv(event_results_file, index=False)
print(f"Test results saved to {event_results_file}")

# Calculate per-sample MSE losses
individual_losses = np.square(residuals)  # MSE per sample
loss_results_df = pd.DataFrame({
    'Polarization': y_test.cpu().numpy(),
    'Loss': individual_losses
})

loss_results_file = os.path.join(performance_dir, f'per_sample_loss_{version}.csv')
loss_results_df.to_csv(loss_results_file, index=False)
print(f"Per-sample loss results saved to {loss_results_file}")

# Plot Polarization vs. Loss (MSE)
plt.figure(figsize=(10, 6))
plt.scatter(y_test.cpu().numpy(), individual_losses, alpha=0.6, color='blue', edgecolors='w', s=50)
plt.xlabel('Polarization (True Values)', fontsize=14)
plt.ylabel('Loss (MSE)', fontsize=14)
plt.title('Polarization vs. Loss (MSE)', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

polarization_loss_plot_path = os.path.join(performance_dir, f'{version}_Polarization_vs_Loss.png')
plt.savefig(polarization_loss_plot_path, dpi=600)
print(f"Polarization vs. Loss plot saved to {polarization_loss_plot_path}")

# Plot Loss Difference (Training - Validation)
loss_diff = np.array(train_losses) - np.array(val_losses)
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(loss_diff) + 1), loss_diff, marker='o', label="Loss Difference (Training - Validation)")
plt.axhline(0, color='red', linestyle='--', linewidth=1, label="Zero Difference")
plt.xlabel("Epoch")
plt.ylabel("Loss Difference")
plt.title("Difference Between Training and Validation Loss")
plt.legend()
plt.grid()

loss_diff_plot_path = os.path.join(performance_dir, f'{version}_Loss_Diff_Plot.png')
plt.savefig(loss_diff_plot_path, dpi=600)
print(f"Loss difference plot saved to {loss_diff_plot_path}")

# Plot histograms of residuals
residuals_mean = np.mean(residuals)
residuals_std = np.std(residuals)

fig = plt.figure(figsize=(16, 6))
gs = fig.add_gridspec(1, 2)

ax1 = fig.add_subplot(gs[0])
plot_histogram(
    residuals * 100,
    'Histogram of Polarization Difference',
    'Difference in Polarization',
    'Count',
    'red',
    ax1,
    plot_norm=False
)

ax2 = fig.add_subplot(gs[1])
plot_histogram(
    np.abs(residuals * 100),
    'Histogram of Mean Absolute Error',
    'Mean Absolute Error',
    '',
    'orange',
    ax2,
    plot_norm=False
)

ax1.text(0.5, -0.2, '(a)', transform=ax1.transAxes, ha='center', fontsize=16, weight='bold')
ax2.text(0.5, -0.2, '(b)', transform=ax2.transAxes, ha='center', fontsize=16, weight='bold')

plt.tight_layout()
plt.subplots_adjust(bottom=0.2)

output_path = os.path.join(performance_dir, f'{version}_Histograms.png')
fig.savefig(output_path, dpi=600)
print(f"Histograms plotted in {output_path}!")

# Save test summary results
test_summary_results = {
    'Date': [str(datetime.now())],
    'Test Loss': [test_loss],
    'Test MSE': [np.mean(individual_losses)]
}

summary_results_df = pd.DataFrame(test_summary_results)
summary_results_file = os.path.join(performance_dir, f'test_summary_results_{version}.csv')
summary_results_df.to_csv(summary_results_file, index=False)

print(f"Test Loss: {test_loss}, Test MSE: {np.mean(individual_losses)}")
print(f"Test summary results saved to {summary_results_file}")