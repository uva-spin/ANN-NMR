import os
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bar and ETA
import time
from Misc_Functions import *

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Read data
data_path = find_file("Sample_Data_V2_1M.csv")
chunk_size = 10000

print("Reading Data...")
chunks = pd.read_csv(data_path, chunksize=chunk_size)

df_list = []
for chunk in chunks:
    df_list.append(chunk)
df = pd.concat(df_list)

print("Data Read and Concatenated...")

# Split data
def split_data(X, y, split=0.1):
    temp_idx = np.random.choice(len(y), size=int(len(y) * split), replace=False)
    
    tst_X = X.iloc[temp_idx].reset_index(drop=True)
    trn_X = X.drop(temp_idx).reset_index(drop=True)
    
    tst_y = y.iloc[temp_idx].reset_index(drop=True)
    trn_y = y.drop(temp_idx).reset_index(drop=True)
    
    return trn_X, tst_X, trn_y, tst_y

y = df['Area']
x = df.drop(['Area', 'SNR'], axis=1)

train_X, test_X, train_y, test_y = split_data(x, y)

version = 'v7'
performance_dir = f"Model Performance/{version}"
model_dir = f"Models/{version}"
os.makedirs(performance_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

train_X_tensor = torch.tensor(train_X.values, dtype=torch.float32).to(device)
train_y_tensor = torch.tensor(train_y.values, dtype=torch.float32).view(-1, 1).to(device)
test_X_tensor = torch.tensor(test_X.values, dtype=torch.float32).to(device)
test_y_tensor = torch.tensor(test_y.values, dtype=torch.float32).view(-1, 1).to(device)

# Hyperparameters
batch_size = 64
learning_rate = 1e-4
num_epochs = 50
log_every_n_epochs = 5  # Log progress every n epochs


train_dataset = TensorDataset(train_X_tensor, train_y_tensor)
test_dataset = TensorDataset(test_X_tensor, test_y_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define model architecture
class CustomModel(nn.Module):
    def __init__(self, input_size, hidden_layers, units_per_layer, activation_funcs, dropouts):
        super(CustomModel, self).__init__()
        layers = []
        for i in range(hidden_layers):
            layers.append(nn.BatchNorm1d(input_size if i == 0 else units_per_layer[i - 1]))
            layers.append(nn.Linear(input_size if i == 0 else units_per_layer[i - 1], units_per_layer[i]))
            layers.append(getattr(nn, activation_funcs[i])())
            layers.append(nn.Dropout(dropouts[i]))
        layers.append(nn.Linear(units_per_layer[-1], 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Evaluation function
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(loader)

# Hyperparameter tuning using a custom loop
def hyperparameter_tuning():
    best_loss = float('inf')
    best_model = None
    for hidden_layers in range(2, 11):  # Loop over number of layers
        units_per_layer = [np.random.randint(64, 1024) for _ in range(hidden_layers)]
        activation_funcs = np.random.choice(['ReLU', 'ReLU6'], size=hidden_layers)
        dropouts = np.random.uniform(0.1, 0.5, hidden_layers)

        model = CustomModel(input_size=train_X.shape[1], hidden_layers=hidden_layers,
                            units_per_layer=units_per_layer, activation_funcs=activation_funcs, dropouts=dropouts).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(20):  # Short training for hyperparameter search
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        val_loss = evaluate(model, test_loader, criterion)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model

    return best_model

# Mixed precision training
scaler = torch.cuda.amp.GradScaler()

# Hyperparameter tuning
best_model = hyperparameter_tuning()

# Final model training with tqdm progress bar
criterion = nn.MSELoss()
optimizer = optim.Adam(best_model.parameters(), lr=learning_rate)

train_losses, val_losses = [], []
best_val_loss = float('inf')
start_time = time.time()


print("Starting training...")
for epoch in range(num_epochs):
    best_model.train()
    total_loss = 0
    
    # Training loop with tqdm for progress and ETA
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit='batch') as pbar:
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():  # Enable mixed precision training
                outputs = best_model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            pbar.update(1)

    train_loss = total_loss / len(train_loader)
    val_loss = evaluate(best_model, test_loader, criterion)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    # Save the best model based on validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(best_model.state_dict(), os.path.join(model_dir, f'best_model.pth'))

    # Log progress and ETA
    if (epoch + 1) % log_every_n_epochs == 0 or epoch == num_epochs - 1:
        elapsed_time = time.time() - start_time
        avg_time_per_epoch = elapsed_time / (epoch + 1)
        remaining_time = avg_time_per_epoch * (num_epochs - epoch - 1)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        print(f"Elapsed Time: {elapsed_time:.2f}s, Estimated Remaining Time: {remaining_time:.2f}s")

# Save final model
torch.save(best_model.state_dict(), os.path.join(model_dir, 'final_model.pth'))

# Plot the losses
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.savefig(os.path.join(performance_dir, f'loss_plot_{version}.png'))
