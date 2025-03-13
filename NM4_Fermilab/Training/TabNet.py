import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Custom_Scripts.Misc_Functions import *
from Plotting.Plot_Script import *
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, PowerTransformer
from pytorch_tabnet.tab_model import TabNetRegressor
import joblib
import random
from scipy.stats import norm
from pytorch_tabnet.callbacks import Callback
import csv

class CustomCSVLogger(Callback):
    def __init__(self, filepath, separator=",", append=False):
        self.filepath = filepath
        self.separator = separator
        self.append = append
        self.writer = None
        self.keys = None
        self.file = None
        
    def on_train_begin(self, logs=None):
        if self.append:
            mode = 'a'
        else:
            mode = 'w'
            
        self.file = open(self.filepath, mode)
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        if not self.keys:
            self.keys = sorted(logs.keys())
            self.writer = csv.DictWriter(self.file, 
                                         fieldnames=['epoch'] + self.keys)
            self.writer.writeheader()
            
        logs_dict = {k: float(logs[k]) for k in self.keys}
        logs_dict['epoch'] = epoch
        self.writer.writerow(logs_dict)
        self.file.flush()
    
    def on_train_end(self, logs=None):
        self.file.close()
        self.writer = None

# Set seeds for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seeds(42)

# Environment setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# File paths and versioning
data_path = find_file("Deuteron_Oversampled_500K.csv") 
version = f'Deuteron_TabNet_V1_Oversampled_Low'
performance_dir = f"Model_Performance/{version}"
model_dir = f"Models/{version}"
os.makedirs(performance_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Data loading and preprocessing
print("Loading data...")
try:
    data = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"Error: File {data_path} not found.")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
print(f"Data loaded successfully from {data_path}")
# Data splitting
print("Splitting data...")
data['P_bin'] = pd.qcut(data['P'], q=10, labels=False)
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42, stratify=data['P_bin'])
val_data, test_data = train_test_split(temp_data, test_size=1/3, random_state=42, stratify=temp_data['P_bin'])
print("Data split successfully")
# Prepare features and targets
print("Preparing features and targets...")
X_train = train_data.drop(columns=["P", 'SNR', 'P_bin']).values.astype(np.float32)
y_train = train_data["P"].values.astype(np.float32).reshape(-1, 1)
X_val = val_data.drop(columns=["P", 'SNR', 'P_bin']).values.astype(np.float32)
y_val = val_data["P"].values.astype(np.float32).reshape(-1, 1)
X_test = test_data.drop(columns=["P", 'SNR', 'P_bin']).values.astype(np.float32)
y_test = test_data["P"].values.astype(np.float32).reshape(-1, 1)
print("Features and targets prepared successfully")
# Normalization
print("Normalizing data...")
scaler1 = RobustScaler().fit(X_train)
X_train_robust = scaler1.transform(X_train)
X_val_robust = scaler1.transform(X_val)
X_test_robust = scaler1.transform(X_test)
print("Data normalized successfully")
# scaler2 = PowerTransformer(method='yeo-johnson').fit(X_train_robust)
# X_train = scaler2.transform(X_train_robust).astype(np.float32)
# X_val = scaler2.transform(X_val_robust).astype(np.float32)
# X_test = scaler2.transform(X_test_robust).astype(np.float32)
# print("Data transformed successfully")  
# Save scalers
print("Saving scalers...")
joblib.dump(scaler1, os.path.join(model_dir, 'scaler1.pkl'))
# joblib.dump(scaler2, os.path.join(model_dir, 'scaler2.pkl'))
print("Scalers saved successfully")
# Training parameters
max_epochs = 500
patience = 20
batch_size = 64

# ### Callbacks ###
custom_csv_logger = CustomCSVLogger(os.path.join(performance_dir, 'training_log.csv'))


# TabNet model configuration
print("Configuring TabNet model...")
model = TabNetRegressor(
    input_dim=500,
    output_dim=1,
    n_d=64,  
    n_a=64,  
    n_steps=5,  
    gamma=1.5,  
    lambda_sparse=1e-3,  
    mask_type='entmax',  
    optimizer_fn=torch.optim.AdamW,
    optimizer_params=dict(lr=5e-4, weight_decay=1e-4),  
    device_name=device,
    seed=42,
    verbose=1,
    momentum=0.3,  
    clip_value=2.0,  
    scheduler_fn=torch.optim.lr_scheduler.OneCycleLR,  
    scheduler_params={
        'max_lr': 5e-4,
        'total_steps': 1000,  
        'pct_start': 0.3,
        'div_factor': 25.0,
        'final_div_factor': 10000.0,
    }
)
print("TabNet model configured successfully")
# Train the model
def weighted_mse_loss(pred, target):
    error = pred - target
    return torch.mean(error * error * torch.exp(-0.1 * torch.abs(error)))

# Enhanced training parameters
max_epochs = 1000  
patience = 30  
batch_size = 64  

print("Training TabNet model...")
model.fit(
    X_train=X_train,
    y_train=y_train,
    eval_set=[(X_val, y_val)],
    eval_name=['val'],
    eval_metric=['mae', 'rmse'],
    max_epochs=max_epochs,
    patience=patience,
    batch_size=batch_size,
    virtual_batch_size=16,  
    num_workers=8,  
    drop_last=False,
    loss_fn=weighted_mse_loss,  
    from_unsupervised=None,
    augmentations=None,
    weights=0,
    warm_start=False,
    callbacks=[custom_csv_logger])
print("Model trained successfully")
# # Save model
print("Saving model...")
model.save_model(os.path.join(model_dir, 'tabnet_model'))
print("Model saved successfully")
# Load best model for evaluation
print("Loading best model...")
model.load_model(os.path.join(model_dir, 'tabnet_model.zip'))
print("Best model loaded successfully")
# Evaluate on test set
print("Evaluating model...")
y_pred = model.predict(X_test)
residuals = y_test.flatten() - y_pred.flatten()
print("Model evaluated successfully")
# Calculate metrics
mae = np.mean(np.abs(residuals))*100
rmse = np.sqrt(np.mean(residuals**2))*100
max_error = np.max(np.abs(residuals))*100
rpe = np.abs((y_test.flatten() - y_pred.flatten()) / np.maximum(np.abs(y_test.flatten()), 1e-10)) * 100
median_rpe = np.median(rpe)
p95_rpe = np.percentile(rpe, 95)

print(f"\nTest Set Metrics:")
print(f"MAE: {mae:.8f}")
print(f"RMSE: {rmse:.8f}")
print(f"Max Error: {max_error:.8f}")
print(f"Median RPE: {median_rpe:.2f}%")
print(f"95th Percentile RPE: {p95_rpe:.2f}%")

# Save metrics
metrics_dict = {
    'mae': mae,
    'rmse': rmse,
    'max_error': max_error,
    'median_rpe': median_rpe,
    'p95_rpe': p95_rpe
}

with open(os.path.join(performance_dir, 'test_metrics.txt'), 'w') as f:
    for metric, value in metrics_dict.items():
        f.write(f"{metric}: {value}\n")

# Save predictions
test_results = pd.DataFrame({
    'Actual': y_test.flatten(),
    'Predicted': y_pred.flatten(),
    'Residuals': residuals,
    'RPE': rpe
})
test_results.to_csv(os.path.join(performance_dir, 'test_predictions.csv'), index=False)

print(f"Evaluation complete. Results saved to {performance_dir}")

plot_rpe_and_residuals(y_test, y_pred, performance_dir, version)

plot_training_history(model.history, performance_dir, version)

print(f"Plots saved to {performance_dir}")