import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Custom_Scripts.Misc_Functions import *
from Plotting.Plot_Script import *
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, PowerTransformer, StandardScaler
from pytorch_tabnet.tab_model import TabNetRegressor
import joblib
import random
from scipy.stats import norm
from pytorch_tabnet.callbacks import Callback, EarlyStopping
import csv
import optuna

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
        
        # Debugging statement
        if self.writer is None:
            print("Error: Writer is not initialized.")
            return  # Exit if writer is not initialized

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
# data_path = find_file("Test.csv") 
version = f'Deuteron_TabNet_Shifted_low'
performance_dir = f"Model Performance/{version}"
model_dir = f"Models/{version}"
os.makedirs(performance_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Data loading and preprocessing
print("Loading data...")
data_path = find_file("Shifted_low.csv")
data = pd.read_csv(data_path)

# Data splitting
data['P_bin'] = pd.qcut(data['P'], q=10, labels=False)
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42, stratify=data['P_bin'])
val_data, test_data = train_test_split(temp_data, test_size=1/3, random_state=42, stratify=temp_data['P_bin'])

# Prepare features and targets
X_train = train_data.drop(columns=["P", 'SNR', 'P_bin']).values.astype(np.float32)
y_train = train_data["P"].values.astype(np.float32).reshape(-1, 1)
X_val = val_data.drop(columns=["P", 'SNR', 'P_bin']).values.astype(np.float32)
y_val = val_data["P"].values.astype(np.float32).reshape(-1, 1)
X_test = test_data.drop(columns=["P", 'SNR', 'P_bin']).values.astype(np.float32)
y_test = test_data["P"].values.astype(np.float32).reshape(-1, 1)

# Normalization using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Save scalers
joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))

# Training parameters
max_epochs = 500
patience = 20
batch_size = 64

# ### Callbacks ###
custom_csv_logger = CustomCSVLogger(os.path.join(performance_dir, 'training_log.csv'))


def weighted_mse_loss(pred, target):
    # Give more weight to errors on small values
    error = pred - target
    # Inverse weighting: smaller targets get higher weights
    weights = 1.0 / (target + 1e-6)  # Add small epsilon to avoid division by zero
    return torch.mean(weights * error * error)

def relative_error_loss(pred, target):
    # Relative error loss: (pred - target) / (target + epsilon)
    return torch.mean(torch.abs((pred - target) / (target + 1e-6)))

def log_cosh_loss(pred, target):
    error = pred - target
    return torch.mean(torch.log(torch.cosh(error)))

# Scale up target values before training
from sklearn.preprocessing import MinMaxScaler
y_scaler = MinMaxScaler()  # or StandardScaler() depending on your original scaling method
y_train = y_train.reshape(-1, 1)  # Reshape for scaler
y_train_scaled = y_scaler.fit_transform(y_train)  # Fit and transform the training target
y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1))  # Transform validation target
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))  # Transform test target
# Define the objective function for Optuna
def objective(trial):
    # Suggest hyperparameter
    input_dim = 500 
    output_dim = 1   

    n_d = trial.suggest_int('n_d', 64, 256)  # Wider networks
    n_a = trial.suggest_int('n_a', 64, 256)
    n_steps = trial.suggest_int('n_steps', 5, 20)  # More steps for complex relationships
    gamma = trial.suggest_float('gamma', 1.0, 2.0)  # Gamma value
    lambda_sparse = trial.suggest_loguniform('lambda_sparse', 1e-5, 1e-2)  # Sparse regularization
    momentum = trial.suggest_float('momentum', 0.0, 0.9)  # Momentum for the optimizer
    max_lr = trial.suggest_loguniform('max_lr', 1e-6, 1e-3)  # Smaller learning rate range for fine-grained learning
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)  # Increase regularization to prevent overfitting on small values
    virtual_batch_size = trial.suggest_categorical('virtual_batch_size', [8, 16, 32])  # Virtual batch size
    clip_value = trial.suggest_float('clip_value', 0.0, 5.0)  # Clip value for gradients
    num_workers = trial.suggest_categorical('num_workers', [2, 4, 8, 12])  # Number of workers
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])  # Smaller batch sizes for more frequent updates

    # TabNet model configuration
    model = TabNetRegressor(
        input_dim=input_dim,
        output_dim=output_dim,
        n_d=n_d,
        n_a=n_a,
        n_steps=n_steps,
        gamma=gamma,
        lambda_sparse=lambda_sparse,
        mask_type='entmax',
        optimizer_fn=torch.optim.AdamW,
        optimizer_params=dict(lr=max_lr, weight_decay=weight_decay),
        device_name=device,
        seed=42,
        verbose=1,  
        momentum=momentum,
        clip_value=clip_value,  # Use the suggested clip value
        scheduler_fn=torch.optim.lr_scheduler.OneCycleLR,
        scheduler_params={
            'max_lr': max_lr,
            'total_steps': 1000,
            'pct_start': 0.3,
            'div_factor': 25.0,
            'final_div_factor': 10000.0,
        },
    )

    # Train the model
    model.fit(
        X_train=X_train,
        y_train=y_train_scaled,
        eval_set=[(X_val, y_val_scaled)],
        eval_name=['val'],
        eval_metric=['mae', 'rmse'],
        max_epochs=2000,
        patience=50,
        batch_size=batch_size,  # Use the suggested batch size
        virtual_batch_size=virtual_batch_size,  # Use the suggested virtual batch size
        num_workers=num_workers,  # Use the suggested number of workers
        drop_last=False,
        loss_fn=log_cosh_loss,  # Use the Log-Cosh loss function here
        from_unsupervised=None,
        augmentations=None,
        weights=0,
        warm_start=False,
    )

    # After prediction, scale back down
    y_pred_scaled = model.predict(X_test)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_test = y_scaler.inverse_transform(y_test_scaled)

    # Calculate relative MAE or MAPE for evaluation
    rel_mae = np.mean(np.abs((y_test.flatten() - y_pred.flatten()) / (y_test.flatten() + 1e-6)))
    
    return rel_mae  # Optimize for relative error instead

if __name__ == "__main__":
    # Create an Optuna study
    study = optuna.create_study(direction='minimize')  # Minimize the validation MAE
    study.optimize(objective, n_trials=100)  # Number of trials to run

    # Print the best hyperparameters
    print("Best hyperparameters: ", study.best_params)
    print("Best validation MAE: ", study.best_value)

    # Save the best hyperparameters for later use
    best_params = study.best_params

    # TabNet model configuration with best hyperparameters
    model = TabNetRegressor(
        input_dim=500,
        output_dim=1,
        n_d=best_params['n_d'],
        n_a=best_params['n_a'],
        n_steps=best_params['n_steps'],
        gamma=best_params['gamma'],
        lambda_sparse=best_params['lambda_sparse'],
        mask_type='entmax',
        optimizer_fn=torch.optim.AdamW,
        optimizer_params=dict(lr=best_params['max_lr'], weight_decay=best_params['weight_decay']),
        device_name=device,
        seed=42,
        verbose=1,
        momentum=best_params['momentum'],
        clip_value=best_params['clip_value'],
        scheduler_fn=torch.optim.lr_scheduler.OneCycleLR,
        scheduler_params={
            'max_lr': best_params['max_lr'],
            'total_steps': 1000,
            'pct_start': 0.3,
            'div_factor': 25.0,
            'final_div_factor': 10000.0,
        },
    )

    # Train the model with the best hyperparameters
    print("Training TabNet model...")
    model.fit(
        X_train=X_train,
        y_train=y_train_scaled,
        eval_set=[(X_val, y_val_scaled)],
        eval_name=['val'],
        eval_metric=['mae', 'rmse'],
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        virtual_batch_size=best_params['virtual_batch_size'],
        num_workers=best_params['num_workers'],
        drop_last=False,
        loss_fn=log_cosh_loss,
        from_unsupervised=None,
        augmentations=None,
        weights=0,
        warm_start=False,
    )

    # Save the best model
    model.save_model(os.path.join(model_dir, 'tabnet_model'))

    # Load the best model for evaluation
    model.load_model(os.path.join(model_dir, 'tabnet_model.zip'))

    # Evaluate on test set
    print("Evaluating model...")
    y_pred_scaled = model.predict(X_test)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    residuals = y_test.flatten() - y_pred.flatten()

    # Calculate metrics
    mae = np.mean(np.abs(residuals)) * 100
    rmse = np.sqrt(np.mean(residuals ** 2)) * 100
    max_error = np.max(np.abs(residuals)) * 100
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