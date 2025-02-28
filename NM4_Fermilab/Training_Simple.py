import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, regularizers, initializers, optimizers
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from Misc_Functions import *
import random
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
from RPE_Histograms import analyze_model_errors
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor

### Let's set a specific seed for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Environment configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
tf.config.optimizer.set_jit(True)

# For high precision tasks, we'll use float64 throughout
tf.keras.mixed_precision.set_global_policy('float64')
tf.keras.backend.set_floatx('float64')

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"Using GPU: {physical_devices[0]}")

# File paths and versioning
data_path = find_file("Deuteron_Low_No_Noise_500K.csv")  
version = 'Deuteron_Low_Noise_HighPrecision_V3'  
performance_dir = f"Model Performance/{version}"  
model_dir = f"Models/{version}"  
os.makedirs(performance_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
# Load and prepare data
print("Loading and preparing data...")
data = pd.read_csv(data_path)

# Scale the input features
feature_scaler = MinMaxScaler()
X = data.drop(columns=["P", 'SNR']).astype('float64').values
X_scaled = feature_scaler.fit_transform(X)

# Do not scale the target values
y = data["P"].astype('float64')

# Split data
print("Splitting data...")
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=1/3, random_state=42)

# Define the target value and range
target_value = 0.0005
range_margin = 0.0005  # Define a margin around the target value
multiplication_factor = 5  # Define how many times to duplicate the augmented data

# Filter the training data for values around 0.05%
mask = (train_data["P"] >= target_value - range_margin) & (train_data["P"] <= target_value + range_margin)
augmented_data = train_data[mask]

# Duplicate the filtered data to increase representation
augmented_data = pd.concat([augmented_data] * multiplication_factor, ignore_index=True)
train_data = pd.concat([train_data, augmented_data], ignore_index=True)

# Prepare feature matrices
X_train = train_data.drop(columns=["P", 'SNR']).astype('float64').values
y_train = train_data["P"].astype('float64')
X_val = val_data.drop(columns=["P", 'SNR']).astype('float64').values
y_val = val_data["P"].astype('float64')
X_test = test_data.drop(columns=["P", 'SNR']).astype('float64').values
y_test = test_data["P"].astype('float64')

# Scale the training, validation, and test sets for features only
X_train_scaled = feature_scaler.fit_transform(X_train)  # Fit and transform on training data
X_val_scaled = feature_scaler.transform(X_val)  # Transform validation data
X_test_scaled = feature_scaler.transform(X_test)  # Transform test data

# Free up memory
del data, train_data, val_data, test_data
import gc
gc.collect()

# Create and train the XGBoost model with the custom loss function
xgb_model = XGBRegressor(
    n_estimators=5000,  # Reduced number of boosting stages for simplicity
    learning_rate=0.01,  # Increased learning rate for faster convergence
    max_depth=7,  # Reduced maximum depth to simplify the model
    random_state=42,
    eval_metric='rmse',  # Metric to evaluate
    objective='reg:squarederror',  # Objective function for regression
    tree_method='gpu_hist',  # Use GPU for training
    early_stopping_rounds=50,  # Reduced rounds for early stopping
    alpha=0.1,  # Removed L1 regularization term
    lambda_=0.1,  # Removed L2 regularization term
    subsample=0.8  # Use 80% of the data for each boosting round
)

# Fit the model with the custom loss function
xgb_model.fit(X_train_scaled, y_train,
              eval_set=[(X_val_scaled, y_val)],  # Specify the validation set
              verbose=True)
# Make predictions
y_val_pred = xgb_model.predict(X_val_scaled)
y_test_pred = xgb_model.predict(X_test_scaled)

y_val *= 100
y_test *= 100
y_val_pred *= 100
y_test_pred *= 100

# Evaluate the model
val_mae = mean_absolute_error(y_val, y_val_pred)
val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)

test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)

print(f"Validation MAE: {val_mae:.6f}")
print(f"Validation RMSE: {val_rmse:.6f}")
print(f"Test MAE: {test_mae:.6f}")
print(f"Test RMSE: {test_rmse:.6f}")

# Calculate relative percent error
relative_percent_error = np.abs((y_test - y_test_pred) / y_test) * 100

# Create a directory for saving model performance plots
output_dir = 'Model Performance/XGBoost_PerformanceV2'
os.makedirs(output_dir, exist_ok=True)

# Plotting the training and validation losses
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, len(xgb_model.evals_result()['validation_0']['rmse']) + 1), 
         xgb_model.evals_result()['validation_0']['rmse'], label='Validation Loss', color='orange')
plt.title('Validation Loss Over Iterations', fontsize=16)
plt.xlabel('Number of Iterations', fontsize=14)
plt.ylabel('Root Mean Squared Error', fontsize=14)
plt.legend()
plt.grid()
# Save the figure
plt.savefig(os.path.join(output_dir, 'validation_loss.png'))
plt.close()  # Close the figure to free up memory

# Histogram of relative percent error
plt.figure(figsize=(10, 6))
plt.hist(relative_percent_error, bins=30, color='skyblue', edgecolor='black')
plt.title('Histogram of Relative Percent Error', fontsize=16)
plt.xlabel('Relative Percent Error (%)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid()
# Save the figure
plt.savefig(os.path.join(output_dir, 'relative_percent_error_histogram.png'))
plt.close()  # Close the figure to free up memory

# Create a figure with two subplots
plt.figure(figsize=(12, 6))

# First subplot: Original scatter plot
plt.subplot(1, 2, 1)
plt.scatter(y_test, relative_percent_error, alpha=0.5)
plt.title('Relative Percent Error as a Function of Polarization', fontsize=16)
plt.xlabel('Polarization', fontsize=14)
plt.ylabel('Relative Percent Error (%)', fontsize=14)
plt.grid()

# Second subplot: Focused scatter plot for polarizations < 0.06%
plt.subplot(1, 2, 2)
mask = y_test < 0.06  # Create a mask for polarizations less than 0.06%
plt.scatter(y_test[mask], relative_percent_error[mask], alpha=0.5, color='orange')
plt.title('Relative Percent Error for Polarization < 0.06%', fontsize=16)
plt.xlabel('Polarization', fontsize=14)
plt.ylabel('Relative Percent Error (%)', fontsize=14)
plt.grid()

# Save the figure
plt.savefig(os.path.join(output_dir, 'relative_percent_error_vs_polarization_combined.png'))
plt.close()  # Close the figure to free up memory