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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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


data_path = find_file("Deuteron_2_100_No_Noise_500K.csv")  
data_path_lower = find_file("Deuteron_0_2_No_Noise_500K.csv")  
data_path_lowest = find_file("Deuteron_Low_No_Noise_500K.csv")  
version = 'Deuteron_All_No_Noise_XGBoost_V1'  
performance_dir = f"Model Performance/{version}"  
model_dir = f"Models/{version}"  
os.makedirs(performance_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

print("Loading and preparing data...")
data = pd.read_csv(data_path)
data_lower = pd.read_csv(data_path_lower)
data_lowest = pd.read_csv(data_path_lowest)

data = pd.concat([data, data_lower, data_lowest], ignore_index=True)

feature_scaler = MinMaxScaler()
X = data.drop(columns=["P", 'SNR']).astype('float64').values
X_scaled = feature_scaler.fit_transform(X)

y = data["P"].astype('float64')

print("Splitting data...")
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=1/3, random_state=42)

target_value = 0.0005
range_margin = 0.00005  
multiplication_factor = 5  # Define how many times to duplicate the augmented data

# Filter the training data for values around 0.05%
mask = (train_data["P"] >= target_value - range_margin) & (train_data["P"] <= target_value + range_margin)
augmented_data = train_data[mask]

# Duplicate the filtered data to increase representation
augmented_data = pd.concat([augmented_data] * multiplication_factor, ignore_index=True)
train_data = pd.concat([train_data, augmented_data], ignore_index=True)

X_train = train_data.drop(columns=["P", 'SNR']).astype('float64').values
y_train = train_data["P"].astype('float64')
X_val = val_data.drop(columns=["P", 'SNR']).astype('float64').values
y_val = val_data["P"].astype('float64')
X_test = test_data.drop(columns=["P", 'SNR']).astype('float64').values
y_test = test_data["P"].astype('float64')

X_train_scaled = feature_scaler.fit_transform(X_train)  
X_val_scaled = feature_scaler.transform(X_val)
X_test_scaled = feature_scaler.transform(X_test)  

# Free up memory
del data, train_data, val_data, test_data
import gc
gc.collect()

xgb_model = XGBRegressor(
    n_estimators=5000, 
    learning_rate=0.01, 
    max_depth=4,  
    random_state=42,
    eval_metric='rmse',  
    objective='reg:squarederror',  
    tree_method='gpu_hist', 
    early_stopping_rounds=50,  
    alpha=0.1,  # L1 regularization term
    lambda_=2,  # L2 regularization term
    subsample=0.8  # Use 80% of the data for each boosting round
)

xgb_model.fit(X_train_scaled, y_train,
              eval_set=[(X_val_scaled, y_val)],  
              verbose=True)


y_val_pred = xgb_model.predict(X_val_scaled)
y_test_pred = xgb_model.predict(X_test_scaled)

y_val *= 100
y_test *= 100
y_val_pred *= 100
y_test_pred *= 100

val_mae = mean_absolute_error(y_val, y_val_pred)
val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)

test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)

print(f"Validation MAE: {val_mae:.6f}")
print(f"Validation RMSE: {val_rmse:.6f}")
print(f"Test MAE: {test_mae:.6f}")
print(f"Test RMSE: {test_rmse:.6f}")

relative_percent_error = np.abs((y_test - y_test_pred) / y_test) * 100

# Filter out outliers where RPE is greater than 10%
mask_outliers = relative_percent_error <= 10  # Create a mask for RPE less than or equal to 10%
relative_percent_error_filtered = relative_percent_error[mask_outliers]

plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, len(xgb_model.evals_result()['validation_0']['rmse']) + 1), 
         xgb_model.evals_result()['validation_0']['rmse'], label='Validation Loss', color='orange')
plt.title('Validation Loss Over Iterations', fontsize=16)
plt.xlabel('Number of Iterations', fontsize=14)
plt.ylabel('Root Mean Squared Error', fontsize=14)
plt.legend()
plt.grid()
plt.savefig(os.path.join(performance_dir, 'validation_loss.png'))
plt.close()

# Histogram of relative percent error
plt.figure(figsize=(10, 6))
plt.hist(relative_percent_error, bins=200, color='skyblue', edgecolor='black')
plt.title('Histogram of Relative Percent Error', fontsize=16)
plt.xlabel('Relative Percent Error (%)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid()
plt.savefig(os.path.join(performance_dir, 'relative_percent_error_histogram.png'))
plt.close()  

plt.figure(figsize=(12, 6))

# First subplot: Original scatter plot
plt.subplot(1, 2, 1)
plt.scatter(y_test, relative_percent_error_filtered, alpha=0.5)
plt.title('Relative Percent Error as a Function of Polarization', fontsize=16)
plt.xlabel('Polarization', fontsize=14)
plt.ylabel('Relative Percent Error (%)', fontsize=14)
plt.grid()

# Second subplot: Focused scatter plot for polarizations < 0.06%
plt.subplot(1, 2, 2)
mask = y_test < 0.06  # Create a mask for polarizations less than 0.06%
plt.scatter(y_test[mask], relative_percent_error_filtered[mask], alpha=0.5, color='orange')
plt.title('Relative Percent Error for Polarization < 0.06%', fontsize=16)
plt.xlabel('Polarization', fontsize=14)
plt.ylabel('Relative Percent Error (%)', fontsize=14)
plt.grid()

plt.savefig(os.path.join(performance_dir, 'relative_percent_error_vs_polarization_combined.png'))
plt.close()  

def build_simple_model():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(input_shape,)))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model here
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model  