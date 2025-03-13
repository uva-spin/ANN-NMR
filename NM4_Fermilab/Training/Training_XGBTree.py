import os
import json
import random
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBRegressor
import gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Plotting.Plot_Script import *
from Custom_Scripts.Misc_Functions import *

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


data_path = find_file("Deuteron_Oversampled_500K.csv")  
version = 'Deuteron_Low_Noisy_XGBoost_V2'  
performance_dir = f"Model Performance/{version}"  
model_dir = f"Models/{version}"  
os.makedirs(performance_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

print("Loading and preparing data...")
data = pd.read_csv(data_path)

feature_scaler = MinMaxScaler()
X = data.drop(columns=["P", 'SNR']).astype('float64').values + np.random.normal(0, 0.1, size=data.drop(columns=["P", 'SNR']).astype('float64').values.shape)
plt.plot(np.linspace(-3,3,500),X[0],label=f'{data["P"][0]}')
plt.legend()
plt.show()
X_scaled = feature_scaler.fit_transform(X)

y = data["P"].astype('float64')

print("Splitting data...")
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=1/3, random_state=42)

X_train = train_data.drop(columns=["P", 'SNR']).astype('float64').values
y_train = train_data["P"].astype('float64').values.reshape(-1, 1)
X_val = val_data.drop(columns=["P", 'SNR']).astype('float64').values
y_val = val_data["P"].astype('float64').values.reshape(-1, 1)
X_test = test_data.drop(columns=["P", 'SNR']).astype('float64').values
y_test = test_data["P"].astype('float64').values.reshape(-1, 1)

X_train_scaled = feature_scaler.fit_transform(X_train)  
X_val_scaled = feature_scaler.transform(X_val)
X_test_scaled = feature_scaler.transform(X_test)  

# Free up memory
del data, train_data, val_data, test_data
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


y_test_pred = xgb_model.predict(X_test_scaled)
y_test_pred = y_test_pred.flatten()

plot_rpe_and_residuals(y_test, y_test_pred, performance_dir, version)
plot_enhanced_results(y_test, y_test_pred, performance_dir, version)