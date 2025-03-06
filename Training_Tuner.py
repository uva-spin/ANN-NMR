import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, regularizers, initializers, optimizers
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,LearningRateScheduler
import keras_tuner as kt
from keras_tuner.tuners import RandomSearch
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LeakyReLU
import time

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from Misc_Functions import *
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("Available devices:")
for device in tf.config.list_physical_devices():
    print(device)

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"Using GPU: {physical_devices[0]}")
    except RuntimeError as e:
        print(f"Error setting GPU: {e}")


tf.keras.mixed_precision.set_global_policy('mixed_float16')

  
# File paths and versioning
data_path = find_file("Deuteron_0_10_No_Noise_500K.csv")  

version = 'Deuteron_0_10_Tuner_V1'  # Rename for each new run
performance_dir = f"Model Performance/{version}"  
model_dir = f"Models/{version}"  

os.makedirs(performance_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

def residual_block(x, units, dropout_rate=0.2, l1=1e-5, l2=1e-4, activation='swish'):
    shortcut = x
    x = layers.Dense(units, activation=activation, 
                     kernel_initializer='he_normal', 
                     kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2), 
                     dtype='float64')(x)
    x = layers.LayerNormalization()(x) 
    x = layers.Dropout(dropout_rate)(x)
    
    if shortcut.shape[-1] != units:
        shortcut = layers.Dense(units, activation=activation, 
                                kernel_initializer='he_normal', 
                                kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2), 
                                dtype='float64')(shortcut)
        shortcut = layers.LayerNormalization()(shortcut)
    
    return layers.Add()([shortcut, x])

class PolarizationHyperModel(kt.HyperModel):
    def build(self, hp):
        inputs = layers.Input(shape=(500,), dtype='float64')
        x = layers.LayerNormalization()(inputs)

        num_layers = hp.Choice('num_layers', values=[2, 3, 4, 6, 8])  # Number of layers
        units_per_layer = hp.Choice('units_per_layer', values=[64, 128, 256])  # Units per layer
        dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.8, step=0.1)
        l1 = hp.Choice('l1', values=[1e-5, 1e-4])
        l2 = hp.Choice('l2', values=[1e-4, 1e-3])
        learning_rate = hp.Choice('learning_rate', values=[0.0001, 0.001])

        loss_function_choice = hp.Choice('loss_function', values=['logcosh', 'mse', 'mae'])  # Loss function choice
        if loss_function_choice == 'logcosh':
            loss_function = tf.keras.losses.LogCosh()
        elif loss_function_choice == 'mse':
            loss_function = tf.keras.losses.MeanSquaredError()
        elif loss_function_choice == 'mae':
            loss_function = tf.keras.losses.MeanAbsoluteError()

        for _ in range(num_layers):
            x = residual_block(x, units_per_layer, dropout_rate=dropout_rate, l1=l1, l2=l2)

        x = layers.Dropout(dropout_rate)(x) 

        outputs = layers.Dense(1, 
                    activation='linear',  
                    kernel_initializer=initializers.HeNormal(),
                    kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),  
                    dtype='float64')(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        optimizer = optimizers.AdamW(
            learning_rate=learning_rate, 
            weight_decay=1e-3, 
            epsilon=1e-6,  
            clipnorm=0.1,  
        )

        model.compile(
            optimizer=optimizer,
            loss=loss_function,  
            metrics=[relative_percent_error, tf.keras.metrics.MeanAbsoluteError(name='mae')]
        )

        return model




    def fit(self, hp, model, *args, **kwargs):
        trial = kwargs.get('trial', None) 
        
        # If no trial found, fallback to creating a default trial ID (e.g., timestamp)
        trial_number = trial.trial_id if trial is not None else str(int(time.time() * 1000))  # Timestamp in milliseconds
        
        log_dir = f'./logs/trial_{trial_number}'

        # Remove 'callbacks' from kwargs to avoid duplication
        callbacks = kwargs.pop('callbacks', [])

        callbacks.extend([
            tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=15, min_lr=1e-7),
            tf.keras.callbacks.TensorBoard(log_dir=log_dir),
            tf.keras.callbacks.CSVLogger(f'{log_dir}/training_log.csv')
        ])
        
        return model.fit(
            *args,
            batch_size=256,
            epochs=100,
            callbacks=callbacks, 
            **kwargs
    )


print("Loading data...")
data = pd.read_csv(data_path)

train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=1/3, random_state=42)

X_train = train_data.drop(columns=["P", 'SNR']).astype('float64').values
y_train = train_data["P"].astype('float64').values
X_val = val_data.drop(columns=["P", 'SNR']).astype('float64').values
y_val = val_data["P"].astype('float64').values
X_test = test_data.drop(columns=["P", 'SNR']).astype('float64').values
y_test = test_data["P"].astype('float64').values

X_train_diffs, X_train_err = compute_differences(X_train)
X_val, X_train_err = compute_differences(X_val)
X_test, X_test_errr = compute_differences(X_test)

y_values = np.zeros_like(X_train[0]) 
x_values = np.linspace(212, 214, 500)
plt.errorbar(x_values, y_values, yerr=X_train_err[0], capsize=5, label='Differences with Error Bars', linestyle = 'none')
plt.xlabel('Frequency Bin')
plt.ylabel('Voltage Difference')
plt.title('Voltage Differences with Error Bars')
plt.legend()
plt.show()

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train).astype('float64')
X_val = scaler.transform(X_val).astype('float64')
X_test = scaler.transform(X_test).astype('float64')

plt.figure(figsize=(10, 6))
plt.plot(x_values, X_train[0], label='Transformed Data (1st Row)', color='b')
plt.title('Transformed Data (First Row)', fontsize=14)
plt.xlabel('f', fontsize=12)
plt.ylabel('Voltage Difference', fontsize=12)
plt.grid(True)
plt.legend()
plt.show()


tuner = kt.GridSearch(
    PolarizationHyperModel(),
    objective='val_mae',
    directory='./keras_tuner',
    project_name='polarization_tuning',
    overwrite=True,
    seed = 42,
    max_trials=5
)

# Perform Hyperparameter Search
tuner.search(
    X_train, y_train,
    validation_data=(X_val, y_val),
    verbose=2
)

# Get All Trials
trials = tuner.oracle.get_best_trials(num_trials=len(tuner.oracle.trials))

# Store Results for Each Trial
all_histories = []
all_residuals = []
all_weights = []

for trial in trials:
    model = tuner.hypermodel.build(trial.hyperparameters)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=256,
        epochs=100,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=15, min_lr=1e-7)
        ],
        verbose=0
    )
    all_histories.append(history.history)
    
    y_test_pred = model.predict(X_test).flatten()
    residuals = y_test - y_test_pred
    all_residuals.append(residuals)
    
    all_weights.append(model.get_weights())

# Plot Overlayed Loss vs. Epoch
plt.figure(figsize=(10, 6))
for i, history in enumerate(all_histories):
    plt.plot(history['loss'], label=f'Trial {i} Training Loss')
    plt.plot(history['val_loss'], label=f'Trial {i} Validation Loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Overlay')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('./keras_tuner/loss_overlay.png', dpi=600)
plt.close()

# Plot Overlayed Histograms of Residuals
plt.figure(figsize=(10, 6))
for i, residuals in enumerate(all_residuals):
    plt.hist(residuals * 100, bins=50, alpha=0.5, label=f'Trial {i}')
plt.xlabel('Difference in Polarization')
plt.ylabel('Count')
plt.title('Histogram of Polarization Difference (Overlayed)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('./keras_tuner/residuals_overlay.png', dpi=600)
plt.close()

# Plot Overlayed Weights
plt.figure(figsize=(10, 6))
for i, weights in enumerate(all_weights):
    flattened_weights = np.concatenate([w.flatten() for w in weights])
    plt.hist(flattened_weights, bins=50, alpha=0.5, label=f'Trial {i}')
plt.xlabel('Weight Value')
plt.ylabel('Count')
plt.title('Histogram of Model Weights (Overlayed)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('./keras_tuner/weights_overlay.png', dpi=600)
plt.close()

print("Hyperparameter tuning complete! Results saved to ./keras_tuner/")

