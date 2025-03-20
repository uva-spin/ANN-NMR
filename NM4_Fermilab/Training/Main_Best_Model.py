import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, regularizers, initializers, optimizers
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import matplotlib.pyplot as plt
from Custom_Scripts.Misc_Functions import *
from Custom_Scripts.Loss_Functions import *
from Custom_Scripts.Lineshape import *
from Plotting.Plot_Script import *
import random


### Let's set a specific seed for benchmarking
random.seed(42)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
tf.config.optimizer.set_jit(True)
    # tf.keras.mixed_precision.set_global_policy('mixed_float64')

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"Using GPU: {physical_devices[0]}")


tf.keras.backend.set_floatx('float64')

# File paths and versioning
data_path = find_file("Shifted_low.csv")  
version = 'Deuteron_Shifted_low_V1'  
performance_dir = f"Model Performance/{version}"  
model_dir = f"Models/{version}"  
os.makedirs(performance_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

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

# Normalize Data
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train).astype('float64')
X_val = scaler.transform(X_val).astype('float64')
X_test = scaler.transform(X_test).astype('float64')

y_test_scaled = np.log(y_test + 1e-10)
y_train_scaled = np.log(y_train + 1e-10)
y_val_scaled = np.log(y_val + 1e-10)


# @tf.function
def residual_block(x, units, dropout_rate=0.2):
    """
    A PreResNet block with BatchNormalization, activation, Dense layers, and Dropout.
    """
    
    # Store input for residual connection
    residual = x
    
    # Pre-activation: BN -> ReLU -> Dense (first layer)
    y = tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=0.99)(x)
    y = tf.keras.layers.Activation('swish')(y)
    # Apply dropout after activation but before dense layer
    y = tf.keras.layers.Dropout(dropout_rate)(y)
    y = tf.keras.layers.Dense(units=units, activation=None, use_bias=True,
                              kernel_initializer=tf.keras.initializers.HeNormal(), 
                              bias_initializer=tf.keras.initializers.Zeros(),
                              kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-05, l2=0.0001))(y)
    
    # Pre-activation: BN -> ReLU -> Dense (second layer)
    y = tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=0.99)(y)
    y = tf.keras.layers.Activation('swish')(y)
    # Apply dropout after activation but before dense layer
    y = tf.keras.layers.Dropout(dropout_rate)(y)
    y = tf.keras.layers.Dense(units=units, activation=None, use_bias=True,
                             kernel_initializer=tf.keras.initializers.HeNormal(), 
                             bias_initializer=tf.keras.initializers.Zeros(),
                             kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-05, l2=0.0001))(y)
    
    # Shortcut connection (projection if dimensions don't match)
    if x.shape[-1] != units:
        # residual = tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=0.99)(x)
        # residual = tf.keras.layers.Activation('relu')(residual)
        # No dropout in the shortcut path to preserve information flow
        residual = tf.keras.layers.Dense(units, kernel_initializer=tf.keras.initializers.HeNormal(),
                                  kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-05, l2=0.0001))(x)
    
    # Add residual connection
    output = tf.keras.layers.Add()([residual, y])
    
    return output

def Polarization():
    inputs = layers.Input(shape=(500,), dtype='float64')
    
    x = layers.Dense(units=512, activation='swish', use_bias=True,
                    kernel_initializer=tf.keras.initializers.HeNormal(), bias_initializer=tf.keras.initializers.Zeros(),
                    kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-05, l2=0.0001))(inputs)
    
    # x = layers.BatchNormalization(epsilon=0.001, momentum=0.99)(inputs)
        
    x = residual_block(x, units=256, dropout_rate=0.2)
    
    x = residual_block(x, units=256, dropout_rate=0.2)
    
    x = residual_block(x, units=256, dropout_rate=0.2)
    
    x = residual_block(x, units=128, dropout_rate=0.2)
    
    x = residual_block(x, units=128, dropout_rate=0.2)
    
    x = residual_block(x, units=128, dropout_rate=0.2)
    
    x = residual_block(x, units=64, dropout_rate=0.2)
    
    x = residual_block(x, units=64, dropout_rate=0.2)
    
    x = residual_block(x, units=64, dropout_rate=0.2)
    
    # x = layers.Dropout(rate=0.1)(x)
                
    outputs = layers.Dense(units=1, activation='linear', use_bias=True,
                         kernel_initializer=tf.keras.initializers.HeNormal(), bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-05, l2=0.0001))(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=5e-5,
        decay_steps=1000,
        alpha=0.0001
    )
    
    optimizer = optimizers.Nadam(
        learning_rate=cosine_decay,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        clipnorm = 1.0
    )
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.LogCosh(),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')]
    )
    
    return model

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_mae',  
    patience=50,        
    min_delta=1e-9,     
    mode='min',         
    restore_best_weights=True  
)


callbacks_list = [
    early_stopping,
    tf.keras.callbacks.ModelCheckpoint(os.path.join(model_dir, 'best_model.keras'),
                   monitor='val_mae',
                   save_best_only=True),
    tf.keras.callbacks.CSVLogger(os.path.join(performance_dir, 'training_log.csv'))
]


### Now we train the model

model = Polarization()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=1000,
    batch_size=128,  
    callbacks=callbacks_list,
    verbose=1
)

y_test_pred = model.predict(X_test).flatten()
# y_test_pred = np.exp(y_test_pred)
y_test_pred = y_test_pred.flatten()
y_test = y_test.flatten()
residuals = y_test - y_test_pred
rpe = np.abs((y_test - y_test_pred) / np.abs(y_test)) * 100  

plot_rpe_and_residuals(y_test, y_test_pred, performance_dir, version)

plot_enhanced_results(y_test, y_test_pred, performance_dir, version)

plot_training_history(history, performance_dir, version)

event_results_file = os.path.join(performance_dir, f'test_event_results_{version}.csv')
test_results_df = pd.DataFrame({
    'Actual': y_test.round(6),
    'Predicted': y_test_pred.round(6),
    'Residuals': residuals.round(6),
    'RPE' : rpe.round(6)
})
test_results_df.to_csv(event_results_file, index=False)

print(f"Test results saved to {event_results_file}")

save_model_summary(model, performance_dir, version)