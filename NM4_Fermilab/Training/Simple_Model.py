import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, regularizers, initializers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Custom_Scripts.Misc_Functions import *
from Plotting.Plot_Script import *
# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Set high precision
tf.keras.mixed_precision.set_global_policy('float64')
tf.keras.backend.set_floatx('float64')

# Create directories for output
version = "HighPrecision_Small_Value_Model"
model_dir = f"Models/{version}"
performance_dir = f"Model_Performance/{version}"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(performance_dir, exist_ok=True)

# Custom loss function focused on high precision for small values
@tf.keras.utils.register_keras_serializable()
def chi_squared_loss(y_true, y_pred):
    """
    Custom TensorFlow loss function that implements the chi-squared statistic.
    
    Chi-squared is defined as: χ² = sum((observed - expected)² / expected)
    
    In the context of a loss function:
    - y_true: The true (observed) values
    - y_pred: The predicted (expected) values
    
    Note: To avoid division by zero, a small epsilon value is added to the denominator.
    This implementation assumes positive values in y_pred.
    
    Args:
        y_true: Tensor of true (observed) values
        y_pred: Tensor of predicted (expected) values
        
    Returns:
        Tensor representing the chi-squared loss
    """
    # Add a small epsilon to avoid division by zero
    epsilon = tf.keras.backend.epsilon()
    
    # Calculate chi-squared: sum((observed - expected)² / expected)
    squared_diff = tf.square(y_true - y_pred)
    chi_squared = tf.reduce_sum(squared_diff / (y_pred + epsilon), axis=-1)
    
    # Return the mean over the batch
    return tf.reduce_mean(chi_squared)

# Simplified residual block for the network
def residual_block(x, units, l2_reg=0.01, dropout_rate=0.2):
    # Store input for residual connection
    shortcut = x
    
    # First dense layer
    y = layers.Dense(units, activation='relu', 
                    kernel_initializer=initializers.HeNormal(),
                    kernel_regularizer=regularizers.l2(l2_reg))(x)
    y = layers.BatchNormalization()(y)
    
    # Second dense layer
    y = layers.Dense(units, activation='relu',
                    kernel_initializer=initializers.HeNormal(),
                    kernel_regularizer=regularizers.l2(l2_reg))(y)
    y = layers.Dropout(dropout_rate)(y)
    
    # Adapt input dimensions if needed
    if shortcut.shape[-1] != units:
        shortcut = layers.Dense(units, kernel_initializer=initializers.HeNormal())(shortcut)
    
    # Add residual connection
    x = layers.Add()([shortcut, y])
    x = layers.Activation('relu')(x)
    
    return x

# Function to build the model
def build_high_precision_model(input_dim=500, log_transform=True):
    inputs = layers.Input(shape=(input_dim,), dtype='float64')
    
    # Initial normalization
    x = layers.BatchNormalization()(inputs)
    
    # Initial dense layer
    x = layers.Dense(256, activation='relu', 
                    kernel_initializer=initializers.HeNormal(),
                    kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.2)(x)
    # Add 3 residual blocks
    x = residual_block(x, 128, l2_reg=1e-4, dropout_rate=0.2)
    x = layers.Dropout(0.2)(x)
    x = residual_block(x, 128, l2_reg=1e-4, dropout_rate=0.2)
    x = layers.Dropout(0.2)(x)
    x = residual_block(x, 64, l2_reg=1e-4, dropout_rate=0.2)
    x = layers.Dropout(0.2)(x)
    # Output layer - no activation for regression
    outputs = layers.Dense(1, activation='linear', 
                          kernel_initializer=initializers.HeNormal())(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-4,
        decay_steps=1000,
        alpha=0.0
    )
    
    # Custom optimizer with conservative settings
    optimizer = optimizers.Adam(
        learning_rate=lr_schedule,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        clipnorm=1.0,
    )
    
    # Custom loss function for small values
    # loss = HighPrecisionLoss(alpha=1.0, beta=5.0, gamma=2.0)

    model.compile(
        optimizer=optimizer,
        loss=chi_squared_loss,
        metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')]
    )
    return model

# Load and prepare data
def prepare_data(data_path, target_column='P'):
    print("Loading data...")
    data = pd.read_csv(data_path)
    
    # Filter for small values in the range of interest
    # data = data[(data[target_column] >= 0.0005) & (data[target_column] <= 0.01)]
    print(f"Number of samples: {len(data)}")
    
    # Split data
    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=1/3, random_state=42)
    
    # Extract features and target
    if 'SNR' in data.columns:
        X_train = train_data.drop(columns=[target_column, 'SNR']).astype('float64').values
        X_val = val_data.drop(columns=[target_column, 'SNR']).astype('float64').values
        X_test = test_data.drop(columns=[target_column, 'SNR']).astype('float64').values
    else:
        X_train = train_data.drop(columns=[target_column]).astype('float64').values
        X_val = val_data.drop(columns=[target_column]).astype('float64').values
        X_test = test_data.drop(columns=[target_column]).astype('float64').values
    
    y_train = train_data[target_column].astype('float64').values
    y_val = val_data[target_column].astype('float64').values
    y_test = test_data[target_column].astype('float64').values
    
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype('float64')
    X_val = scaler.transform(X_val).astype('float64')
    X_test = scaler.transform(X_test).astype('float64')
    
    # Reshape targets
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    # Store original values before transformation
    y_train_original = y_train.copy()
    y_val_original = y_val.copy()
    y_test_original = y_test.copy()
    
    # Log transform targets for better handling of small values
    epsilon = 1e-10
    y_train_log = np.log(y_train + epsilon)
    y_val_log = np.log(y_val + epsilon)
    y_test_log = np.log(y_test + epsilon)
    
    return (X_train, X_val, X_test, 
            y_train_log, y_val_log, y_test_log, 
            y_train_original, y_val_original, y_test_original,
            scaler)

# Training function
def train_model(X_train, X_val, y_train, y_val, input_dim):
    model = build_high_precision_model(input_dim=input_dim)
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_mae',
        patience=30,
        min_delta=1e-9,
        mode='min',
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        os.path.join(model_dir, 'best_model.keras'),
        monitor='val_mae',
        save_best_only=True
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=256,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    return model, history

def evaluate_model(model, history,X_test, y_test_original):
    # Get predictions
    y_test_pred_log = model.predict(X_test)
    
    # Transform back from log space
    epsilon = 1e-10
    y_test_pred = np.exp(y_test_pred_log) - epsilon
    y_test = y_test_original.flatten()
    y_test_pred = y_test_pred.flatten()
    
    # Calculate metrics
    mae = np.mean(np.abs(y_test - y_test_pred))
    mse = np.mean(np.square(y_test - y_test_pred))
    rmse = np.sqrt(mse)
    
    # Calculate relative percentage error
    rpe = np.abs((y_test - y_test_pred) / np.abs(y_test)) * 100
    mean_rpe = np.mean(rpe)
    median_rpe = np.median(rpe)
    
    print(f"MAE: {mae:.10f}")
    print(f"RMSE: {rmse:.10f}")
    print(f"Mean RPE: {mean_rpe:.2f}%")
    print(f"Median RPE: {median_rpe:.2f}%")
    
    plot_rpe_and_residuals(y_test, y_test_pred, performance_dir, version)
    plot_enhanced_results(y_test, y_test_pred, performance_dir, version)
    plot_training_history(history, performance_dir, version)
    
    return y_test, y_test_pred, rpe


data_path = find_file("Deuteron_Oversampled_1M.csv")
X_train, X_val, X_test, y_train, y_val, y_test, y_train_original, y_val_original, y_test_original, scaler = prepare_data(data_path)
input_dim = X_train.shape[1]
model = build_high_precision_model(input_dim=input_dim)
model.summary()

model, history = train_model(X_train, X_val, y_train, y_val, input_dim)

y_test, y_test_pred, rpe = evaluate_model(model, history, X_test, y_test_original)


results = pd.DataFrame({'y_test': y_test, 'y_test_pred': y_test_pred, 'rpe': rpe})
results.to_csv(f"{performance_dir}/results.csv", index=False)










