import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, regularizers, initializers, optimizers
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import matplotlib.pyplot as plt
from Misc_Functions import *
import random


### Let's set a specific seed for benchmarking
random.seed(42)


# Configure environment for maximum performance
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
tf.config.optimizer.set_jit(True)
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# GPU Configuration
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"Using GPU: {physical_devices[0]}")

# Precision Configuration
tf.keras.backend.set_floatx('float64')

# File paths and versioning
data_path = find_file("Deuteron_0_10_No_Noise_500K.csv")  
version = 'Deuteron_0_10_ResNet_V10'  # Updated version
performance_dir = f"Model Performance/{version}"  
model_dir = f"Models/{version}"  
os.makedirs(performance_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# 🔹 Residual Block with L1 & L2 Regularization for Precision
def residual_block(x, units, dropout_rate=0.2, l1=1e-5, l2=1e-4):
    shortcut = x
    x = layers.Dense(units, activation='swish', 
                     kernel_initializer='he_normal', 
                     kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2), 
                     dtype='float64')(x)
    x = layers.LayerNormalization()(x)  # 🔹 LayerNorm for numerical stability
    x = layers.Dropout(dropout_rate)(x)
    
    if shortcut.shape[-1] != units:
        shortcut = layers.Dense(units, activation='swish', 
                                kernel_initializer='he_normal', 
                                kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2), 
                                dtype='float64')(shortcut)
        shortcut = layers.LayerNormalization()(shortcut)
    
    return layers.Add()([shortcut, x])

# 🔹 Model Definition with L1 & L2 Regularization
def Polarization():
    inputs = layers.Input(shape=(500,), dtype='float64')

    x = layers.LayerNormalization()(inputs)  # 🔹 Normalize input for better precision

    units = [64, 32]
    for u in units:
        x = residual_block(x, u, dropout_rate=0.2, l1=1e-5, l2=1e-4)

    x = layers.Dropout(0.1)(x)  # 🔹 Monte Carlo Dropout for uncertainty tracking

    outputs = layers.Dense(1, 
                activation='linear',  
                kernel_initializer=initializers.HeNormal(),
                kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),  # 🔹 Apply L1 & L2 to output layer
                dtype='float64')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    optimizer = optimizers.AdamW(
        learning_rate=0.0001, 
        weight_decay=1e-3, 
        epsilon=1e-6,  # 🔹 Smaller epsilon for precise updates
        clipnorm=0.1,  
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.LogCosh(),
        metrics=[relative_percent_error,tf.keras.metrics.MeanAbsoluteError(name='mae')]
    )

    return model


# 🔹 Load Data
print("Loading data...")
data = pd.read_csv(data_path)

# 🔹 Split Data
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=1/3, random_state=42)

# 🔹 Feature/Target Separation
X_train = train_data.drop(columns=["P", 'SNR']).astype('float64').values
y_train = train_data["P"].astype('float64').values
X_val = val_data.drop(columns=["P", 'SNR']).astype('float64').values
y_val = val_data["P"].astype('float64').values
X_test = test_data.drop(columns=["P", 'SNR']).astype('float64').values
y_test = test_data["P"].astype('float64').values

# 🔹 Normalize Data
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train).astype('float64')
X_val = scaler.transform(X_val).astype('float64')
X_test = scaler.transform(X_test).astype('float64')

# 🔹 Callbacks for Monitoring & Training Stability
tensorboard_callback = CustomTensorBoard(log_dir='./logs')
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_mae',  
    patience=10,        
    min_delta=1e-6,     
    mode='min',         
    restore_best_weights=True  
)

callbacks_list = [
    early_stopping,
    tf.keras.callbacks.ModelCheckpoint(os.path.join(model_dir, 'best_model.keras'),
                   monitor='val_mae',
                   save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=15, min_lr=1e-7),
    tensorboard_callback,
    tf.keras.callbacks.CSVLogger(os.path.join(performance_dir, 'training_log.csv'))
]

# 🔹 Train Model
model = Polarization()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=1000,
    batch_size=256,  # 🔹 Increased batch size for GPU
    callbacks=callbacks_list,
    verbose=2
)

# Post-processing and Evaluation
y_test_pred = model.predict(X_test).flatten()
# y_test_pred = np.expm1(model.predict(X_test))  # Inverse transform
residuals = y_test - y_test_pred

# Save results with high precision
test_results_df = pd.DataFrame({
    'Actual': y_test.round(6),
    'Predicted': y_test_pred.round(6),
    'Residuals': residuals.round(6)
})

print("Calculating per-sample RPE losses...")

individual_losses = relative_percent_error(y_test,y_test_pred)
loss_results_df = pd.DataFrame({
    'Polarization': y_test,
    'Loss': individual_losses
})
loss_results_file = os.path.join(performance_dir, f'per_sample_loss_{version}.csv')
loss_results_df.to_csv(loss_results_file, index=False)
print(f"Per-sample loss results saved to {loss_results_file}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, individual_losses, alpha=0.6, color='blue', edgecolors='w', s=50)
plt.xlabel('Polarization (True Values)', fontsize=14)
plt.ylabel('Loss (RPE)', fontsize=14)
plt.title('Polarization vs. Loss (RPE)', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

polarization_loss_plot_path = os.path.join(performance_dir, f'{version}_Polarization_vs_Loss.png')
plt.savefig(polarization_loss_plot_path, dpi=600)

print(f"Polarization vs. Loss plot saved to {polarization_loss_plot_path}")

loss_diff = np.array(history.history['loss']) - np.array(history.history['val_loss'])
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


event_results_file = os.path.join(performance_dir, f'test_event_results_{version}.csv')
test_results_df.to_csv(event_results_file, index=False)

print(f"Test results saved to {event_results_file}")

plt.figure(figsize=(10, 6))

residuals_mean = np.mean(residuals)
residuals_std = np.std(residuals)

fig = plt.figure(figsize=(16, 6))  

gs = fig.add_gridspec(1, 2)  

ax1 = fig.add_subplot(gs[0])

plot_histogram(
    residuals*100, 
    'Histogram of Polarization Difference', 
    'Difference in Polarization', 
    'Count', 
    'red', 
    ax1,
    plot_norm=False
)
ax2 = fig.add_subplot(gs[1])
plot_histogram(
    np.abs(residuals*100), 
    'Histogram of Mean Absolute Error', 
    'Mean Absolute Error',
    '', 
    'orange', 
    ax2,
    plot_norm=False
)

ax1.text(0.5, -0.2, '(a)', transform=ax1.transAxes, 
         ha='center', fontsize=16,weight='bold')
ax2.text(0.5, -0.2, '(b)', transform=ax2.transAxes, 
         ha='center', fontsize=16,weight='bold')

plt.tight_layout()
plt.subplots_adjust(bottom=0.2)

output_path = os.path.join(performance_dir, f'{version}_Histograms.png')
fig.savefig(output_path,dpi=600)

print(f"Histograms plotted in {output_path}!")
