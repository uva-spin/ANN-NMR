import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, regularizers, initializers, optimizers
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # Changed from MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime
from Misc_Functions import *
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
tf.keras.backend.set_floatx('float32')

# File paths and versioning
data_path_2_100 = find_file("Deuteron_2_100_No_Noise_500K.csv")  
version = 'Deuteron_10_80_ResNet_V6'  # Updated version
performance_dir = f"Model Performance/{version}"  
model_dir = f"Models/{version}"  

os.makedirs(performance_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Optimized Model Architecture
def residual_block(x, units):
    shortcut = x
    x = layers.Dense(units, activation=tf.nn.silu,  # Swish activation
                     kernel_initializer="he_normal",
                     kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    
    if shortcut.shape[-1] != units:
        shortcut = layers.Dense(units, kernel_initializer="he_normal")(shortcut)
        
    x = layers.Add()([x, shortcut])
    return x

def Polarization(input_dim):
    inputs = layers.Input(shape=(input_dim,), dtype='float32')
    
    # Feature transformation
    x = layers.Dense(512, activation=tf.nn.silu,
                    kernel_initializer=initializers.HeNormal())(inputs)
    x = layers.BatchNormalization()(x)
    
    # Residual blocks
    units = [512, 512, 256, 256, 128, 128]
    for u in units:
        x = residual_block(x, u)
    
    # Precision-focused final layers
    x = layers.Dense(64, activation=tf.nn.silu,
                    kernel_initializer=initializers.GlorotNormal())(x)
    outputs = layers.Dense(1, activation='linear',
                          kernel_initializer=initializers.RandomNormal(stddev=1e-4))(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    optimizer = optimizers.Nadam(
        learning_rate=5e-5,
        beta_1=0.9,
        beta_2=0.999,
        clipnorm=0.1
    )
    
    model.compile(
        optimizer=optimizer,
        loss=log_cosh_precision_loss,
        metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')]
    )
    return model

# Custom Loss Functions
def log_cosh_precision_loss(y_true, y_pred):
    """Hybrid loss combining log-cosh and precision weighting"""
    error = y_true - y_pred
    precision_weights = tf.math.exp(-10.0 * y_true) + 1e-6  # Higher weight near zero
    return tf.reduce_mean(precision_weights * tf.math.log(cosh(error)))

def cosh(x):
    return (tf.math.exp(x) + tf.math.exp(-x)) / 2

# Data Preparation
print("Getting data...")
data_2_100 = pd.read_csv(data_path_2_100)
data_2_100 = data_2_100.query("0.1 <= P <= 0.8").sample(frac=1, random_state=42)

# Split data
train_data, temp_data = train_test_split(data_2_100, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=1/3, random_state=42)

# Feature/target separation
X_train = train_data.drop(columns=["P", 'SNR']).values
y_train = train_data["P"].values
X_val = val_data.drop(columns=["P", 'SNR']).values
y_val = val_data["P"].values
X_test = test_data.drop(columns=["P", 'SNR']).values
y_test = test_data["P"].values

# Data Preprocessing
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Callbacks
callbacks_list = [
    EarlyStopping(monitor='val_mae', patience=50, min_delta=1e-6),
    ModelCheckpoint(os.path.join(model_dir, 'best_model.keras'),
                   monitor='val_mae',
                   save_best_only=True),
    ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=15, min_lr=1e-7)
]

# Training
model = Polarization(X_train.shape[1])
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=1000,
    batch_size=256,  # Increased batch size for GPU
    callbacks=callbacks_list,
    verbose=2
)

# Post-processing and Evaluation
y_test_pred = model.predict(X_test).flatten()
residuals = y_test - y_test_pred

# Save results with high precision
test_results_df = pd.DataFrame({
    'Actual': y_test.round(6),
    'Predicted': y_test_pred.round(6),
    'Residuals': residuals.round(6)
})

print("Calculating per-sample MSE losses...")
individual_losses = np.square(y_test - y_test_pred.flatten())  # MSE per sample

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
plt.ylabel('Loss (MSE)', fontsize=14)
plt.title('Polarization vs. Loss (MSE)', fontsize=16)
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

# test_summary_results = {
#     'Date': [str(datetime.now())],
#     'Test Loss': [test_loss],
#     'Test MSE': [test_mse]
# }

# summary_results_df = pd.DataFrame(test_summary_results)

# summary_results_file = os.path.join(performance_dir, f'test_summary_results_{version}.csv')
# summary_results_df.to_csv(summary_results_file, index=False)

