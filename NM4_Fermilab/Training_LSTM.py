import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, regularizers, initializers, optimizers
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from Misc_Functions import *
# Configure environment for maximum performance
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
tf.config.optimizer.set_jit(True)
tf.keras.mixed_precision.set_global_policy('mixed_float16')
# tf.keras.backend.set_floatx('float64'))

# GPU Configuration
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"Using GPU: {physical_devices[0]}")

# Precision Configuration
tf.keras.backend.set_floatx('float64')

# File paths and versioning
data_path_2_100 = find_file("Deuteron_No_Noise_1M.csv")  
version = 'Deuteron_0_10_ResNet_LSTM_V1'  # Updated version
performance_dir = f"Model Performance/{version}"  
model_dir = f"Models/{version}"  
os.makedirs(performance_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

def residual_block(x, units, dropout_rate=0.5):
    """
    A residual block with two dense layers, dropout, and a skip connection.
    If the number of units changes, a projection layer is added to the shortcut.
    """
    shortcut = x

    # Main path
    x = layers.Dense(units, activation=tf.nn.silu, kernel_initializer=initializers.HeNormal())(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)  # Dropout after activation
    x = layers.Dense(units, activation=tf.nn.silu, kernel_initializer=initializers.HeNormal())(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)  # Dropout after activation

    # Shortcut path: Add a projection layer if the shape changes
    if shortcut.shape[-1] != units:
        shortcut = layers.Dense(units, kernel_initializer=initializers.HeNormal())(shortcut)

    # Add skip connection
    x = layers.Add()([shortcut, x])
    return x

def Polarization(input_dim=500, num_layers=10, initial_nodes=500):
    inputs = layers.Input(shape=(input_dim,), dtype='float64')

    # **LSTM Feature Extractor**
    x = layers.Reshape((input_dim, 1))(inputs)  # Reshape for LSTM
    x = layers.LSTM(128, return_sequences=True, activation='tanh')(x)  
    x = layers.LSTM(64, activation='tanh')(x)

    # Fully Connected Layer after LSTM
    x = layers.Dense(initial_nodes, activation=tf.nn.silu, kernel_initializer=initializers.HeNormal())(x)
    x = layers.BatchNormalization()(x)

    # **Residual Network**
    reduction_factor = (1 / initial_nodes) ** (1 / num_layers)
    units = [max(1, int(initial_nodes * (reduction_factor ** i))) for i in range(num_layers)]
    for u in units:
        x = residual_block(x, u)

    # **Output Layer (Softplus for precision)**
    outputs = layers.Dense(1, activation=tf.nn.softplus, kernel_initializer=initializers.HeNormal())(x)

    # **Optimizer with Cosine Decay**
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=3e-4, decay_steps=20000, alpha=1e-6
    )
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-3, epsilon=1e-7)

    # **Compile Model**
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=adaptive_weighted_huber_loss, metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')])

    return model




# Data Preparation
print("Getting data...")
data_2_100 = pd.read_csv(data_path_2_100)
data_2_100 = data_2_100.query("0.0 <= P <= 0.1").sample(frac=1, random_state=42)

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


y_train = np.log1p(y_train)  # log1p(x) = log(1+x), avoids log(0)
y_val = np.log1p(y_val)
y_test = np.log1p(y_test)

# Data Preprocessing
scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# noise_std = 0.0004  # Standard deviation of the Gaussian noise (adjust as needed)
# X_train = X_train + np.random.normal(loc=0.0, scale=noise_std, size=X_train.shape)

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
    batch_size=128,  # Increased batch size for GPU
    callbacks=callbacks_list,
    verbose=2
)

# Post-processing and Evaluation
# y_test_pred = model.predict(X_test).flatten()
y_test_pred = np.expm1(model.predict(X_test))  # Inverse transform
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

