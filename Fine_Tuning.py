import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, regularizers, initializers, optimizers
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # Changed from MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime
from Custom_Scripts.Misc_Functions import *
from Custom_Scripts.Loss_Functions import *
from Custom_Scripts.Lineshape import *
from Plotting.Plot_Script import *


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
data_path = find_file("Deuteron_No_Noise_1M.csv")  
version = 'Deuteron_0_80_TransferLearning_V1'  # Updated version
performance_dir = f"Model Performance/{version}"  
model_dir = f"Models/{version}"  
os.makedirs(performance_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

@tf.keras.utils.register_keras_serializable(package="custom")
def swish(x):
    return tf.nn.silu(x)

def residual_block(x, units, block_id):
    shortcut = x
    x = layers.Dense(units, activation=swish,
                     kernel_initializer="he_normal",
                     kernel_regularizer=regularizers.l2(1e-5),
                     name=f"dense_{block_id}_1")(x)  # Unique name
    x = layers.BatchNormalization(name=f"batch_norm_{block_id}_1")(x)
    x = layers.Dropout(0.3, name=f"dropout_{block_id}_1")(x)
    
    if shortcut.shape[-1] != units:
        shortcut = layers.Dense(units, kernel_initializer="he_normal",
                                name=f"shortcut_dense_{block_id}")(shortcut)
        
    x = layers.Add(name=f"add_{block_id}")([x, shortcut])
    return x

def Polarization(input_dim):

    base_model = tf.keras.models.load_model(
        '/home/devin/Documents/ANN-NMR/NM4_Fermilab/Models/Deuteron_10_80_ResNet_V6/best_model.keras',
        custom_objects={
            'swish': swish,
            'log_cosh_precision_loss': log_cosh_precision_loss
        }
    )
    
    # Freeze initial layers
    for layer in base_model.layers[:-4]:
        layer.trainable = False
        
    x = base_model.layers[-5].output  # Get layer before final blocks
    
    # Add new residual blocks for low-range specificity
    x = residual_block(x, 128, block_id="low_range_1")
    x = residual_block(x, 64, block_id="low_range_2")
    
    outputs = layers.Dense(1, activation='sigmoid',
                          kernel_initializer=initializers.RandomNormal(stddev=1e-4),
                          name="output_dense")(x)
    outputs = layers.Lambda(lambda x: x * 0.8, name="output_scaling")(outputs)
    
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    
    optimizer = optimizers.Adam(
        learning_rate=1e-5,  # Lower initial LR for fine-tuning
        weight_decay=1e-4,
        clipnorm=0.1
    )
    
    model.compile(
        optimizer=optimizer,
        loss=adaptive_loss,
        metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')]
    )
    return model

# Data Preparation
print("Getting data...")
data = pd.read_csv(data_path)
data = data.query("0.0 <= P <= 0.8").sample(frac=1, random_state=42)

# Split data
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=1/3, random_state=42)

X_train = train_data.drop(columns=["P", 'SNR']).values
y_train = train_data["P"].values
X_val = val_data.drop(columns=["P", 'SNR']).values
y_val = val_data["P"].values
X_test = test_data.drop(columns=["P", 'SNR']).values
y_test = test_data["P"].values

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

callbacks_list = [
    EarlyStopping(monitor='val_mae', patience=100, min_delta=1e-6),
    ModelCheckpoint(os.path.join(model_dir, 'full_range_model.keras')),
    ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=20, min_lr=1e-7),
    tf.keras.callbacks.TerminateOnNaN()
]

def create_weights(y):
    # Focus weights: 0-0.1% gets 5x weight, 0.1-1% 3x, others 1x
    weights = np.ones_like(y)
    weights[(y >= 0) & (y <= 0.001)] *= 5
    weights[(y > 0.001) & (y <= 0.01)] *= 3
    return weights

train_weights = create_weights(y_train)
val_weights = create_weights(y_val)

model = Polarization(X_train.shape[1])


# Phase 1: Feature adaptation
history1 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    sample_weight=train_weights,
    epochs=300,
    batch_size=512,
    callbacks=callbacks_list,
    verbose=2
)

# Phase 2: Full fine-tuning
for layer in model.layers:
    layer.trainable = True  # Unfreeze all layers

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-6),
    loss=adaptive_loss,
    metrics=['mae']
)

history2 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    sample_weight=train_weights,
    epochs=200,
    batch_size=256,
    callbacks=callbacks_list,
    verbose=2
)

y_test_pred = model.predict(X_test).flatten()
residuals = y_test - y_test_pred

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

# loss_diff = np.array(history.history['loss']) - np.array(history.history['val_loss'])
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(loss_diff) + 1), loss_diff, marker='o', label="Loss Difference (Training - Validation)")
# plt.axhline(0, color='red', linestyle='--', linewidth=1, label="Zero Difference")
# plt.xlabel("Epoch")
# plt.ylabel("Loss Difference")
# plt.title("Difference Between Training and Validation Loss")
# plt.legend()
# plt.grid()

# loss_diff_plot_path = os.path.join(performance_dir, f'{version}_Loss_Diff_Plot.png')
# plt.savefig(loss_diff_plot_path, dpi=600)

# print(f"Loss difference plot saved to {loss_diff_plot_path}")

plot_training_metrics(history1, history2, performance_dir, version)
plot_range_specific_metrics(y_test, y_test_pred, performance_dir, version)


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