import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, regularizers, initializers, optimizers
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
tf.keras.mixed_precision.set_global_policy('mixed_float16')

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"Using GPU: {physical_devices[0]}")

tf.keras.backend.set_floatx('float64')

# File paths and versioning
data_path = find_file("Deuteron_Low_No_Noise_500K.csv")  
version = 'Deuteron_Low_ResNet_V11_Weighted_Binning_Error'  
performance_dir = f"Model Performance/{version}"  
model_dir = f"Models/{version}"  
os.makedirs(performance_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

def residual_block(x, units, dropout_rate=0.2, l1=1e-5, l2=1e-4):
    shortcut = x
    x = layers.Dense(units, activation='swish', 
                     kernel_initializer='he_normal', 
                     kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2), 
                     dtype='float64')(x)
    x = layers.LayerNormalization()(x)  
    x = layers.Dropout(dropout_rate)(x)
    
    if shortcut.shape[-1] != units:
        shortcut = layers.Dense(units, activation='swish', 
                                kernel_initializer='he_normal', 
                                kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2), 
                                dtype='float64')(shortcut)
        shortcut = layers.LayerNormalization()(shortcut)
    
    return layers.Add()([shortcut, x])

def Polarization():
    inputs = layers.Input(shape=(500,), dtype='float64')
    
    x = layers.LayerNormalization()(inputs)
    
    units = [128, 64, 32] 
    for u in units:
        x = residual_block(x, u, dropout_rate=0.2, l1=1e-5, l2=1e-4)
    
    x = layers.Dropout(0.1)(x)
    
    outputs = layers.Dense(1, 
                activation='linear',
                kernel_initializer=initializers.HeNormal(),
                kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                dtype='float64')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    optimizer = optimizers.AdamW(
        learning_rate=0.0001,
        weight_decay=1e-3,
        epsilon=1e-6,
        clipnorm=0.1,
    )

    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name='mae'),
            tf.keras.metrics.RootMeanSquaredError(name='rmse')
        ]
)
    
    return model


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

feature_space_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
feature_space_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
feature_space_test = tf.convert_to_tensor(X_test, dtype=tf.float32)

tensorboard_callback = CustomTensorBoard(log_dir='./logs')
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_mae',  
    patience=10,        
    min_delta=1e-6,     
    mode='min',         
    restore_best_weights=True  
)

def cosine_decay_with_warmup(epoch, lr):
    warmup_epochs = 5
    total_epochs = 1000
    if epoch < warmup_epochs:
        return lr * (epoch + 1) / warmup_epochs
    else:
        return lr * 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

callbacks_list = [
    early_stopping,
    tf.keras.callbacks.ModelCheckpoint(os.path.join(model_dir, 'best_model.keras'),
                   monitor='val_mae',
                   save_best_only=True),
    tf.keras.callbacks.LearningRateScheduler(cosine_decay_with_warmup),
    tensorboard_callback,
    tf.keras.callbacks.CSVLogger(os.path.join(performance_dir, 'training_log.csv'))
]

model = Polarization()

    
history = model.fit(
    feature_space_train, y_train,
    validation_data=(feature_space_val, y_val),
    epochs=1000,
    batch_size=256,  
    callbacks=callbacks_list,
    verbose=2
)

y_test_pred = model.predict(feature_space_test).flatten()
residuals = y_test - y_test_pred

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

