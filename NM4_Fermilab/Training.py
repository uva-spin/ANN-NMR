import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import regularizers, initializers
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime
from Misc_Functions import *
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

# Load datasets
data_path_2_100 = find_file("Deuteron_2_100_No_Noise_500K.csv")  # Replace with actual file name
data_path_0_2 = find_file("Deuteron_0_2_No_Noise_500K.csv")  # Replace with actual file name
data_2_100 = pd.read_csv(data_path_2_100)
data_0_2 = pd.read_csv(data_path_0_2)

# Define version and directories
version = 'Deuteron_All_V2'
performance_dir = f"Model Performance/{version}"
model_dir = f"Models/{version}"
os.makedirs(performance_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Define the model
def Polarization(input_dim: int):
    model = tf.keras.Sequential()
    layer_configs = [
        {"units": 1024, "dropout": 0.25},
        {"units": 512, "dropout": 0.25},
        {"units": 256, "dropout": 0.2},
        {"units": 128, "dropout": 0.2},
        {"units": 64, "dropout": 0.2},
        {"units": 32, "dropout": 0.2},
    ]
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(
        units=layer_configs[0]["units"],
        activation='relu',
        kernel_initializer=initializers.HeNormal(),
        kernel_regularizer=regularizers.l2(1e-4),
        input_shape=(input_dim,)
    ))
    model.add(tf.keras.layers.Dropout(layer_configs[0]["dropout"]))
    for config in layer_configs[1:]:
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(
            units=config["units"],
            activation='relu',
            kernel_initializer=initializers.HeNormal(),
            kernel_regularizer=regularizers.l2(1e-4)
        ))
        model.add(tf.keras.layers.Dropout(config["dropout"]))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32'))
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=1e-4,
        weight_decay=1e-2,
        clipnorm=1.0
    )
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

# Callbacks
custom_metrics_log_path = os.path.join(performance_dir, f'custom_metrics_log_{version}.csv')
callbacks_list = [
    CSVLogger(os.path.join(performance_dir, f'training_log_{version}.csv'), append=True, separator=';'),
    EarlyStopping(monitor='val_loss', mode='min', patience=8, verbose=0, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0, min_lr=1e-10),
    ModelCheckpoint(filepath=os.path.join(model_dir, f'best_model_{version}.keras'), save_best_only=True, monitor='val_loss', mode='min'),
    MetricsLogger(log_path=custom_metrics_log_path)
]

# Preprocess data
def preprocess_data(data, val_fraction=0.2, test_fraction=0.1):
    train_split_index = int(len(data) * (1 - val_fraction - test_fraction))
    val_split_index = int(len(data) * (1 - test_fraction))
    train_data = data.iloc[:train_split_index]
    val_data = data.iloc[train_split_index:val_split_index]
    test_data = data.iloc[val_split_index:]
    target_variable = "P"
    X_train = train_data.drop([target_variable, 'SNR'], axis=1).values
    y_train = train_data[target_variable].values
    X_val = val_data.drop([target_variable, 'SNR'], axis=1).values
    y_val = val_data[target_variable].values
    X_test = test_data.drop([target_variable, 'SNR'], axis=1).values
    y_test = test_data[target_variable].values
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler

# Preprocess 2_100 data
X_train_2_100, y_train_2_100, X_val_2_100, y_val_2_100, X_test_2_100, y_test_2_100, scaler_2_100 = preprocess_data(data_2_100)

# Preprocess 0_2 data
X_train_0_2, y_train_0_2, X_val_0_2, y_val_0_2, X_test_0_2, y_test_0_2, scaler_0_2 = preprocess_data(data_0_2)

# Train on 2_100 data
print("Training on 2_100 data...")
with tf.device("/GPU:0"):
    model = Polarization(X_train_2_100.shape[1])
history_2_100 = model.fit(
    X_train_2_100, y_train_2_100,
    validation_data=(X_val_2_100, y_val_2_100),
    epochs=200,
    batch_size=256,
    callbacks=callbacks_list,
    verbose=1
)
print("Training on 2_100 data finished!")

# Retrain on 0_2 data
print("Retraining on 0_2 data...")
history_0_2 = model.fit(
    X_train_0_2, y_train_0_2,
    validation_data=(X_val_0_2, y_val_0_2),
    epochs=100,  # Fewer epochs for fine-tuning
    batch_size=256,
    callbacks=callbacks_list,
    verbose=1
)
print("Retraining on 0_2 data finished!")


plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid()

loss_plot_path = os.path.join(performance_dir, f'{version}_Loss_Plot.png')
plt.savefig(loss_plot_path, dpi=600)

print(f"Loss plot saved to {loss_plot_path}")


model_summary_path = os.path.join(performance_dir, 'model_summary.txt')
with open(model_summary_path, 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

model.save(os.path.join(model_dir, f'final_model_{version}.keras'))

print("Evaluating on test data...")

test_loss, test_mse, *is_anything_else_being_returned  = model.evaluate(X_test, y_test, batch_size=32)

y_test_pred = model.predict(X_test)
residuals = y_test - y_test_pred.flatten()

test_results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_test_pred.flatten(),
    'Residuals': residuals
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

test_summary_results = {
    'Date': [str(datetime.now())],
    'Test Loss': [test_loss],
    'Test MSE': [test_mse]
}

summary_results_df = pd.DataFrame(test_summary_results)

summary_results_file = os.path.join(performance_dir, f'test_summary_results_{version}.csv')
summary_results_df.to_csv(summary_results_file, index=False)

print(f"Test Loss: {test_loss}, Test MSE: {test_mse}")
print(f"Test summary results saved to {summary_results_file}")
print(f"Model summary saved to {model_summary_path}")

