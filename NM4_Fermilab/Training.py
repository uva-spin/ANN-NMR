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

from sklearn.preprocessing import MinMaxScaler
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
data_path_2_100 = find_file("Deuteron_2_100_No_Noise_500K.csv")  
# data_path_0_2 = find_file("Deuteron_0_2_No_Noise_500K.csv")  
# data_path_All = find_file("Deuteron_No_Noise_1M.csv")  

version = 'Deuteron_10_80_ResNet_V1'  # Rename for each new run
performance_dir = f"Model Performance/{version}"  
model_dir = f"Models/{version}"  

# Create necessary directories
os.makedirs(performance_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Define a residual block with dropout
def residual_block(x, units, activation, dropout_rate):
    shortcut = x
    
    # First part of the residual block
    x = layers.Dense(units, activation=activation, kernel_initializer=initializers.HeNormal(),
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(rate=dropout_rate)(x)  # Apply dropout
    x = layers.Dense(units, activation=None, kernel_initializer=initializers.HeNormal(),
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)

    # Adjust shortcut if the dimensions do not match
    if (shortcut.shape)[-1] != units:
        shortcut = layers.Dense(units, kernel_initializer=initializers.HeNormal(),
                                kernel_regularizer=regularizers.l2(1e-4))(shortcut)

    x = layers.add([x, shortcut])  # Add the shortcut connection
    x = layers.Activation(activation)(x)  # Apply activation after addition
    return x

def Polarization(hp):
    # Input Layer
    input_dim = 500
    inputs = layers.Input(shape=(input_dim,))

    # Hyperparameters for tuning
    num_blocks = hp.Int("num_blocks", min_value=2, max_value=6)  # Number of residual blocks
    units = hp.Choice("units", values=[64, 128, 256, 512])  # Units in each block
    activation = hp.Choice("activation", values=["relu", "swish", "tanh"])  # Activation function
    dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.1)  # Dropout rate

    # Add residual blocks
    x = inputs
    for _ in range(num_blocks):
        x = residual_block(x, units, activation, dropout_rate)

    # Output Layer (Sigmoid for Regression)
    outputs = layers.Dense(1, activation="sigmoid")(x)  # Sigmoid for output between 0 and 1

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Tune the optimizer (Adam or Nadam)
    optimizer_choice = hp.Choice("optimizer", values=["adam", "nadam"])
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")

    if optimizer_choice == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"]
    )

    return model



class MetricsLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_path):
        super().__init__()
        self.log_path = log_path
        self.epoch_data = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        training_loss = logs.get('loss', None)
        validation_loss = logs.get('val_loss', None)
        loss_diff = None
        if training_loss is not None and validation_loss is not None:
            loss_diff = training_loss - validation_loss
        self.epoch_data.append({
            'Epoch': epoch + 1,
            'Learning Rate': lr,
            'Training Loss': training_loss,
            'Validation Loss': validation_loss,
            'Loss Difference': loss_diff
        })

    def on_train_end(self, logs=None):
        df = pd.DataFrame(self.epoch_data)
        df.to_csv(self.log_path, index=False)
        print(f"Custom metrics log saved to {self.log_path}")

custom_metrics_log_path = os.path.join(performance_dir, f'custom_metrics_log_{version}.csv')

### Learning Rate Scheduler ###
def lr_scheduler(epoch, lr):
    return lr * 0.9 if (epoch + 1) % 10 == 0 else lr  # Reduce LR every 10 epochs

callbacks_list = [
    CSVLogger(os.path.join(performance_dir, f'training_log_{version}.csv'), append=True, separator=';'),
    EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=0, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0, min_lr=1e-10),
    ModelCheckpoint(filepath=os.path.join(model_dir, f'best_model_{version}.keras'), save_best_only=True, monitor='val_loss', mode='min'),
    MetricsLogger(log_path=custom_metrics_log_path)
]


print("Getting data...")
# Load datasets
data_2_100 = pd.read_csv(data_path_2_100)
# data_0_2 = pd.read_csv(data_path_0_2)
# data_All = pd.read_csv(data_path_All)

# Define the target variable
target_variable = "P"

# Filter datasets efficiently using .query()
data_2_100 = data_2_100.query("0.1 <= P <= 0.8")
# data_0_2 = data_0_2.query("0.1 <= P <= 0.8")
# data_All = data_All.query("0.1 <= P <= 0.8")

print(f"Data found at: {data_2_100}")

# Split data into train (70%), validation (20%), and test (10%)
train_data, temp_data = train_test_split(data_2_100, test_size=0.3, random_state=42, shuffle=True)
val_data, test_data = train_test_split(temp_data, test_size=1/3, random_state=42, shuffle=True)

# Separate features and target
X_train, y_train = train_data.drop(columns=[target_variable, 'SNR']).values, train_data[target_variable].values
X_val, y_val = val_data.drop(columns=[target_variable, 'SNR']).values, val_data[target_variable].values
X_test, y_test = test_data.drop(columns=[target_variable, 'SNR']).values, test_data[target_variable].values

# Normalize X values to [0,1]
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Add Gaussian noise (mean = 0, small stddev)
noise_std = 0.05  # Adjust as needed
X_train += np.random.normal(0, noise_std, X_train.shape)
X_val += np.random.normal(0, noise_std, X_val.shape)
X_test += np.random.normal(0, noise_std, X_test.shape)


print("Starting training on full dataset...")

# Define input dimension
input_dim = X_train.shape[1]  # Number of features

tuner = kt.BayesianOptimization(
    Polarization,
    objective="val_loss",
    max_trials=30,
    directory="tuning_results",
    project_name="polarization_model"
)
tuner.search(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    callbacks=callbacks_list 
)

# Get the best hyperparameters and model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)

# Train the best model on the full dataset
history = best_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=64,  # Use best batch size or default
    callbacks=callbacks_list,
    verbose=1
)

print("Training finished!")

print(best_hps.get("optimizer"))

# --- SAVE MODEL DETAILS ---

# Save hyperparameters
hparams_path = os.path.join(performance_dir, "best_hyperparameters.json")
with open(hparams_path, "w") as f:
    json.dump(best_hps.values, f, indent=4)

# Save model architecture
architecture_path = os.path.join(performance_dir, "model_architecture.txt")
with open(architecture_path, "w") as f:
    best_model.summary(print_fn=lambda x: f.write(x + "\n"))

# Save training history
history_path = os.path.join(performance_dir, "training_history.json")
with open(history_path, "w") as f:
    json.dump(history.history, f, indent=4)

# Save model weights
weights_path = os.path.join(model_dir, "best_model_weights.weights.h5")
best_model.save_weights(weights_path)

# Save full model
full_model_path = os.path.join(model_dir, "best_model.h5")
best_model.save(full_model_path)

print(f"✅ Model details saved in '{performance_dir}' and '{model_dir}' folders.")

model = best_model


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


# model_summary_path = os.path.join(performance_dir, 'model_summary.txt')
# with open(model_summary_path, 'w') as f:
#     model.summary(print_fn=lambda x: f.write(x + '\n'))

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
# print(f"Model summary saved to {model_summary_path}")

