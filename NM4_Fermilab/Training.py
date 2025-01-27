import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import regularizers, initializers,losses
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,LearningRateScheduler
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


data_path_2_100 = find_file("Deuteron_2_100_No_Noise_500K.csv") ### Type in file name here
data_path_0_2 = find_file("Deuteron_0_2_No_Noise_500K.csv") ### Type in file name here
data_path_All = find_file("Deuteron_No_Noise_1M.csv") ### Type in file name here
version = 'Deuteron_All_V5' ### Rename for each time your run it
performance_dir = f"Model Performance/{version}" ### Automatically create performanc directory for saving performance metrics
model_dir = f"Models/{version}" ### Automatically makes model directory to saving the weights
os.makedirs(performance_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

def Polarization(input_dim: int):
    # Initialize the model
    model = tf.keras.Sequential()

    # Layer configurations
    layer_configs = [
        # {"units": 1024, "dropout": 0.25},
        # {"units": 512, "dropout": 0.25},
        # {"units": 256, "dropout": 0.2},
        {"units": 128, "dropout": 0.2},
        {"units": 64, "dropout": 0.2},
        {"units": 32, "dropout": 0.2},
    ]

    # Input layer
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(
        units=layer_configs[0]["units"],
        activation='relu',
        kernel_initializer=initializers.HeNormal(),
        kernel_regularizer=regularizers.l2(1e-4),
        input_shape=(input_dim,)
    ))
    model.add(tf.keras.layers.Dropout(layer_configs[0]["dropout"]))

    # Hidden layers
    for config in layer_configs[1:]:
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(
            units=config["units"],
            activation='relu',
            kernel_initializer=initializers.HeNormal(),
            kernel_regularizer=regularizers.l2(1e-4)
        ))
        model.add(tf.keras.layers.Dropout(config["dropout"]))

    # Output layer with sigmoid activation for [0, 1] output
    model.add(tf.keras.layers.Dense(1, activation='linear', dtype='float32'))

    # Optimizer
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=1e-4,  # Adjusted learning rate for better convergence
        weight_decay = 1e-2,
        clipnorm=1.0
    )

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss='mse',  # Use MSE for regression tasks
        metrics=['mae']  # Monitor MAE for regression
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
    EarlyStopping(monitor='val_loss', mode='min', patience=8, verbose=0, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0, min_lr=1e-10),
    ModelCheckpoint(filepath=os.path.join(model_dir, f'best_model_{version}.keras'), save_best_only=True, monitor='val_loss', mode='min'),
    MetricsLogger(log_path=custom_metrics_log_path)
]


print("Getting data...")
# Load datasets
data_2_100 = pd.read_csv(data_path_2_100)
data_0_2 = pd.read_csv(data_path_0_2)
data_All = pd.read_csv(data_path_All)

# Define the target variable
target_variable = "P"  # Replace with the actual column name for the target variable

# Filter datasets to include only rows where the target variable is between 0.1 and 0.8
data_2_100 = data_2_100[(data_2_100[target_variable] >= 0.1) & (data_2_100[target_variable] <= 0.8)]
data_0_2 = data_0_2[(data_0_2[target_variable] >= 0.1) & (data_0_2[target_variable] <= 0.8)]
data_All = data_All[(data_All[target_variable] >= 0.1) & (data_All[target_variable] <= 0.8)]
print(f"Data found at: {data_2_100}")

val_fraction = 0.2
test_fraction = 0.1

train_split_index = int(len(data_2_100) * (1 - val_fraction - test_fraction))
val_split_index = int(len(data_2_100) * (1 - test_fraction))

train_data = data_2_100.iloc[:train_split_index]
val_data = data_2_100.iloc[train_split_index:val_split_index]
test_data = data_2_100.iloc[val_split_index:]

target_variable = "P"
X_train, y_train = train_data.drop([target_variable, 'SNR'], axis=1).values, train_data[target_variable].values
X_val, y_val = val_data.drop([target_variable, 'SNR'], axis=1).values, val_data[target_variable].values
X_test, y_test = test_data.drop([target_variable, 'SNR'], axis=1).values, test_data[target_variable].values

# Normalize X values to [0,1]
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Add Gaussian noise (mean = 0, small stddev)
noise_std = 0.15  # Adjust as needed
X_train += np.random.normal(0, noise_std, X_train.shape)
X_val += np.random.normal(0, noise_std, X_val.shape)
X_test += np.random.normal(0, noise_std, X_test.shape)

# Train using a single GPU
print("Compiling model...")
with tf.device("/GPU:0"):
    model = Polarization(X_train.shape[1]) ### This is num. of columns in training data
print("Model compiled!")


print("Starting training on full dataset...")
history = model.fit(
    X_train,
    y_train, 
    validation_data=(X_val, y_val), 
    epochs=200,
    batch_size=128,
    callbacks=callbacks_list,
    verbose = 1
)

print("Training finished!")


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

