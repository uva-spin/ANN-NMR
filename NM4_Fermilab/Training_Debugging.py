import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import regularizers, initializers, losses
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
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


data_path = find_file("Deuteron_V8_2_100_No_Noise_500K.csv")
version = 'Deuteron_2_100_V1'
performance_dir = f"Model Performance/{version}"
model_dir = f"Models/{version}"
os.makedirs(performance_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

def Polarization():
    normalization_layer = tf.keras.layers.Normalization(axis=-1)
    normalization_layer.adapt(X_train)  

    model = tf.keras.Sequential()
    model.add(normalization_layer)
    
    model.add(tf.keras.layers.Dense(
        units=512, 
        activation='swish', 
        kernel_initializer=initializers.HeNormal(),
        kernel_regularizer=regularizers.l2(1e-4)
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.4))
    
    model.add(tf.keras.layers.Dense(
        units=256, 
        activation='swish', 
        kernel_initializer=initializers.HeNormal(),
        kernel_regularizer=regularizers.l2(1e-4)
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.4))
    
    model.add(tf.keras.layers.Dense(
        units=128, 
        activation='swish', 
        kernel_initializer=initializers.HeNormal(),
        kernel_regularizer=regularizers.l2(1e-4)
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Dense(
        units=64, 
        activation='swish', 
        kernel_initializer=initializers.HeNormal(),
        kernel_regularizer=regularizers.l2(1e-4)
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Dense(
        1, 
        activation='sigmoid',  
        dtype='float32'
    ))
    
    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4, clipnorm=1.0)

    model.compile(optimizer=optimizer, loss='mae', metrics=['mae', 'mse'])
    
    return model


class CustomMetricsLogger(tf.keras.callbacks.Callback):
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


data = pd.read_csv(data_path)
sampled_data = data.sample(n=50000, random_state=42)  # Use random_state for reproducibility

val_fraction = 0.2
test_fraction = 0.1

train_split_index = int(len(sampled_data) * (1 - val_fraction - test_fraction))
val_split_index = int(len(sampled_data) * (1 - test_fraction))

train_data = sampled_data.iloc[:train_split_index]
val_data = sampled_data.iloc[train_split_index:val_split_index]
test_data = sampled_data.iloc[val_split_index:]

target_variable = "P"
X_train, y_train = train_data.drop([target_variable, 'SNR'], axis=1).values, train_data[target_variable].values
X_val, y_val = val_data.drop([target_variable, 'SNR'], axis=1).values, val_data[target_variable].values
X_test, y_test = test_data.drop([target_variable, 'SNR'], axis=1).values, test_data[target_variable].values


epochs_list = [5,10, 15, 20, 25, 30]  
batch_size = 1

for epochs in epochs_list:
    current_version = f"{version}_epochs_{epochs}_batch_{batch_size}"
    performance_dir = f"Model Performance/{current_version}"
    model_dir = f"Models/{current_version}"
    os.makedirs(performance_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    custom_metrics_log_path = os.path.join(performance_dir, f'custom_metrics_log_{current_version}.csv')
    custom_metrics_logger = CustomMetricsLogger(log_path=custom_metrics_log_path)

    callbacks_list = [
        CSVLogger(os.path.join(performance_dir, f'training_log_{current_version}.csv'), append=True, separator=';'),
        # EarlyStopping(monitor='val_loss', mode='min', patience=8, verbose=0, restore_best_weights=True),
        # ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0, min_lr=1e-10),
        ModelCheckpoint(filepath=os.path.join(model_dir, f'best_model_{current_version}.keras'), save_best_only=True, monitor='val_loss', mode='min'),
        custom_metrics_logger
    ]

    strategy = tf.distribute.MirroredStrategy(["GPU:0"])
    
    with strategy.scope():
        model = Polarization()

    print(f"Starting training with {epochs} epochs and batch size {batch_size}...")
    history = model.fit(
        X_train,
        y_train, 
        validation_data=(X_val, y_val), 
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list
    )

    print("Evaluating on test data...")
    test_loss, test_mse = model.evaluate(X_test, y_test, batch_size=batch_size)

    model_summary_path = os.path.join(performance_dir, 'model_summary.txt')
    with open(model_summary_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    model.save(os.path.join(model_dir, f'final_model_{version}.keras'))

    # Plotting the results with epochs and learning rate
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label="Training Loss")
    plt.plot(history.history['val_loss'], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss - {epochs} Epochs, {batch_size} Batch Size, LR={1e-4}")
    plt.legend()
    plt.grid()

    loss_plot_path = os.path.join(performance_dir, f'{current_version}_Loss_Plot.png')
    plt.savefig(loss_plot_path, dpi=600)
    print(f"Loss plot saved to {loss_plot_path}")     

    y_test_pred = model.predict(X_test)
    residuals = y_test - y_test_pred.flatten()   

    event_results_file = os.path.join(performance_dir, f'test_event_results_{current_version}.csv')
    test_results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_test_pred.flatten(),
        'Residuals': residuals
    })
    test_results_df.to_csv(event_results_file, index=False)
    print(f"Test results saved to {event_results_file}")

    print("Calculating per-sample MSE losses...")
    individual_losses = np.square(y_test - y_test_pred.flatten())  

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
    plt.title(f'Polarization vs. Loss (MSE) (LR={lr}, Epochs={epochs})', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    polarization_loss_plot_path = os.path.join(performance_dir, f'{version}_Polarization_vs_Loss.png')
    plt.savefig(polarization_loss_plot_path, dpi=600)

    print(f"Polarization vs. Loss plot saved to {polarization_loss_plot_path}")

    event_results_file = os.path.join(performance_dir, f'test_event_results_{version}.csv')
    test_results_df.to_csv(event_results_file, index=False)
    print(f"Test results saved to {event_results_file}")

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

