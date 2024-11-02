import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
from Misc_Functions import *
from sklearn.decomposition import PCA
from scipy.stats import norm
from datetime import datetime
import io

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.debugging.set_log_device_placement(True)

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

def data_generator(file_path, chunk_size=10000, batch_size=1024):
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        target_variable = "P"
        X = chunk.drop([target_variable, 'SNR'], axis=1).values  
        y = chunk[target_variable].values
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        for batch in dataset:
            yield batch

def split_data_in_batches(data_generator, val_fraction=0.1):
    for X_batch, y_batch in data_generator:
        split_index = int(X_batch.shape[0] * (1 - val_fraction))
        X_train_batch, X_val_batch = X_batch[:split_index], X_batch[split_index:]
        y_train_batch, y_val_batch = y_batch[:split_index], y_batch[split_index:]
        yield (X_train_batch, y_train_batch), (X_val_batch, y_val_batch)

def test_data_generator(file_path, chunk_size=10000, test_fraction=0.1):
    test_data = pd.read_csv(file_path, chunksize=chunk_size)
    test_df = pd.concat([chunk for chunk in test_data]).sample(frac=test_fraction)
    target_variable = "P"
    X_test = test_df.drop([target_variable, 'SNR'], axis=1).values
    y_test = test_df[target_variable].values
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(1024).prefetch(tf.data.experimental.AUTOTUNE)
    return test_dataset

data_path = find_file("Deuteron_V7_2_100_No_Noise_500K.csv")
version = 'Deuteron_2_100_v13'
performance_dir = f"Model Performance/{version}"
model_dir = f"Models/{version}"
os.makedirs(performance_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

def Polarization():
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(500,)))  

    model.add(tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=0.005,
                                                 beta_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1), 
                                                 gamma_initializer=tf.keras.initializers.Constant(value=0.9)))
    model.add(tf.keras.layers.Dense(units=256, activation='swish', kernel_regularizer=regularizers.L2(1e-3)))

    model.add(tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=0.005,
                                                 beta_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1), 
                                                 gamma_initializer=tf.keras.initializers.Constant(value=0.9)))
    model.add(tf.keras.layers.Dense(units=256, activation='swish', kernel_regularizer=regularizers.L2(1e-3)))

    model.add(tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=0.005,
                                                 beta_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1), 
                                                 gamma_initializer=tf.keras.initializers.Constant(value=0.9)))
    model.add(tf.keras.layers.Dense(units=256, activation='swish', kernel_regularizer=regularizers.L2(1e-3)))

    model.add(tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=0.005,
                                                 beta_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1), 
                                                 gamma_initializer=tf.keras.initializers.Constant(value=0.9)))
    model.add(tf.keras.layers.Dense(units=256, activation='elu', kernel_regularizer=regularizers.L2(1e-3)))
    
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5, amsgrad=True)

    def weighted_huber_loss(y_true, y_pred, delta=1.0):
        error = y_true - y_pred
        is_small_error = tf.abs(error) <= delta
        small_error_loss = tf.square(error) / 2
        large_error_loss = delta * (tf.abs(error) - 0.5 * delta)
        weighted_loss = tf.where(is_small_error, small_error_loss, large_error_loss)
        return tf.reduce_mean(weighted_loss * tf.abs(error + 1)) 

    model.compile(
        optimizer=optimizer,
        loss=weighted_huber_loss,
        metrics=['mse']
    )

    return model


callbacks_list = [
    CSVLogger(os.path.join(performance_dir, f'training_log_{version}.csv'), append=True, separator=';'),
    EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=0, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0, min_lr=1e-10),
    ModelCheckpoint(filepath=os.path.join(model_dir, f'best_model_{version}.keras'), save_best_only=True, monitor='val_loss', mode='min')
]


data = pd.read_csv(data_path)

val_fraction = 0.1
test_fraction = 0.1

train_split_index = int(len(data) * (1 - val_fraction - test_fraction))
val_split_index = int(len(data) * (1 - test_fraction))

train_data = data.iloc[:train_split_index]
val_data = data.iloc[train_split_index:val_split_index]
test_data = data.iloc[val_split_index:]

target_variable = "P"
X_train, y_train = train_data.drop([target_variable, 'SNR'], axis=1).values, train_data[target_variable].values
X_val, y_val = val_data.drop([target_variable, 'SNR'], axis=1).values, val_data[target_variable].values
X_test, y_test = test_data.drop([target_variable, 'SNR'], axis=1).values, test_data[target_variable].values

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(256).prefetch(tf.data.experimental.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(256).prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(256).prefetch(tf.data.experimental.AUTOTUNE)

model = Polarization()

print("Starting training on full dataset...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=150,
    batch_size=16,
    callbacks=callbacks_list
)



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

test_loss, test_mse = model.evaluate(test_dataset)

y_test_pred = model.predict(X_test)
residuals = y_test - y_test_pred.flatten()

test_results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_test_pred.flatten(),
    'Residuals': residuals
})

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