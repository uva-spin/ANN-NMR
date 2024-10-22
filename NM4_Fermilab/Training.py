import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
from Misc_Functions import *
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
tf.debugging.set_log_device_placement(True)

# Check available devices
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
        # Split each batch into training and validation sets
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


data_path = find_file("Sample_Data_Deuteron_15M_V5.csv")


version = 'Deuteron_v14_SGD'
performance_dir = f"Model Performance/{version}"
model_dir = f"Models/{version}"
os.makedirs(performance_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)


def build_model(): ### I need to try more layers here (?)
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(500,))) 

    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(
        units=256, activation='relu6', kernel_regularizer=regularizers.L2(1e-2)
    ))
    model.add(tf.keras.layers.Dropout(0.6))  
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(
        units=128, activation='relu6', kernel_regularizer=regularizers.L2(1e-2)
    ))
    model.add(tf.keras.layers.Dropout(0.6))
    
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(
        units=64, activation='relu6', kernel_regularizer=regularizers.L2(1e-2)
    ))
    model.add(tf.keras.layers.Dropout(0.6))

    model.add(tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32'))  

    model.compile( ###SGD seems to be working much better than Adam for 15M data events###
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=1e-2, momentum=0.8, nesterov=True
        ),
        loss='mse',
        metrics=['mse']
    )

    return model


callbacks_list = [
    CSVLogger(os.path.join(performance_dir, f'training_log_{version}.csv'), append=True, separator=';'),
    EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=1e-6),
    ModelCheckpoint(filepath=os.path.join(model_dir, f'best_model_{version}.keras'), save_best_only=True, monitor='val_loss', mode='min')
]

train_val_data_gen = split_data_in_batches(data_generator(data_path, chunk_size=10000, batch_size=4096))

model = build_model()
model.summary()

for (X_train_batch, y_train_batch), (X_val_batch, y_val_batch) in train_val_data_gen:
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_batch, y_train_batch)).batch(4096)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val_batch, y_val_batch)).batch(4096)

    print(f"Fitting model on batch...")
    model.fit(train_dataset, validation_data=val_dataset, epochs=50, callbacks=callbacks_list)

model.save(os.path.join(model_dir, f'final_model_{version}.keras'))

print("Evaluating on test data...")
test_dataset = test_data_generator(data_path, chunk_size=10000, test_fraction=0.1)  # 10% of the dataset as test set

test_loss, test_mse = model.evaluate(test_dataset)

test_df = pd.concat([chunk for chunk in pd.read_csv(data_path, chunksize=10000)]).sample(frac=0.1)  # Same test data as used in test_data_generator
X_test = test_df.drop(["P", 'SNR'], axis=1).values
y_test_actual = test_df["P"].values

y_test_pred = model.predict(X_test)

residuals = y_test_actual - y_test_pred.flatten()

test_results_df = pd.DataFrame({
    'Actual': y_test_actual,
    'Predicted': y_test_pred.flatten(),
    'Residuals': residuals
})

event_results_file = os.path.join(performance_dir, f'test_event_results_{version}.csv')
test_results_df.to_csv(event_results_file, index=False)

print(f"Results saved to {event_results_file}")

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
    # os.path.join(".", 'Deuteron_Histogram_P_Difference.png'),
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
    # os.path.join(".", 'Deuteron_Histogram_Absolute_Error.png'),
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