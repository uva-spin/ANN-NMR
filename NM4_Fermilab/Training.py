import os
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from keras_tuner import RandomSearch
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import joblib
from Misc_Functions import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
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

def data_generator(file_path, chunk_size=10000):
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        target_variable = "P"
        y = chunk[target_variable].values
        X = chunk.drop([target_variable, 'SNR'], axis=1).values
        yield X, y

def unscaled_data_generator(file_path, chunk_size=10000):
    for X, y in data_generator(file_path, chunk_size):
        yield X, y.reshape(-1, 1) 

#data_path = find_file("Sample_Data_Deuteron_15M_V5.csv")

data_path = r'/project/ptgroup/Devin/NMR_Final/Training_Data/Deuteron_V5/Sample_Data_Deuteron_15M_V5.csv'

def tf_dataset_generator(file_path, chunk_size=10000, batch_size=256):
    dataset = tf.data.Dataset.from_generator(
        lambda: unscaled_data_generator(file_path, chunk_size),
        output_signature=(
            tf.TensorSpec(shape=(500,), dtype=tf.float32),  
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32)  
        )
    )
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

train_dataset = tf_dataset_generator(data_path, chunk_size=10000, batch_size=256)
test_dataset = tf_dataset_generator(data_path, chunk_size=10000, batch_size=256)

version = 'Deuteron_v10'
performance_dir = f"Model Performance/{version}"
model_dir = f"Models/{version}"
os.makedirs(performance_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)


strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    def Area_Model(hp):
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(500,)))

        for i in range(hp.Int('layers', 1, 20)):
            units = hp.Int(f'units_{i}', min_value=64, max_value=2048, step=64)
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dense(
                units=units,
                activation=hp.Choice(f'act_{i}', ['relu', 'relu6', 'swish', 'mish', 'elu']),
                kernel_regularizer=regularizers.L2(1e-8)
            ))
            model.add(tf.keras.layers.Dropout(hp.Float(f'dropout_{i}', min_value=0.05, max_value=0.8, step=0.05)))

        model.add(tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', [1e-5, 1e-4, 5e-4])),
                      loss='huber', metrics=['mse'])
        return model

tuner = RandomSearch(
    Area_Model,
    objective='val_loss',
    max_trials=3,
    executions_per_trial=1
)

callbacks_list = [
    CSVLogger(os.path.join(performance_dir, f'training_log_{version}.csv'), append=True, separator=';'),
    EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=1e-6),
    ModelCheckpoint(filepath=os.path.join(model_dir, f'best_model_{version}.keras'), save_best_only=True, monitor='val_loss', mode='min')
]

print("Beginning tuning process...")
tuner.search(train_dataset, validation_data=test_dataset, epochs=30, callbacks=callbacks_list)
print("Tuning process completed...")

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
with strategy.scope():
    model_tuned = tuner.hypermodel.build(best_hps)
    model_tuned.summary()

with open(os.path.join(performance_dir, f'model_summary_{version}.txt'), 'w') as f:
    model_tuned.summary(print_fn=lambda x: f.write(x + '\n'))

print("Fitting model...")
fitted_data = model_tuned.fit(train_dataset, validation_data=test_dataset,
                              epochs=50, callbacks=callbacks_list)
print("Model fitted...")

model_tuned.save(os.path.join(model_dir, f'final_model_{version}.keras'))
print("Model saved...")

plt.figure()
plt.plot(fitted_data.history['loss'])
plt.plot(fitted_data.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(os.path.join(performance_dir, f'loss_plot_{version}.png'))
print("Loss plots saved at", performance_dir)

