import os
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from keras_tuner import RandomSearch
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import io
import sys
from Misc_Functions import *

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

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

<<<<<<< HEAD
data_path = find_file("Sample_Data_V2_1M.csv")
=======
data_path = find_file("Sample_Data_V2_1M.csv") ### Change this here depending on what data file you want
>>>>>>> 485d8911707482acb79463586b71461f27571a0c
chunk_size = 10000
chunks = pd.read_csv(data_path, chunksize=chunk_size)

df_list = []
for chunk in chunks:
    df_list.append(chunk)
df = pd.concat(df_list)


def split_data(X, y, split=0.1):
    temp_idx = np.random.choice(len(y), size=int(len(y) * split), replace=False)
    
    tst_X = X.iloc[temp_idx].reset_index(drop=True)
    trn_X = X.drop(temp_idx).reset_index(drop=True)
    
    tst_y = y.iloc[temp_idx].reset_index(drop=True)
    trn_y = y.drop(temp_idx).reset_index(drop=True)
    
    return trn_X, tst_X, trn_y, tst_y

y = df['Area']
x = df.drop(['Area', 'SNR'], axis=1)

train_X, test_X, train_y, test_y = split_data(x, y)

version = 'v9' ### We can change the version number here
performance_dir = f"Model Performance/{version}"
model_dir = f"Models/{version}"
os.makedirs(performance_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    def Area_Model(hp):
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(train_X.shape[1],)))

        for i in range(hp.Int('layers', 1, 10)):  
            units = hp.Int(f'units_{i}', min_value=64, max_value=1024, step=64)  
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dense(
                units=units,
                activation=hp.Choice(f'act_{i}', ['relu', 'relu6', 'swish','mish']),
                kernel_regularizer=regularizers.L2(1e-6)
            ))
            model.add(tf.keras.layers.Dropout(hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.8, step=0.05)))
        
        model.add(tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32'))  # Use float32 for final layer

        lr_schedule = hp.Choice('learning_rate', [1e-5, 1e-4, 5e-4])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                      loss='mean_squared_logarithmic_error', metrics=['mean_squared_logarithmic_error'])
        
        return model

log_dir = os.path.join("NMR", datetime.datetime.now().strftime("%m%d-%H%M"))
os.makedirs(log_dir, exist_ok=True)

def create_dataset(X, y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X.values, y.values))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

batch_size = 128 
train_dataset = create_dataset(train_X, train_y, batch_size)
test_dataset = create_dataset(test_X, test_y, batch_size)

tuner = RandomSearch(
    Area_Model,
    objective='val_loss',
    max_trials=3,  
    executions_per_trial=1,  
    directory=log_dir,
    project_name="hyperparameter_tuning"
)

callbacks_list = [
    CSVLogger(os.path.join(performance_dir, f'training_log_{version}.csv'), append=True, separator=';'),
    EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=1e-6),
    ModelCheckpoint(filepath=os.path.join(model_dir, f'best_model_{version}.keras'), save_best_only=True, monitor='val_loss', mode='min'),
]
print("Beginning tuning process...")
tuner.search(train_dataset, validation_data=test_dataset, epochs=15, callbacks=callbacks_list)
print("Tuning process completed...")
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
with strategy.scope():
    model_tuned = tuner.hypermodel.build(best_hps)
    model_tuned.summary()

with open(os.path.join(performance_dir, f'model_summary_{version}.txt'), 'w') as f:
    model_tuned.summary(print_fn=lambda x: f.write(x + '\n'))

print("Fitting model...")
fitted_data = model_tuned.fit(train_dataset, validation_data=test_dataset,
                              epochs=20, callbacks=callbacks_list)  
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
print("Loss plots saved at",performance_dir)

