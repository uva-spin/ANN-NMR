import os
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from keras_tuner import RandomSearch
from tensorflow.keras import regularizers
<<<<<<< HEAD
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
=======
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
>>>>>>> 9f226776c453369d6cdece695b8b50a621a41467
import io
import sys
from Misc_Functions import *

<<<<<<< HEAD
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

# Enable logging to check GPU usage
=======
>>>>>>> 9f226776c453369d6cdece695b8b50a621a41467
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
tf.debugging.set_log_device_placement(True)

print("Available devices:")
for device in tf.config.list_physical_devices():
    print(device)

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
<<<<<<< HEAD
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Enable mixed precision to maximize GPU throughput
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Data loading (without header)
data_path = find_file("Sample_Data_1M.csv")
df = pd.read_csv(data_path, header=None)
=======
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"Using GPU: {physical_devices[0]}")
    except RuntimeError as e:
        print(f"Error setting GPU: {e}")

tf.keras.mixed_precision.set_global_policy('mixed_float16')

data_path = r'/media/devin/Z/Users/Devin/Desktop/Spin Physics Work/ANN Github/NMR-Fermilab/Big_Data/ANN_Sample_Data/Sample_Data_1M.csv'
chunk_size = 10000
chunks = pd.read_csv(data_path, header=None, chunksize=chunk_size)

df_list = []
for chunk in chunks:
    df_list.append(chunk)
df = pd.concat(df_list)
>>>>>>> 9f226776c453369d6cdece695b8b50a621a41467

df.columns = [f'feature_{i}' for i in range(500)] + ['Area', 'SNR']

<<<<<<< HEAD
# Preprocessing: Normalize the features and log-transform the target
scaler_X = StandardScaler()

X = df.drop(['Area', 'SNR'], axis=1)
y = df['Area'].values.reshape(-1, 1)

# Log-transform the target (add a small constant to avoid log(0))
y_log = np.log1p(y)

# Fit scaler on features
X_scaled = scaler_X.fit_transform(X)

# Split data
=======
>>>>>>> 9f226776c453369d6cdece695b8b50a621a41467
def split_data(X, y, split=0.1):
    temp_idx = np.random.choice(len(y), size=int(len(y) * split), replace=False)
    
    tst_X = X.iloc[temp_idx].reset_index(drop=True)
    trn_X = X.drop(temp_idx).reset_index(drop=True)
    
    tst_y = y.iloc[temp_idx].reset_index(drop=True)
    trn_y = y.drop(temp_idx).reset_index(drop=True)
    
    return trn_X, tst_X, trn_y, tst_y

<<<<<<< HEAD
train_X, test_X, train_y, test_y = split_data(X_scaled, y_log)

# Create directories if they don't exist
version = 'v3'
=======
y = df['Area']
x = df.drop(['Area', 'SNR'], axis=1)

train_X, test_X, train_y, test_y = split_data(x, y)

version = 'v1'
>>>>>>> 9f226776c453369d6cdece695b8b50a621a41467
performance_dir = f"Model Performance/{version}"
model_dir = f"Models/{version}"
os.makedirs(performance_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    def build_model(hp):
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(train_X.shape[1],)))

<<<<<<< HEAD
        for i in range(hp.Int('layers', 2, 10)):
            units = hp.Int(f'units_{i}', min_value=64, max_value=1024, step=64)
=======
        for i in range(hp.Int('layers', 2, 10)):  
            units = hp.Int(f'units_{i}', min_value=64, max_value=1024, step=64)  
>>>>>>> 9f226776c453369d6cdece695b8b50a621a41467
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dense(
                units=units,
                activation=hp.Choice(f'act_{i}', ['relu', 'relu6', 'swish']),
                kernel_regularizer=regularizers.L2(1e-6)
            ))
            model.add(tf.keras.layers.Dropout(hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1)))
        
<<<<<<< HEAD
        model.add(tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32'))
=======
        model.add(tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32'))  # Use float32 for final layer
>>>>>>> 9f226776c453369d6cdece695b8b50a621a41467

        lr_schedule = hp.Choice('learning_rate', [1e-5, 1e-4, 5e-4])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                      loss='mean_squared_error', metrics=['mean_squared_error'])
        
        return model

log_dir = os.path.join("NMR", datetime.datetime.now().strftime("%m%d-%H%M"))
os.makedirs(log_dir, exist_ok=True)

def create_dataset(X, y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X.values, y.values))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

batch_size = 64 
train_dataset = create_dataset(train_X, train_y, batch_size)
test_dataset = create_dataset(test_X, test_y, batch_size)

tuner = RandomSearch(
    build_model,
    objective='val_loss',
<<<<<<< HEAD
    max_trials=20,
    executions_per_trial=1,
=======
    max_trials=20,  
    executions_per_trial=1,  
>>>>>>> 9f226776c453369d6cdece695b8b50a621a41467
    directory=log_dir,
    project_name="hyperparameter_tuning"
)

callbacks_list = [
    CSVLogger(os.path.join(performance_dir, f'training_log_{version}.csv'), append=True, separator=';'),
    EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=1e-6),
<<<<<<< HEAD
    ModelCheckpoint(filepath=os.path.join(model_dir, f'best_model_{version}.keras'), save_best_only=True, monitor='val_loss', mode='min'),
=======
    ModelCheckpoint(filepath=os.path.join(model_dir, f'best_model_{version}.h5'), save_best_only=True, monitor='val_loss', mode='min'),
>>>>>>> 9f226776c453369d6cdece695b8b50a621a41467
]

tuner.search(train_dataset, validation_data=test_dataset, epochs=100, callbacks=callbacks_list)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
with strategy.scope():
    model_tuned = tuner.hypermodel.build(best_hps)
    model_tuned.summary()

with open(os.path.join(performance_dir, f'model_summary_{version}.txt'), 'w') as f:
    model_tuned.summary(print_fn=lambda x: f.write(x + '\n'))

<<<<<<< HEAD
# Train the final model using the best hyperparameters
fitted_data = model_tuned.fit(train_X, train_y, validation_data=(test_X, test_y),
                              epochs=200, callbacks=callbacks_list, batch_size=128)
=======
fitted_data = model_tuned.fit(train_dataset, validation_data=test_dataset,
                              epochs=200, callbacks=callbacks_list)  
>>>>>>> 9f226776c453369d6cdece695b8b50a621a41467

model_tuned.save(os.path.join(model_dir, f'final_model_{version}.h5'))

plt.figure()
plt.plot(fitted_data.history['loss'])
plt.plot(fitted_data.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(os.path.join(performance_dir, f'loss_plot_{version}.png'))

