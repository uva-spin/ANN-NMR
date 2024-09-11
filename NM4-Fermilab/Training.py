import os
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from keras_tuner import RandomSearch
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
import io
import sys
from Misc_Functions import *

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

# Enable logging to check GPU usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
tf.debugging.set_log_device_placement(True)

# Check for available devices
print("Available devices:")
for device in tf.config.list_physical_devices():
    print(device)

if tf.config.list_physical_devices('GPU'):
    print("GPU is available and recognized by TensorFlow!")
else:
    print("No GPU detected. Please ensure that your GPU and drivers are properly configured.")

# Set memory growth to prevent GPU memory allocation issues
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Enable mixed precision to maximize GPU throughput
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Data loading (without header)
data_path = find_file("Sample_Data_1M.csv")
df = pd.read_csv(data_path, header=None)

# Rename columns for clarity
df.columns = [f'feature_{i}' for i in range(500)] + ['Area', 'SNR']

# Preprocessing: Normalize the features and log-transform the target
scaler_X = StandardScaler()

X = df.drop(['Area', 'SNR'], axis=1)
y = df['Area'].values.reshape(-1, 1)

# Log-transform the target (add a small constant to avoid log(0))
y_log = np.log1p(y)

# Fit scaler on features
X_scaled = scaler_X.fit_transform(X)

# Split data
def split_data(X, y, split=0.1):
    temp_idx = np.random.choice(len(y), size=int(len(y) * split), replace=False)
    
    tst_X = X[temp_idx]
    trn_X = np.delete(X, temp_idx, axis=0)
    
    tst_y = y[temp_idx]
    trn_y = np.delete(y, temp_idx, axis=0)
    
    return trn_X, tst_X, trn_y, tst_y

train_X, test_X, train_y, test_y = split_data(X_scaled, y_log)

# Create directories if they don't exist
version = 'v3'
performance_dir = f"Model Performance/{version}"
model_dir = f"Models/{version}"
os.makedirs(performance_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Use MirroredStrategy for multi-GPU support (will work with one GPU too)
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    def build_model(hp):
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(train_X.shape[1],)))

        for i in range(hp.Int('layers', 2, 10)):
            units = hp.Int(f'units_{i}', min_value=64, max_value=1024, step=64)
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dense(
                units=units,
                activation=hp.Choice(f'act_{i}', ['relu', 'relu6', 'swish']),
                kernel_regularizer=regularizers.L2(1e-6)
            ))
            model.add(tf.keras.layers.Dropout(hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1)))
        
        model.add(tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32'))

        lr_schedule = hp.Choice('learning_rate', [1e-5, 1e-4, 5e-4])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                      loss='mean_squared_error', metrics=['mean_squared_error'])
        
        return model

# Set up logging directory
log_dir = os.path.join("NMR", datetime.datetime.now().strftime("%m%d-%H%M"))
os.makedirs(log_dir, exist_ok=True)

# Hyperparameter Tuning with Keras Tuner
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=20,
    executions_per_trial=1,
    directory=log_dir,
    project_name="hyperparameter_tuning"
)

# Callbacks for efficiency and accuracy improvements
callbacks_list = [
    CSVLogger(os.path.join(performance_dir, f'training_log_{version}.csv'), append=True, separator=';'),
    EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=1e-6),
    ModelCheckpoint(filepath=os.path.join(model_dir, f'best_model_{version}.keras'), save_best_only=True, monitor='val_loss', mode='min'),
]

# Start tuning with the GPU
tuner.search(train_X, train_y, validation_data=(test_X, test_y), epochs=100, callbacks=callbacks_list, batch_size=128)

# Retrieve best hyperparameters and build the final model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
with strategy.scope():
    model_tuned = tuner.hypermodel.build(best_hps)
    model_tuned.summary()

# Save model summary to the performance directory
with open(os.path.join(performance_dir, f'model_summary_{version}.txt'), 'w') as f:
    model_tuned.summary(print_fn=lambda x: f.write(x + '\n'))

# Train the final model using the best hyperparameters
fitted_data = model_tuned.fit(train_X, train_y, validation_data=(test_X, test_y),
                              epochs=200, callbacks=callbacks_list, batch_size=128)

# Save the final model to the model directory
model_tuned.save(os.path.join(model_dir, f'final_model_{version}.keras'))

# Plot the training loss and save it in the performance directory
plt.figure()
plt.plot(fitted_data.history['loss'])
plt.plot(fitted_data.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(os.path.join(performance_dir, f'loss_plot_{version}.png'))

