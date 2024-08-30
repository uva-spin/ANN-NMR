<<<<<<< HEAD
import os
import datetime
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from keras_tuner import RandomSearch
from tensorflow.keras import regularizers, callbacks
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tqdm.keras import TqdmCallback
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.activations import *
import sys
import io

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

physical_devices
if physical_devices:
    try:
        # Restrict TensorFlow to only use the first GPU
        tf.config.set_visible_devices(physical_devices[0], 'GPU')
        print(f"Using GPU: {physical_devices[0]}")
    except RuntimeError as e:
        print(f"Error setting GPU: {e}")

# Enable mixed precision to maximize GPU throughput
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Data loading (without header)
data_path = r'/media/devin/Z/Users/Devin/Desktop/Spin Physics Work/ANN Github/NMR-Fermilab/Big_Data/ANN_Sample_Data/Sample_Data_1M.csv'
df = pd.read_csv(data_path, header=None)

# Rename columns for clarity
df.columns = [f'feature_{i}' for i in range(500)] + ['Area', 'SNR']

# Preprocessing: Normalize the features and the target
scaler_X = StandardScaler()
scaler_y = MinMaxScaler()  # MinMaxScaler is better suited for small target values

X = df.drop(['Area', 'SNR'], axis=1)
y = df['Area'].values.reshape(-1, 1)

# Fit scalers
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Split data
def split_data(X, y, split=0.1):
    temp_idx = np.random.choice(len(y), size=int(len(y) * split), replace=False)
    
    tst_X = X[temp_idx]
    trn_X = np.delete(X, temp_idx, axis=0)
    
    tst_y = y[temp_idx]
    trn_y = np.delete(y, temp_idx, axis=0)
    
    return trn_X, tst_X, trn_y, tst_y

train_X, test_X, train_y, test_y = split_data(X_scaled, y_scaled)

# Create directories if they don't exist
version = 'v1'  # Change this to reflect different versions
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

        # Apply advanced hyperparameters tuning with efficient architectures
        for i in range(hp.Int('layers', 2, 10)):  # Use at least 2 layers for complexity
            units = hp.Int(f'units_{i}', min_value=64, max_value=1024, step=64)  # Larger range for units
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dense(
                units=units,
                activation=hp.Choice(f'act_{i}', ['relu', 'relu6', 'swish']),
                kernel_regularizer=regularizers.L2(1e-6)  # Regularization for better generalization
            ))
            # Use dropout for better generalization
            model.add(tf.keras.layers.Dropout(hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1)))
        
        # Final output layer
        model.add(tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32'))

        # Use advanced learning rate tuning strategies
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
    max_trials=20,  # More trials for better tuning
    executions_per_trial=1,  # Execute once per trial to reduce redundancy
    directory=log_dir,
    project_name="hyperparameter_tuning"
)

# Callbacks for efficiency and accuracy improvements
callbacks_list = [
    CSVLogger(os.path.join(performance_dir, f'training_log_{version}.csv'), append=True, separator=';'),
    # TensorBoard(log_dir=log_dir, histogram_freq=1),
    EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True),  # Optimal patience
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=1e-6),  # Aggressive learning rate reduction
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
                              epochs=200, callbacks=callbacks_list, batch_size=128)  # Increase epochs and batch size for better training

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

=======
import os
import datetime
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from keras_tuner import RandomSearch
from tensorflow.keras import regularizers, callbacks
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tqdm.keras import TqdmCallback
from keras.activations import *
import sys
import io

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

physical_devices
if physical_devices:
    try:
        # Restrict TensorFlow to only use the first GPU
        tf.config.set_visible_devices(physical_devices[0], 'GPU')
        print(f"Using GPU: {physical_devices[0]}")
    except RuntimeError as e:
        print(f"Error setting GPU: {e}")

# Enable mixed precision to maximize GPU throughput
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Data loading (without header)
# data_path = r'J:\Users\Devin\Desktop\Spin Physics Work\ANN Github\NMR-Fermilab\Big_Data\ANN_Sample_Data\Sample_Data_1M.csv'
data_path = r'/media/devin/Z/Users/Devin/Desktop/Spin Physics Work/ANN Github/NMR-Fermilab/Big_Data/ANN_Sample_Data/Sample_Data_1M.csv'
df = pd.read_csv(data_path, header=None)

# Rename columns for clarity
df.columns = [f'feature_{i}' for i in range(500)] + ['Area', 'SNR']

# Split data
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

# Create directories if they don't exist
version = 'v1'  # Change this to reflect different versions
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

        # Apply advanced hyperparameters tuning with efficient architectures
        for i in range(hp.Int('layers', 2, 10)):  # Use at least 2 layers for complexity
            units = hp.Int(f'units_{i}', min_value=64, max_value=1024, step=64)  # Larger range for units
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dense(
                units=units,
                activation=hp.Choice(f'act_{i}', ['relu', 'relu6', 'swish']),
                kernel_regularizer=regularizers.L2(1e-6)  # Regularization for better generalization
            ))
            # Use dropout for better generalization
            model.add(tf.keras.layers.Dropout(hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1)))
        
        # Final output layer
        model.add(tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32'))

        # Use advanced learning rate tuning strategies
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
    max_trials=20,  # More trials for better tuning
    executions_per_trial=1,  # Execute once per trial to reduce redundancy
    directory=log_dir,
    project_name="hyperparameter_tuning"
)

# Callbacks for efficiency and accuracy improvements
callbacks_list = [
    CSVLogger(os.path.join(performance_dir, f'training_log_{version}.csv'), append=True, separator=';'),
    # TensorBoard(log_dir=log_dir, histogram_freq=1),
    EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True),  # Optimal patience
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=1e-6),  # Aggressive learning rate reduction
    ModelCheckpoint(filepath=os.path.join(model_dir, f'best_model_{version}.h5'), save_best_only=True, monitor='val_loss', mode='min'),
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

callbacks_list = [
    CSVLogger(os.path.join(performance_dir, f'training_log_{version}.csv'), append=True, separator=';'),
    # TensorBoard(log_dir=log_dir, histogram_freq=1), 
    EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True),  # Optimal patience
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=1e-6),  # Aggressive learning rate reduction
    ModelCheckpoint(filepath=os.path.join(model_dir, f'best_model_{version}.h5'), save_best_only=True, monitor='val_loss', mode='min'),
]

# Train the final model using the best hyperparameters
fitted_data = model_tuned.fit(train_X, train_y, validation_data=(test_X, test_y),
                              epochs=200, callbacks=callbacks_list, batch_size=128)  # Increase epochs and batch size for better training

# Save the final model to the model directory
model_tuned.save(os.path.join(model_dir, f'final_model_{version}.h5'))

# Plot the training loss and save it in the performance directory
plt.figure()
plt.plot(fitted_data.history['loss'])
plt.plot(fitted_data.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(os.path.join(performance_dir, f'loss_plot_{version}.png'))
>>>>>>> 81a34ec91f499ef117ead7aad47bf20eb51f11bf
