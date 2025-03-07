import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, regularizers, initializers, optimizers
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import matplotlib.pyplot as plt
from Custom_Scripts.Misc_Functions import *
from Custom_Scripts.Loss_Functions import *
from Custom_Scripts.Lineshape import *
from Plotting.Plot_Script import *
import random


### Let's set a specific seed for benchmarking
random.seed(42)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
tf.config.optimizer.set_jit(True)
tf.keras.mixed_precision.set_global_policy('mixed_float16')

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"Using GPU: {physical_devices[0]}")


tf.keras.backend.set_floatx('float32')

# File paths and versioning
data_path = find_file("Deuteron_Low_No_Noise_500K.csv")  
version = 'Deuteron_Low_ResNet_V19_LogCosh'  
performance_dir = f"Model Performance/{version}"  
model_dir = f"Models/{version}"  
os.makedirs(performance_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)


def residual_block(x, units):
    shortcut = x
    x = layers.Dense(units, activation='swish',  # Swish activation
                     kernel_initializer="he_normal",
                     kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.LayerNormalization()(x)
    
    if shortcut.shape[-1] != units:
        shortcut = layers.Dense(units, kernel_initializer="he_normal")(shortcut)
        
    x = layers.Add()([x, shortcut])
    return x

def Polarization():
    inputs = layers.Input(shape=(500,), dtype='float32')
    
    # x = layers.Dense(512, activation=tf.nn.silu,
    #                 kernel_initializer=initializers.HeNormal())(inputs)
    x = layers.LayerNormalization()(inputs)
    
    units = [256, 128, 128, 64, 32]
    for u in units:
        x = residual_block(x, u)
    
    x = layers.Dense(64, activation='swish',
                    kernel_initializer=initializers.GlorotNormal())(x)
    outputs = layers.Dense(1, activation='linear',
                          kernel_initializer=initializers.RandomNormal(stddev=1e-4))(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    optimizer = optimizers.Nadam(
        learning_rate=5e-5,
        beta_1=0.9,
        beta_2=0.999,
        # clipnorm=0.1
        epsilon=1e-6,
        clipnorm = 1.0
    )
    
    model.compile(
        optimizer=optimizer,
        loss=log_cosh_precision_loss,
        metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')]
    )
    return model


print("Loading data...")
data = pd.read_csv(data_path)

train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=1/3, random_state=42)

X_train = train_data.drop(columns=["P", 'SNR']).astype('float32').values
y_train = train_data["P"].astype('float32').values
X_val = val_data.drop(columns=["P", 'SNR']).astype('float32').values
y_val = val_data["P"].astype('float32').values
X_test = test_data.drop(columns=["P", 'SNR']).astype('float32').values
y_test = test_data["P"].astype('float32').values

# Normalize Data
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train).astype('float32')
X_val = scaler.transform(X_val).astype('float32')
X_test = scaler.transform(X_test).astype('float32')


# tensorboard_callback = CustomTensorBoard(log_dir='./logs')
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_mae',  
    patience=50,        
    min_delta=1e-9,     
    mode='min',         
    restore_best_weights=True  
)


def cosine_decay_with_warmup(epoch, lr):
    warmup_epochs = 5
    total_epochs = 1000
    if epoch < warmup_epochs:
        return lr * (epoch + 1) / warmup_epochs
    else:
        return lr * 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

callbacks_list = [
    early_stopping,
    tf.keras.callbacks.ModelCheckpoint(os.path.join(model_dir, 'best_model.keras'),
                   monitor='val_mae',
                   save_best_only=True),
    tf.keras.callbacks.LearningRateScheduler(cosine_decay_with_warmup),
    # tensorboard_callback,
    tf.keras.callbacks.CSVLogger(os.path.join(performance_dir, 'training_log.csv'))
]


#Training the model
model = Polarization()

    
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=256,  
    callbacks=callbacks_list,
    verbose=2
)

y_test_pred = model.predict(X_test).flatten()
residuals = y_test - y_test_pred

rpe = np.abs((y_test - y_test_pred) / np.abs(y_test)) * 100  

### Plotting the results
# plot_rpe_and_residuals_over_range(y_test, y_test_pred, performance_dir, version)

plot_rpe_and_residuals(y_test, y_test_pred, performance_dir, version)


plot_training_history(history, performance_dir, version)

event_results_file = os.path.join(performance_dir, f'test_event_results_{version}.csv')
test_results_df = pd.DataFrame({
    'Actual': y_test.round(6),
    'Predicted': y_test_pred.round(6),
    'Residuals': residuals.round(6)
})
test_results_df.to_csv(event_results_file, index=False)

print(f"Test results saved to {event_results_file}")

save_model_summary(model, performance_dir, version)


