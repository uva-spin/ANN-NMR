import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, MultiHeadAttention, LayerNormalization, Dropout, GlobalAveragePooling1D, Lambda
)
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow import keras
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
    # tf.keras.mixed_precision.set_global_policy('mixed_float64')

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"Using GPU: {physical_devices[0]}")


tf.keras.backend.set_floatx('float64')

# File paths and versioning
data_path = find_file("Shifted_low_V2.csv")  
version = 'Deuteron_1DConv_V3'  
performance_dir = f"Model Performance/{version}"  
model_dir = f"Models/{version}"  
os.makedirs(performance_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
print("Loading data...")
data = pd.read_csv(data_path)

print(f"Number of rows in {data_path}: {len(data)}")

train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=1/3, random_state=42)

X_train = train_data.drop(columns=["P", 'SNR']).astype('float64').values
y_train = train_data["P"].astype('float64').values
X_val = val_data.drop(columns=["P", 'SNR']).astype('float64').values
y_val = val_data["P"].astype('float64').values
X_test = test_data.drop(columns=["P", 'SNR']).astype('float64').values
y_test = test_data["P"].astype('float64').values

# Normalize Data
scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train).astype('float64')
X_val = scaler.transform(X_val).astype('float64')
X_test = scaler.transform(X_test).astype('float64')

y_train = np.log1p(y_train)
y_val = np.log1p(y_val)

# train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)) 

# train_data = train_data.shuffle(buffer_size=len(X_train))  
# train_data = train_data.batch(32)  
# train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE) 

# val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val))
# val_data = val_data.batch(32)
# val_data = val_data.prefetch(tf.data.experimental.AUTOTUNE)

# test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))
# test_data = test_data.batch(32)
# test_data = test_data.prefetch(tf.data.experimental.AUTOTUNE)



# def Polarization_1DConv():
#     inputs = Input(shape = (500,))
#     x = tf.keras.layers.LayerNormalization()(inputs)
#     x = tf.keras.layers.Dense(64, activation = 'swish')(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.LayerNormalization()(x)
#     x = tf.keras.layers.Dense(64, activation = 'swish')(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.LayerNormalization()(x)
#     x = tf.keras.layers.Dense(64, activation = 'swish')(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     outputs = tf.keras.layers.Dense(1, activation = 'linear')(x)
    
#     optimizer = tf.keras.optimizers.RMSprop(1e-5)
#     model = tf.keras.Model(inputs = inputs, outputs = outputs)
#     model.compile(optimizer = optimizer, loss = 'mse', metrics = ['mae'])
    
#     return model

    
sample_size = X_train.shape[0]
feature_size = X_train.shape[1]
input_dimension = 1


X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], input_dimension)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], input_dimension)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], input_dimension)

y_train = y_train.reshape(y_train.shape[0], 1)
y_val = y_val.reshape(y_val.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)

def Polarization_1DConv_Model():
    sample_size = X_train.shape[0]
    feature_size = X_train.shape[1]
    input_dimension = 1

    inputs = Input(shape=(feature_size, input_dimension))
    x = tf.keras.layers.BatchNormalization()(inputs)
    
    conv1 = tf.keras.layers.Conv1D(
        filters=64, 
        kernel_size=3, 
        strides=1, 
        padding='same',
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    x = tf.keras.layers.BatchNormalization()(conv1)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.SpatialDropout1D(0.2)(x) 
    
    conv2 = tf.keras.layers.Conv1D(
        filters=32, 
        kernel_size=5, 
        strides=1, 
        padding='same',
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    x = tf.keras.layers.BatchNormalization()(conv2)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.SpatialDropout1D(0.2)(x)
    
    x_max = tf.keras.layers.GlobalMaxPooling1D()(x)
    x_avg = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Concatenate()([x_max, x_avg])
    
    x = tf.keras.layers.Dense(
        64, 
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    
    x = tf.keras.layers.Dense(
        16, 
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    
    x = tf.keras.layers.Dense(1)(x)
    outputs = tf.keras.layers.Activation('sigmoid')(x) * 0.6
    
    def weighted_huber_loss(y_true, y_pred):
        huber = tf.keras.losses.Huber(delta=0.1)(y_true, y_pred)
        
        weights = tf.exp(-5.0 * y_true) + 0.5
        
        return huber * weights
    
    cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=5e-5,
        decay_steps=1000,
        alpha=0.0001
    )
    
    optimizer = tf.keras.optimizers.Nadam(
        learning_rate=cosine_decay,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        clipnorm = 0.001
    )
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizer, 
        loss=weighted_huber_loss, 
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError()]
    )

    return model


model = Polarization_1DConv_Model()


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_mae',  
    patience=50,        
    min_delta=1e-9,     
    mode='min',         
    restore_best_weights=True  
)

EPOCHS = 500
callbacks_list = [
    early_stopping,
    tf.keras.callbacks.ModelCheckpoint(os.path.join(model_dir, 'best_model.keras'),
                   monitor='val_mae',
                   save_best_only=True),
    tf.keras.callbacks.CSVLogger(os.path.join(performance_dir, 'training_log.csv'))
]


history = model.fit(
    X_train, y_train, 
    epochs = EPOCHS, 
    validation_data = (X_val, y_val), 
    verbose = 1,
    callbacks = callbacks_list,
    batch_size = 256
)

y_pred = model.predict(X_test).flatten()
y_pred = np.exp(y_pred) - 1

plot_rpe_and_residuals(y_test, y_pred, performance_dir, version)
plot_enhanced_results(y_test, y_pred, performance_dir, version)
plot_training_history(history, performance_dir, version)

    



    
    
    
    