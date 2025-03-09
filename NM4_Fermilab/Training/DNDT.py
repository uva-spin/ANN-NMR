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

def soft_binning(x, n_cuts, temperature=1.0):
    """
    Soft binning function to make split decisions.
    
    Args:
        x: Input tensor of shape (batch_size, 1).
        n_cuts: Number of cut points.
        temperature: Temperature parameter for softmax.
    
    Returns:
        Binned output tensor of shape (batch_size, n_cuts + 1).
    """
    # Initialize cut points as trainable variables
    cut_points = tf.Variable(tf.random.uniform([n_cuts], minval=-1.0, maxval=1.0), trainable=True)
    
    # Sort cut points to ensure monotonicity
    sorted_cut_points = tf.sort(cut_points)
    
    # Compute the logits for softmax
    logits = (x - sorted_cut_points) / temperature
    
    # Apply softmax to get the bin probabilities
    bin_probs = tf.nn.softmax(logits, axis=-1)
    
    return bin_probs

def dndt_layer(x, n_cuts, n_features, temperature=1.0):
    """
    Deep Neural Decision Tree (DNDT) layer.
    
    Args:
        x: Input tensor of shape (batch_size, n_features).
        n_cuts: Number of cut points for each feature.
        n_features: Number of features in the input.
        temperature: Temperature parameter for softmax.
    
    Returns:
        Output tensor of shape (batch_size, (n_cuts + 1) ** n_features).
    """
    # Apply soft binning to each feature
    bin_probs_list = [soft_binning(x[:, i:i+1], n_cuts, temperature) for i in range(n_features)]
    
    # Compute the Kronecker product of all bin probabilities
    z = bin_probs_list[0]
    for i in range(1, n_features):
        z = tf.einsum('bi,bj->bij', z, bin_probs_list[i])
        z = tf.reshape(z, [-1, (n_cuts + 1) ** (i + 1)])
    
    return z

def DNDT(input_shape, n_cuts=3):
    """
    Deep Neural Decision Tree (DNDT) model.
    
    Args:
        input_shape: Shape of the input data (n_features,).
        n_cuts: Number of cut points for each feature.
        n_classes: Number of output classes.
    
    Returns:
        A Keras model implementing the DNDT architecture.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Apply the DNDT layer
    x = dndt_layer(inputs, n_cuts, input_shape[0])
    
    outputs = layers.Dense(1, activation='linear')(x)
    
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
        loss='mse',
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

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train).astype('float32')
X_val = scaler.transform(X_val).astype('float32')
X_test = scaler.transform(X_test).astype('float32')

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

input_shape = (500,) 
n_cuts = 3         

model = DNDT(input_shape, n_cuts)
    
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