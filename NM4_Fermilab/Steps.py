import numpy as np
import matplotlib.pyplot as plt
from Lineshape import *
from Misc_Functions import *
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.preprocessing import MinMaxScaler
import sklearn.model_selection as skm
from tensorflow.keras import regularizers  
import random

# Set a seed for reproducibility
seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

P = .3
samples = 50000
frequency_bins = np.linspace(-3, 3, samples)
frequency_bins_shape = frequency_bins.shape

X, _, _ = GenerateLineshape(P,frequency_bins)

# X += np.random.normal(0,0.005,samples)

bins_centers, errF = calculate_binned_errors(X, samples)

def cosine_decay_with_warmup(epoch, lr):
    warmup_epochs = 5
    total_epochs = 1000
    if epoch < warmup_epochs:
        return lr * (epoch + 1) / warmup_epochs
    else:
        return lr * 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

lr_scheduler = LearningRateScheduler(cosine_decay_with_warmup)

def weighted_mse_loss(y_true, y_pred, errF):
    """
    Custom loss function that computes a weighted mean squared error.

    Parameters:
    - y_true: True values (ground truth).
    - y_pred: Predicted values from the model.
    - errF: Binned errors used for weighting.

    Returns:
    - Computed loss value.
    """
    # Ensure errF is a tensor and has the same shape as y_true and y_pred
    errF = tf.convert_to_tensor(errF, dtype=tf.float32)

    
    # Compute the squared differences
    squared_diff = tf.square(y_true - y_pred)
    
    # Compute the weighted mean squared error
    loss = tf.reduce_mean(squared_diff / tf.square(errF))  # Add epsilon to avoid division by zero
    
    return loss

def create_loss_function(errF):
    """
    Creates a loss function that captures the binned errors.

    Parameters:
    - errF: Binned errors used for weighting.

    Returns:
    - A function that computes the weighted MSE loss.
    """
    def loss(y_true, y_pred):
        return weighted_mse_loss(y_true, y_pred, errF)
    
    return loss

all_X = np.array(X) 

original_shape = all_X.shape 

model = keras.Sequential([
    layers.Input(shape=(1,)),  
    layers.Dense(32, activation='relu'),  
    layers.Dense(32, activation='relu'),  
    layers.Dense(32, activation='relu'),  
    layers.Dense(32,activation='relu'),
    # layers.Dropout(0.1),  
    layers.Dense(1)  
])

loss_function = create_loss_function(errF)


initial_learning_rate = 0.01  
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=initial_learning_rate), 
    loss=loss_function)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  
    patience=10,        
    min_delta=1e-6,     
    mode='min',         
    restore_best_weights=True  
)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=10, 
    min_lr=1e-6, 
    verbose=1)

model.fit(frequency_bins, all_X, epochs=1000, batch_size=4, validation_split=0.2,
          callbacks=[early_stopping,
                     lr_scheduler])

model.save('trained_model.keras')


x_new = np.linspace(-3, 3, 500)
predictions = model.predict(x_new)
all_X = all_X.reshape(original_shape)
frequency_bins = frequency_bins.reshape(frequency_bins_shape)

print(all_X.shape)
print(frequency_bins.shape)

plt.figure(figsize=(10, 6))
plt.scatter(frequency_bins, all_X, label='Data Points', color='blue', alpha=0.5, s=4)  
plt.plot(x_new, predictions, label='Interpolation', color='red')
plt.xlabel('MHz')
plt.ylabel('V')
plt.title('Interpolation using 1-1 NN')
plt.legend()
plt.show()




