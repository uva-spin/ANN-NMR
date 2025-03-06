import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Custom_Scripts.Lineshape import *
from Custom_Scripts.Misc_Functions import *
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.preprocessing import MinMaxScaler
import sklearn.model_selection as skm
from tensorflow.keras import regularizers  
import random


#Seeds
seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

P = .0005 # nOT IN PERCENTAGE 
samples = 500000
R = np.linspace(-2, 3, samples)


X, _, _ = GenerateLineshape(P,R)

X = np.log(X)

# fig, axs = plt.subplots(1, 2, figsize=(14, 6))  # Create side-by-side subplots

# # Plot X vs. R
# axs[0].plot(R, X, color='blue')
# axs[0].set_title('f(x) vs. R', fontsize=16)
# axs[0].set_xlabel('R', fontsize=14)
# axs[0].set_ylabel('f(x)', fontsize=14)
# axs[0].grid(True, linestyle='--', alpha=0.7)

# # Plot log(X) vs. R
# axs[1].plot(R, np.log(X), color='red')
# axs[1].set_title('log(f(x)) vs. R', fontsize=16)
# axs[1].set_xlabel('R', fontsize=14)
# axs[1].set_ylabel('log(f(x))', fontsize=14)
# axs[1].grid(True, linestyle='--', alpha=0.7)

# plt.tight_layout()  # Adjust layout for better spacing
# plt.show()

frequency_bins, errF, _ = calculate_binned_errors(X, num_bins=500)



def cosine_decay_with_warmup(epoch, lr):
    warmup_epochs = 5
    total_epochs = 1000
    if epoch < warmup_epochs:
        return lr * (epoch + 1) / warmup_epochs
    else:
        return lr * 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

lr_scheduler = LearningRateScheduler(cosine_decay_with_warmup)

def Binned_MSE(y_true, y_pred, errF):

    errF = tf.convert_to_tensor(errF, dtype=tf.float32)

    squared_diff = tf.square(y_true - y_pred)
    
    loss = tf.reduce_mean(squared_diff / tf.square(errF)) 
    
    return loss

def Binning(errF):
    def loss(y_true, y_pred):
        return Binned_MSE(y_true, y_pred, errF)
    
    return loss

X = np.array(X) 


model = keras.Sequential([
    layers.Input(shape=(1,)),  
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),  
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),  
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),  
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(1)  
])

loss_function = Binning(errF)


initial_learning_rate = 0.01  
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=initial_learning_rate), 
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

model.fit(
    R, X, 
    epochs=100, 
    batch_size=128, 
    validation_split=0.2,
    callbacks=[early_stopping,
            lr_scheduler])

model.save('Interpolation_Model.keras')


x_new = np.linspace(-2, 3, 50000)
predictions = model.predict(x_new)

predictions = np.exp(predictions)
X = np.exp(X)

plt.figure(figsize=(10, 6))
plt.scatter(R, X, label='Data Points', color='blue', alpha=0.5, s=4)  
plt.plot(x_new, predictions, label='Interpolation', color='red')
plt.xlabel('R')
plt.ylabel('Intensity')
plt.title('Interpolation using 1-1 NN')
plt.legend()
plt.show()




