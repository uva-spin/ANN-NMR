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

#Seeds
seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

P = .0005
samples = 500
R = np.linspace(-3, 3, samples)


X, _, _ = GenerateLineshape(P,R)


frequency_bins, errF, _ = calculate_binned_errors(X, samples, num_bins=5000)


def cosine_decay_with_warmup(epoch, lr):
    warmup_epochs = 5
    total_epochs = 1000
    if epoch < warmup_epochs:
        return lr * (epoch + 1) / warmup_epochs
    else:
        return lr * 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

lr_scheduler = LearningRateScheduler(cosine_decay_with_warmup)

def weighted_mse_loss(y_true, y_pred, errF):

    errF = tf.convert_to_tensor(errF, dtype=tf.float32)

    squared_diff = tf.square(y_true - y_pred)
    
    loss = tf.reduce_mean(squared_diff / tf.square(errF)) 
    
    return loss

def Binning(errF):
    def loss(y_true, y_pred):
        return weighted_mse_loss(y_true, y_pred, errF)
    
    return loss

X = np.array(X) 


model = keras.Sequential([
    layers.Input(shape=(1,)),  
    layers.Dense(32, activation='relu'),  
    # layers.Dense(32, activation='relu'),  
    # layers.Dense(32, activation='relu'),  
    # layers.Dense(32,activation='relu'),
    layers.Dense(1)  
])

loss_function = Binning(errF)


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

model.fit(
    frequency_bins, X, 
    epochs=1000, 
    batch_size=4, 
    validation_split=0.2,
    callbacks=[early_stopping,
            lr_scheduler])

model.save('Interpolation_Model.keras')


x_new = np.linspace(-3, 3, 100000)
predictions = model.predict(x_new)

plt.figure(figsize=(10, 6))
plt.scatter(frequency_bins, X, label='Data Points', color='blue', alpha=0.5, s=4)  
plt.plot(x_new, predictions, label='Interpolation', color='red')
plt.xlabel('R')
plt.ylabel('Intensity')
plt.title('Interpolation using 1-1 NN')
plt.legend()
plt.show()




