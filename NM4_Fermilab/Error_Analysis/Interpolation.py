import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
current_dir = os.path.dirname(os.path.abspath(__file__))

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
from tqdm import tqdm

#Seeds
seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

### Version ###
version = 'varied_polarization'


### Number of points in lineshape ###
samples = 50000

### Polarizations ###
Ps = np.empty(samples)

### Lineshapes ###

Xs = []
R = np.linspace(-3.5, 3.5, samples)  

for i in tqdm(range(1)):
    Ps[i] = .0005 + np.random.uniform(0.00001, 0.00001)  
    X, _, _ = GenerateLineshape(Ps[i], R)
    Xs.append(np.log(X))  
    

Xs = np.array(Xs) 
Xs_df = pd.DataFrame(Xs) 

df = pd.DataFrame(Ps, columns=["P"])

result_df = pd.concat([df, Xs_df], axis=1)

csv_file_path = os.path.join(current_dir, f'Interpolation_Data_{version}.csv') 
result_df.to_csv(csv_file_path, index=False)

print(f"Data saved to {csv_file_path}")

print(Xs.shape)
print(Ps.shape)

frequency_bins, errF, _ = calculate_binned_errors(Xs, num_bins=samples)

print(errF.shape)


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


model = keras.Sequential([
    layers.Input(shape=(1,)),  
    layers.Dense(256, activation=tf.nn.swish, kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(128, activation=tf.nn.swish, kernel_regularizer=regularizers.l2(0.001)),  
    layers.Dense(128, activation=tf.nn.swish, kernel_regularizer=regularizers.l2(0.001)),  
    layers.Dense(64, activation=tf.nn.swish, kernel_regularizer=regularizers.l2(0.001)),  
    layers.Dense(64, activation=tf.nn.swish, kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(32, activation=tf.nn.swish, kernel_regularizer=regularizers.l2(0.001)),
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
    min_delta=1e-8,     
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
    R, Xs, 
    epochs=1000, 
    batch_size=64, 
    validation_split=0.2,
    callbacks=[early_stopping,
            lr_scheduler])

model.save(os.path.join(current_dir, f'Interpolation_Model_{version}.keras'))


x_new = np.linspace(-3, 3, 50000)
predictions = model.predict(x_new)

predictions = np.exp(predictions)
X = np.exp(Xs)

plt.figure(figsize=(10, 6))
plt.scatter(R, X, label='Data Points', color='blue', alpha=0.5, s=4)  
plt.plot(x_new, predictions, label='Interpolation', color='red')
plt.xlabel('R')
plt.ylabel('Intensity')
plt.title(f'Interpolation using 1-1 NN (P = {Ps[0]})')
plt.legend()
plt.show()




