import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
from tensorflow.keras.models import load_model
import h5py as h
from matplotlib import pyplot as plt
from tensorflow.keras import regularizers
from sklearn.utils import class_weight
import random
from keras.layers import GaussianNoise
from keras.callbacks import CSVLogger
import sklearn.preprocessing 
import keras_tuner as kt
from keras.models import Sequential
import time
from kerastuner.tuners import RandomSearch
import datetime

df = pd.read_csv('/project/ptgroup/Devin/Neural_Network/Testing_Data_v5/Sample_Data_1M.csv') # This data will be 500 bins, P, SNR, relative error, and absolute error

def split_data(X, y, split=0.1):
    temp = np.random.choice(list(range(len(y))), size=int(len(y)*split), replace=False)
    
    tst_X = pd.DataFrame.from_dict({k: v[temp] for k,v in X.items()})
    trn_X = pd.DataFrame.from_dict({k: v.drop(temp) for k,v in X.items()})
    
    tst_y = y[temp]
    trn_y = y.drop(temp)
    
    
    return trn_X, tst_X, trn_y, tst_y

y = df['Abs','Rel']
x = df.drop(['P','SNR','Abs','Rel'],axis=1)
train_X, test_X, train_y, test_y = split_data(x,y, split=0.2)


def model(hp):
  model = tf.keras.Sequential()
  model.add(tf.keras.Input(shape=(500)))
  for i in range(hp.Int('layers', 1, 10)):
    unis = hp.Int('units_' + str(i), min_value = 10, max_value = 500,default=250, step=10)
    model.add(tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=0.005, beta_initializer=tf.keras.initializers.RandomNormal(0.0, 0.05), gamma_initializer=tf.keras.initializers.Constant(value=0.9)))
    model.add(tf.keras.layers.Dense(units=unis,
                                    activation=hp.Choice('act_' + str(i), ['relu6', 'relu','LeakyReLU','swish']),kernel_regularizer=regularizers.L2(10**(-26)),activity_regularizer=regularizers.L2(10**(-26))))
  model.add(tf.keras.layers.Dense(2, activation='sigmoid'))
  hp_learning_rate = hp.Choice('learning_rate',values = [1e-6,.0001,1e-5])
  opt = tf.keras.optimizers.Adam(learning_rate = hp_learning_rate)
  model.compile(optimizer = opt, loss = 'mean_squared_error',metrics=['mean_squared_error'])
  return model

LOG_DIR = f"{int(time.time())}"

tuner = RandomSearch(
    model,
    objective='val_loss',
    max_trials=2,  
    executions_per_trial=2, 
    directory=LOG_DIR)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1,
                              patience=5, mode='auto')

csv_logger = CSVLogger('Error_Model/log_1M_v1.csv', append=True, separator=';')
earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode = 'min', patience = 5,verbose=0)

log_dir = "NMR/" + datetime.datetime.now().strftime("%m%d-%H%M")

hist_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    embeddings_freq=1,
    write_graph=True,
    update_freq='batch')

tuner.search(train_X, train_y,validation_data = (test_X, test_y), epochs=100, callbacks=[earlystop,hist_callback],use_multiprocessing=True)
best_hps=tuner.get_best_hyperparameters(num_trials=5)[0]
model_tuned = tuner.hypermodel.build(best_hps)
model_tuned.summary()

from contextlib import redirect_stdout

with open('modelsummary.txt', 'w') as f:
    with redirect_stdout(f):
        model_tuned.summary()
        
print(tuner.search_space_summary())

fitted_data = model_tuned.fit(train_X,train_y, validation_data = (test_X, test_y), epochs = 150,callbacks=[csv_logger, earlystop],batch_size = 64)

model_tuned.save('Error_Model/Error_Model_V1.h5', save_format='h5')
plt.figure()
plt.plot(fitted_data.history['loss'])
plt.plot(fitted_data.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val_loss'], loc='upper left')
plt.savefig('Error_Model/Loss_Plot_V1_tuned.png')