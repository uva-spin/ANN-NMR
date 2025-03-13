import ydf
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers, initializers, optimizers
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
version = 'Deuteron_Low_Ensemble_Method_V1'  
performance_dir = f"Model Performance/{version}"  
model_dir = f"Models/{version}"  
os.makedirs(performance_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

def create_ydf_decision_tree_model(X_train, y_train):
    # Convert the data to a Pandas DataFrame (YDF works well with Pandas)
    train_df = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])])
    train_df["P"] = y_train

    # Train the model
    model = ydf.GradientBoostedTreesLearner(
        task=ydf.Task.REGRESSION,
        label="P",  # Target column
        num_trees=300,  # Number of trees in the forest
        max_depth=10,   # Maximum depth of each tree
    ).train(train_df)

    return model

import numpy as np

class EnsembleModel:
    def __init__(self, nn_model, ydf_model):
        self.nn_model = nn_model
        self.ydf_model = ydf_model

    def predict(self, X):
        # Predict with the neural network
        nn_pred = self.nn_model.predict(X).flatten()

        # Predict with the YDF model
        ydf_pred = self.ydf_model.predict(pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])]))

        # Average the predictions
        return (nn_pred + ydf_pred) / 2.0
    


def residual_block(x, units):
    shortcut = x
    x = layers.Dense(units, activation='swish',  # Swish activation
                     kernel_initializer="he_normal",
                     kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    
    if shortcut.shape[-1] != units:
        shortcut = layers.Dense(units, kernel_initializer="he_normal")(shortcut)
        
    x = layers.Add()([x, shortcut])
    return x

def Polarization():
    inputs = layers.Input(shape=(500,), dtype='float32')
    
    x = layers.Dense(512, activation=tf.nn.silu,
                    kernel_initializer=initializers.HeNormal())(inputs)
    x = layers.BatchNormalization()(x)
    
    units = [128, 64, 32]
    for u in units:
        x = residual_block(x, u)
    
    # x = layers.Dense(64, activation='swish',
    #                 kernel_initializer=initializers.GlorotNormal())(x)
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
Polarization_model = Polarization()

    
Polarization_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=256,  
    callbacks=callbacks_list,
    verbose=2
)

ydf_model = create_ydf_decision_tree_model(X_train, y_train)


ensemble_model = EnsembleModel(Polarization_model, ydf_model)

ensemble_predictions = ensemble_model.predict(X_test)

# Calculate MAE for the ensemble model
mae = np.mean(np.abs(ensemble_predictions - y_test))
print(f"Ensemble Model MAE: {mae}")

y_test_pred = ensemble_model.predict(X_test).flatten()
residuals = y_test - y_test_pred

rpe = np.abs((y_test - y_test_pred) / np.abs(y_test)) * 100  

### Plotting the results

plot_rpe_and_residuals(y_test, y_test_pred, performance_dir, version)


plot_training_history(Polarization_model, performance_dir, version)

event_results_file = os.path.join(performance_dir, f'test_event_results_{version}.csv')
test_results_df = pd.DataFrame({
    'Actual': y_test.round(6),
    'Predicted': y_test_pred.round(6),
    'Residuals': residuals.round(6)
})
test_results_df.to_csv(event_results_file, index=False)

print(f"Test results saved to {event_results_file}")

save_model_summary(Polarization_model, performance_dir, version)