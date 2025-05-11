import tensorflow as tf
from tensorflow.keras import layers, models, Input
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Custom_Scripts.Misc_Functions import *
from Custom_Scripts.Loss_Functions import *
from Custom_Scripts.Lineshape import *
from Plotting.Plot_Script import *
import optuna
from optuna.integration import TFKerasPruningCallback



random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

data_path = find_file("Deuteron_TE_60_Noisy_Shifted.parquet")  
version = 'Deuteron_TE_60_Noisy_Shifted_1M_CNN_Attention_Optuna_V1'  
performance_dir = f"Model_Performance/{version}"  
model_dir = f"Models/{version}"  
os.makedirs(performance_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

try:
    data = pd.read_parquet(data_path, engine='pyarrow')
    print("Data loaded successfully from Parquet file!")
except Exception as e:
    print(f"Error loading data: {e}")
    
data = data.sample(frac=1, random_state=42).reset_index(drop=True)


train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=1/3, random_state=42)

X_train = train_data.drop(columns=["P", 'SNR']).astype('float64').values
y_train = train_data["P"].astype('float64').values * 100
X_val = val_data.drop(columns=["P", 'SNR']).astype('float64').values
y_val = val_data["P"].astype('float64').values * 100
X_test = test_data.drop(columns=["P", 'SNR']).astype('float64').values
y_test = test_data["P"].astype('float64').values * 100
snr_test = test_data["SNR"].values if "SNR" in test_data.columns else None


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train).astype('float64')
X_val = scaler.transform(X_val).astype('float64')
X_test = scaler.transform(X_test).astype('float64')

y_train = y_train.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)


def multi_scale_conv_block(x, filters):
    conv3 = layers.Conv1D(filters, 3, padding='same', activation='relu')(x)
    conv5 = layers.Conv1D(filters, 5, padding='same', activation='relu')(x)
    conv7 = layers.Conv1D(filters, 7, padding='same', activation='relu')(x)
    concat = layers.Concatenate()([conv3, conv5, conv7])
    # concat = layers.BatchNormalization()(concat)
    return concat

def attention_block(x):
    squeeze = layers.GlobalAveragePooling1D()(x)
    excitation = layers.Dense(x.shape[-1] // 2, activation='relu')(squeeze)
    excitation = layers.Dense(x.shape[-1], activation='sigmoid')(excitation)
    excitation = layers.Reshape((1, x.shape[-1]))(excitation)
    # excitation = layers.BatchNormalization()(excitation)
    return layers.Multiply()([x, excitation])

def residual_block(x, filters):
    shortcut = x
    conv = layers.Conv1D(filters, 3, padding='same', activation='relu')(x)
    conv = layers.Conv1D(filters, 3, padding='same')(conv)
    conv = layers.LayerNormalization()(conv)
    return layers.Add()([shortcut, conv])

def build_model(params, input_shape=(500, 1)):
    inputs = Input(shape=input_shape)
    x = multi_scale_conv_block(inputs, params['filters'])
    
    for _ in range(params['num_residual_blocks']):
        x = residual_block(x, x.shape[-1])
    
    x = attention_block(x)
    x = layers.GlobalAveragePooling1D()(x)

    classifier = layers.Dense(params['classifier_units'], activation='relu')(x)
    is_low_P = layers.Dense(1, activation='sigmoid')(classifier)
    temperature = params['temperature']
    is_low_P = layers.Activation(lambda z: tf.sigmoid(z * temperature), name='classifier')(is_low_P)

    low_reg = layers.Dense(params['reg_units'], activation='relu')(x)
    low_reg = layers.Dense(1, name='reg_low')(low_reg)

    high_reg = layers.Dense(params['reg_units'], activation='relu')(x)
    high_reg = layers.Dense(1, name='reg_high')(high_reg)

    output = layers.Lambda(lambda tensors: tensors[0] * tensors[1] + (1 - tensors[0]) * tensors[2], name='P_output')([is_low_P, low_reg, high_reg])

    model = models.Model(inputs=inputs, outputs=[is_low_P, output])
    return model

def objective(trial):
    params = {
        'filters': trial.suggest_categorical('filters', [16, 32, 64, 128]),
        'classifier_units': trial.suggest_categorical('classifier_units', [8, 16, 32, 64]),
        'reg_units': trial.suggest_categorical('reg_units', [16, 32, 64, 128]),
        'temperature': trial.suggest_float('temperature', 1.0, 5.0),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        'num_residual_blocks': trial.suggest_int('num_residual_blocks', 1, 5),
        'epochs': 50,
    }

    model = build_model(params)
    y_class = (y_train < 1.0).astype(int)
    y_reg = y_train.astype('float32')

    # Add cosine decay learning rate schedule
    initial_learning_rate = params['learning_rate']
    decay_steps = params['epochs'] * (len(X_train) // params['batch_size'])
    cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        alpha=0.0  # Final learning rate value as a fraction of initial_learning_rate
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cosine_decay),
        loss={'classifier': 'binary_crossentropy', 'P_output': 'mse'},
        loss_weights={'classifier': 0.2, 'P_output': 1.0},
        metrics={'P_output': 'mae'}
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_P_output_mae', 
        patience=10, 
        restore_best_weights=True,
        mode='min'
    )

    history = model.fit(
        X_train, {'classifier': y_class, 'P_output': y_reg},
        validation_data=(X_val, {'classifier': (y_val < 1.0).astype(int), 'P_output': y_val}),
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        callbacks=[early_stopping, TFKerasPruningCallback(trial, 'val_P_output_mae')],
        verbose=0
    )

    val_mae = min(history.history['val_P_output_mae'])
    return val_mae

if __name__ == "__main__":
    
    print("Starting hyperparameter optimization with Optuna...")
    studies_dir = os.path.join(os.path.dirname(__file__), "Optuna_Studies")
    os.makedirs(studies_dir, exist_ok=True)

    db_path = os.path.join(studies_dir, f"optuna_study_{version}.db")
    storage = optuna.storages.RDBStorage(
            f"sqlite:///{db_path}",
            skip_compatibility_check=False, 
            skip_table_creation=False  
        )
    try:
        study = optuna.create_study(
                direction='minimize', 
                pruner=optuna.pruners.MedianPruner(), 
                study_name=version, 
                storage=storage,
                load_if_exists=True
            )
    except Exception as e:
        print(f"Error creating study: {e}")
    
    study.optimize(objective, n_trials=100) 

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (val_mae): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    with open(f"{performance_dir}/best_params.json", "w") as f:
        import json
        json.dump(trial.params, f, indent=4)

    best_params = trial.params
    best_params['epochs'] = 100
    model = build_model(best_params)
    y_class = (y_train < 1.0).astype(int)
    y_reg = y_train.astype('float32')

    initial_learning_rate = best_params['learning_rate']
    decay_steps = best_params['epochs'] * (len(X_train) // best_params['batch_size'])
    cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        alpha=0.0
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cosine_decay),
        loss={'classifier': 'binary_crossentropy', 'P_output': 'mse'},
        loss_weights={'classifier': 0.2, 'P_output': 1.0},
        metrics={'P_output': 'mae'}
    )
    
    # Defining the log file for loss info     
    final_log_file = f"{performance_dir}/training_log_final_model.csv"
    csv_logger = tf.keras.callbacks.CSVLogger(final_log_file, append=True, separator=',')



    history = model.fit(
        X_train, {'classifier': y_class, 'P_output': y_reg},
        validation_data=(X_val, {'classifier': (y_val < 1.0).astype(int), 'P_output': y_val}),
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor='val_P_output_mae', 
            patience=20, 
            restore_best_weights=True,
            mode='min'
        ),
        csv_logger],
        verbose=1
    )

    model.save(f"{model_dir}/best_model.keras")

    _, y_pred = model.predict(X_test)
    y_test_flat = y_test.flatten()
    y_pred_flat = y_pred.flatten()

    plot_enhanced_performance_metrics(y_test_flat, y_pred_flat, snr_test, performance_dir, version)
    plot_enhanced_results(y_test_flat, y_pred_flat, performance_dir, version)
    plot_training_history(history, performance_dir, version)



