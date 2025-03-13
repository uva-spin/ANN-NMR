import sys
import os
import json
import gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, regularizers, initializers, optimizers
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from Custom_Scripts.Misc_Functions import *
from Custom_Scripts.Loss_Functions import *
from Custom_Scripts.Lineshape import *
from Plotting.Plot_Script import *
import random
import optuna
from optuna.integration import TFKerasPruningCallback

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Environment setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
tf.config.optimizer.set_jit(True)
# Change this line
tf.keras.mixed_precision.set_global_policy('float32')  # Instead of mixed_float16
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"Using GPU: {physical_devices[0]}")

tf.keras.backend.set_floatx('float32')

# File paths and versioning
data_path = find_file("Deuteron_Oversampled_1M.csv")  
version = 'Deuteron_Low_ResNet_Optuna_V2'  
performance_dir = f"Model Performance/{version}"  
model_dir = f"Models/{version}"  
os.makedirs(performance_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

### Defining a custom loss function here for precision
class HighPrecisionLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=1.0, beta=10.0, gamma=5.0, epsilon=1e-10, 
                 reduction='sum_over_batch_size', name="high_precision_loss"):
        """
        Args:
            alpha (float): Weight for the relative error component
            beta (float): Weight for the absolute error component
            gamma (float): Weight for the log-space error component
            epsilon (float): Small constant to prevent division by zero
            reduction: Type of reduction to apply to the loss
            name: Name of the loss function
        """
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        
    def call(self, y_true, y_pred):

        y_pred = tf.clip_by_value(y_pred, self.epsilon, float('inf'))
        
        # Relative error component (scale-invariant)
        relative_error = tf.abs(y_pred - y_true) / (y_true + self.epsilon)
        
        # Absolute error component (important for very small values)
        absolute_error = self.beta * tf.abs(y_pred - y_true)
        
        # Log-space error component (emphasizes relative differences in small values)
        log_predictions = tf.math.log(y_pred + self.epsilon)
        log_targets = tf.math.log(y_true + self.epsilon)
        log_space_error = self.gamma * tf.abs(log_predictions - log_targets)
        
        combined_loss = (
            self.alpha * relative_error + 
            absolute_error + 
            log_space_error
        )
        
        return combined_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "epsilon": self.epsilon
        })
        return config


def residual_block(x, units, l2_reg=0.01, dropout_rate=0.2):
    y = layers.Dense(units, activation=None, 
                    kernel_initializer=initializers.GlorotNormal(),
                    kernel_regularizer=regularizers.l2(l2_reg))(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)
    y = layers.Dropout(dropout_rate)(y)
    
    y = layers.Dense(units, activation=None, 
                    kernel_initializer=initializers.GlorotNormal(),
                    kernel_regularizer=regularizers.l2(l2_reg))(y)
    y = layers.BatchNormalization()(y)

    if x.shape[-1] != units:
        x = layers.Dense(units, 
                        kernel_initializer=initializers.GlorotNormal(),
                        kernel_regularizer=regularizers.l2(l2_reg))(x)
    
    x = layers.Add()([x, y])
    x = layers.Activation('relu')(x)
    # x = layers.Dropout(dropout_rate)(x)
    return x

def Polarization_Model(params):
    inputs = layers.Input(shape=(500,), dtype='float32')
    
    x = layers.BatchNormalization()(inputs)
    # x = layers.Dropout(params.get('input_dropout_rate', 0.1))(x)
    
    num_layers = params['num_layers']
    units_per_layer = params['units_per_layer']
    l2_reg = params['l2_reg']
    dropout_rate = params.get('dropout_rate', 0.2) 
    
    for i in range(num_layers):
        units = units_per_layer[i]
        x = residual_block(x, units, l2_reg, dropout_rate)
        
    # Add dropout before the final layer
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(1, activation='linear', 
                          kernel_initializer=initializers.GlorotNormal())(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    optimizer = optimizers.Nadam(
        learning_rate=params['learning_rate'],
        beta_1=params['beta_1'],
        beta_2=params['beta_2'],
        epsilon=params['epsilon'],
        clipnorm=params['clipnorm']
    )
    # loss_function = HighPrecisionLoss(alpha=1.0, beta=10.0, gamma=5.0)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.LogCosh(),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')]
    )
    return model

def objective(trial):
    num_layers = trial.suggest_int('num_layers', 2, 6)
    
    units_per_layer = []
    for i in range(num_layers):
        max_units = 256  
        min_units = 16  
        
        units = trial.suggest_categorical(f'units_{i}', [min_units, min_units*2, min_units*4, max_units])
        units_per_layer.append(units)
    
    
    params = {
        'num_layers': num_layers,
        'units_per_layer': units_per_layer,  # Pass the entire list
        'l2_reg': trial.suggest_float('l2_reg', 1e-4, 1e-1, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'beta_1': trial.suggest_float('beta_1', 0.8, 0.99),
        'beta_2': trial.suggest_float('beta_2', 0.9, 0.999),
        'epsilon': trial.suggest_float('epsilon', 1e-8, 1e-6, log=True),
        'clipnorm': trial.suggest_float('clipnorm', 0.1, 2.0),
    }
    
    model = Polarization_Model(params)
    
    early_stopping = EarlyStopping(
        monitor='val_mae',
        patience=20,
        min_delta=1e-9,
        mode='min',
        restore_best_weights=True
    )
    
    pruning_callback = TFKerasPruningCallback(trial, 'val_mae')
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,  
        batch_size=256,  
        callbacks=[early_stopping, pruning_callback],
        verbose=0  
    )
    
    # Clear memory after each trial
    tf.keras.backend.clear_session()
    gc.collect()  # Collect garbage to free up memory

    
    return history.history['val_mae'][-1]

print("Loading data...")
try:
    data = pd.read_csv(data_path)
    print("Data loaded successfully!")
except Exception as e:
    print(f"Error loading data: {e}")
data = data[data['P'] <= 0.01]
print(f"Number of samples: {len(data)}")
    
    
    
    
print("Creating bins for stratified splitting...")
data['P_bin'] = pd.qcut(data['P'], q=10, labels=False)

train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42, stratify=data['P_bin'])
val_data, test_data = train_test_split(temp_data, test_size=1/3, random_state=42, stratify=temp_data['P_bin'])
print("Data split successfully")

print("Preparing features and targets...")
X_train = train_data.drop(columns=["P", 'SNR', 'P_bin']).astype('float32').values
y_train = train_data["P"].astype('float32').values
X_val = val_data.drop(columns=["P", 'SNR', 'P_bin']).astype('float32').values
y_val = val_data["P"].astype('float32').values
X_test = test_data.drop(columns=["P", 'SNR', 'P_bin']).astype('float32').values
y_test = test_data["P"].astype('float32').values
print("Features and targets prepared successfully")

print("Normalizing data...")
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train).astype('float32')
X_val = scaler.transform(X_val).astype('float32')
X_test = scaler.transform(X_test).astype('float32')
print("Data normalized successfully")


batch_size = 256  # Define your batch size

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=len(X_train))  
train_dataset = train_dataset.batch(batch_size)  
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)  

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.batch(batch_size)  
val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)  

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.batch(batch_size)  
test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)  


# Run Optuna study
if __name__ == "__main__":
    try:
        print("Starting hyperparameter optimization with Optuna...")
        storage = optuna.storages.RDBStorage(
            "sqlite:///optuna_study.db",
            skip_compatibility_check=False, 
            skip_table_creation=False  
        )
        
        
        study = optuna.create_study(
            direction='minimize', 
            pruner=optuna.pruners.MedianPruner(), 
            study_name=version, 
            storage=storage,
            load_if_exists=True
            )
        study.optimize(objective, n_trials=1)  
        
        print("Best trial:")
        trial = study.best_trial
        print(f"  Value (val_mae): {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        
        # Extract the best parameters
        num_layers = trial.params['num_layers']
        units_per_layer = []
        for i in range(num_layers):
            units_per_layer.append(trial.params[f'units_{i}'])
        
        best_params = {
            'num_layers': num_layers,
            'units_per_layer': units_per_layer,
            'l2_reg': trial.params['l2_reg'],
            'learning_rate': trial.params['learning_rate'],
            'beta_1': trial.params['beta_1'],
            'beta_2': trial.params['beta_2'],
            'epsilon': trial.params['epsilon'],
            'clipnorm': trial.params['clipnorm'],
        }
        
        with open(os.path.join(performance_dir, 'best_params.json'), 'w') as f:
            json.dump(best_params, f, indent=4)
        
        print("Training final model with best parameters...")
        final_model = Polarization_Model(best_params)
        
        early_stopping = EarlyStopping(
            monitor='val_mae',
            patience=50,
            min_delta=1e-9,
            mode='min',
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_mae',
            factor=0.5,
            patience=10,
            min_delta=1e-9,
            mode='min'
        )
        
        model_checkpoint = ModelCheckpoint(
            os.path.join(model_dir, 'best_model.keras'),
            monitor='val_mae',
            save_best_only=True
        )
        
        csv_logger = CSVLogger(os.path.join(performance_dir, 'training_log.csv'))
        
        history = final_model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=10,
            callbacks=[early_stopping, reduce_lr, model_checkpoint, csv_logger],
            verbose=2
        )
        
        final_model = tf.keras.models.load_model(
            os.path.join(model_dir, 'best_model.keras'),
            custom_objects={'Polarization_Lineshape_Loss': Polarization_Lineshape_Loss}
        )
        
        y_test_pred = final_model.predict(test_dataset) 
        residuals = y_test - y_test_pred
        
        rpe = np.abs((y_test - y_test_pred) / np.abs(y_test)) * 100
        
        plot_rpe_and_residuals(y_test, y_test_pred, performance_dir, version)
        plot_enhanced_results(y_test, y_test_pred, performance_dir, version)
        plot_training_history(history, performance_dir, version)
        
        event_results_file = os.path.join(performance_dir, f'test_event_results_{version}.csv')
        test_results_df = pd.DataFrame({
            'Actual': y_test.round(6),
            'Predicted': y_test_pred.round(6),
            'Residuals': residuals.round(6),
            'RPE': rpe.round(6)
        })
        test_results_df.to_csv(event_results_file, index=False)
        
        print(f"Test results saved to {event_results_file}")
        
        save_model_summary(final_model, performance_dir, version)
        
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.savefig(os.path.join(performance_dir, 'optuna_optimization_history.png'))
        
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.savefig(os.path.join(performance_dir, 'optuna_param_importances.png'))
        
        print("Hyperparameter optimization and model training complete!")
    except Exception as e:
        print(f"Error during hyperparameter optimization: {e}")
