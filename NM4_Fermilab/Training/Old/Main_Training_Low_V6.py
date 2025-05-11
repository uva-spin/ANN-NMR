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
from tensorflow.keras.optimizers.schedules import CosineDecay
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
tf.config.optimizer.set_jit(True)

tf.keras.mixed_precision.set_global_policy('float64') 
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], False)
    print(f"Using GPU: {physical_devices[0]}")

tf.keras.backend.set_floatx('float64')

# File paths and versioning
data_path = find_file("Deuteron_Oversampled_1M.csv")  
version = 'Deuteron_Low_ResNet_Optuna_V6'  
performance_dir = f"Model Performance/{version}"  
model_dir = f"Models/{version}"  
os.makedirs(performance_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

## Defining a custom loss function here for precision
@tf.keras.utils.register_keras_serializable() 
class HighPrecisionLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=1.0, beta=10.0, gamma=5.0, epsilon=1e-10, 
                 reduction='sum_over_batch_size', name="enhanced_high_precision_loss"):
        """
        Enhanced loss function focused on high precision regression for small values.
        
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
        
        value_importance = 1.0 / (y_true + self.epsilon)
        
        relative_error = value_importance * tf.abs(y_pred - y_true) / (y_true + self.epsilon)
        
        absolute_error = self.beta * tf.abs(y_pred - y_true)
        
        log_predictions = tf.math.log(y_pred + self.epsilon)
        log_targets = tf.math.log(y_true + self.epsilon)
        log_space_error = self.gamma * tf.abs(log_predictions - log_targets)
        
        squared_log_error = tf.square(log_predictions - log_targets)
        
        combined_loss = (
            self.alpha * relative_error + 
            absolute_error + 
            log_space_error +
            squared_log_error
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



def residual_block(x, units, l1_reg=0.0, l2_reg=0.01, dropout_rate=0.2, Momentum=0.99, Epsilon=1e-5):
    
    x = layers.Dense(units, activation=layers.PReLU(), 
                    kernel_initializer=initializers.HeNormal(),
                    kernel_regularizer=regularizers.L1L2(l1=l1_reg, l2=l2_reg))(x)

    y = layers.BatchNormalization(momentum=Momentum, epsilon=Epsilon)(x)
    y = layers.Dense(units, activation=layers.PReLU(), 
                    kernel_initializer=initializers.HeNormal(),
                    kernel_regularizer=regularizers.L1L2(l1=l1_reg, l2=l2_reg))(y)
    y = layers.Dropout(dropout_rate)(y)
            
    y = layers.Dense(units, activation=None, 
                    kernel_initializer=initializers.HeNormal(),
                    kernel_regularizer=regularizers.L1L2(l1=l1_reg, l2=l2_reg))(y)
    y = layers.BatchNormalization(momentum=Momentum, epsilon=Epsilon)(y)

    if x.shape[-1] != units:
        x = layers.Dense(units, 
                        kernel_initializer=initializers.HeNormal(),
                        kernel_regularizer=regularizers.L1L2(l1=l1_reg, l2=l2_reg))(x)
    
    x = layers.Add()([x, y])
    x = layers.Activation(layers.PReLU())(x)
    x = layers.Dropout(dropout_rate)(x)
    return x


def residual_block_for_small_values(x, units, l1_reg=0.0, l2_reg=0.01, dropout_rate=0.2):


    y = layers.Dense(units // 2, activation=layers.PReLU(),
                    kernel_initializer=initializers.HeNormal(),
                    kernel_regularizer=regularizers.L1L2(l1=l1_reg, l2=l2_reg))(x)
                    
    y = layers.Dense(units // 4, activation=layers.PReLU(),
                    kernel_initializer=initializers.HeNormal(),
                    kernel_regularizer=regularizers.L1L2(l1=l1_reg, l2=l2_reg))(y)
    
    y = layers.Dense(units // 2, activation=None,
                    kernel_initializer=initializers.HeNormal(),
                    kernel_regularizer=regularizers.L1L2(l1=l1_reg, l2=l2_reg))(y)
    
    if x.shape[-1] != units // 2:
        x = layers.Dense(units // 2, 
                      kernel_initializer=initializers.HeNormal(),
                      kernel_regularizer=regularizers.L1L2(l1=l1_reg, l2=l2_reg))(x)
    
    combined = layers.Add()([x, y])
    combined = layers.Activation(layers.PReLU())(combined)
    combined = layers.Dropout(dropout_rate)(combined)
    
    return combined


def Polarization_Model(params):
    inputs = layers.Input(shape=(500,), dtype='float64')
    
    x = layers.BatchNormalization(momentum=params['Momentum'], epsilon=params['Epsilon'])(inputs)
    
    x = layers.Dense(params['initial_units'], activation=layers.PReLU(), 
                    kernel_initializer=initializers.HeNormal(),
                    kernel_regularizer=regularizers.L1L2(l1=params['l1_reg_1'], l2=params['l2_reg_1']))(x)
    
    for i in range(params['num_layers']):
        units = params['units_per_layer_1'][i]
        x = residual_block(x, units, params['l1_reg_2'], params['l2_reg_2'], params['dropout_rate'], 
                          params['Momentum'], params['Epsilon'])
    
    small_value_branch = layers.Dense(params['initial_units'] // 2, activation=layers.PReLU())(x)
    
    for i in range(2): 
        small_value_branch = residual_block_for_small_values(
            small_value_branch, 
            params['units_per_layer_2'][i], 
            params['l1_reg_3'], 
            params['l2_reg_3'], 
            params['dropout_rate']
        )
    
    x = layers.Concatenate()([x, small_value_branch])
    
    x = layers.Dense(params['final_units'] // 2, activation=layers.PReLU())(x)
    x = layers.Dense(params['final_units'] // 4, activation=layers.PReLU())(x)
    
    outputs = layers.Dense(1, activation='exponential',
                         kernel_initializer=initializers.HeNormal())(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    scheduler = CosineDecay(
        initial_learning_rate=params['learning_rate'],
        decay_steps=100,
        alpha=0.001
    )
    
    optimizer = optimizers.Nadam(
        learning_rate=scheduler,
        beta_1=params['beta_1'],
        beta_2=params['beta_2'],
        epsilon=params['epsilon'],
        clipnorm=params['clipnorm'],
    )
    
    loss = HighPrecisionLoss(
        alpha=params['alpha'], 
        beta=params['beta'], 
        gamma=params['gamma']
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')]
    )
    return model

def objective(trial):
    num_layers = trial.suggest_int('num_layers', 1, 6)
    
    units_per_layer = []
    for i in range(num_layers):
        units = trial.suggest_int(f'units_{i}', 16, 512)
        units_per_layer.append(units)
    
    
    params = {
        'num_layers': num_layers,
        'units_per_layer_1': units_per_layer, 
        'units_per_layer_2': units_per_layer,
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'beta_1': trial.suggest_float('beta_1', 0.8, 0.99),
        'beta_2': trial.suggest_float('beta_2', 0.9, 0.999),
        'epsilon': trial.suggest_float('epsilon', 1e-8, 1e-6, log=True),
        'clipnorm': trial.suggest_float('clipnorm', 0.1, 2.0),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.5),
        'initial_units': trial.suggest_int('initial_units', 256, 1024),
        'final_units': trial.suggest_int('final_units', 16, 1024),
        'l2_reg_1': trial.suggest_float('l2_reg_1', 1e-6, 1e-1, log=True),
        'l1_reg_1': trial.suggest_float('l1_reg_1', 1e-6, 1e-1, log=True),
        'l2_reg_2': trial.suggest_float('l2_reg_2', 1e-6, 1e-1, log=True),
        'l1_reg_2': trial.suggest_float('l1_reg_2', 1e-6, 1e-1, log=True),
        'l2_reg_3': trial.suggest_float('l2_reg_3', 1e-6, 1e-1, log=True),
        'l1_reg_3': trial.suggest_float('l1_reg_3', 1e-6, 1e-1, log=True),
        'Momentum': trial.suggest_float('Momentum', 0.7, 0.999),
        'Epsilon': trial.suggest_float('Epsilon', 1e-6, 1e-2, log=True),
        'alpha': trial.suggest_float('alpha', 0.0, 10.0),
        'beta': trial.suggest_float('beta', 0.0, 10.0),
        'gamma': trial.suggest_float('gamma', 0.0, 10.0),
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
        batch_size=1024,  
        callbacks=[early_stopping, pruning_callback],
        verbose=1  
    )
    
    tf.keras.backend.clear_session()
    gc.collect()  

    
    return history.history['val_mae'][-1]

print("Loading data...")
try:
    data = pd.read_csv(data_path)
    print("Data loaded successfully!")
except Exception as e:
    print(f"Error loading data: {e}")
data = data[data['P'] <= 0.01]
print(f"Number of samples: {len(data)}")
    
    


print("Splitting data...")
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=1/3, random_state=42)
print("Data split successfully")

print("Preparing features and targets...")
X_train = train_data.drop(columns=["P", 'SNR']).astype('float64').values
y_train = train_data["P"].astype('float64').values
X_val = val_data.drop(columns=["P", 'SNR']).astype('float64').values
y_val = val_data["P"].astype('float64').values
X_test = test_data.drop(columns=["P", 'SNR']).astype('float64').values
y_test = test_data["P"].astype('float64').values
print("Features and targets prepared successfully")

print("Normalizing data...")


# scaler = StandardScaler()

# X_train = scaler.fit_transform(X_train).astype('float64')
# X_val = scaler.transform(X_val).astype('float64')
# X_test = scaler.transform(X_test).astype('float64')


X_train_log = np.log(X_train + 1e-10)
X_val_log = np.log(X_val + 1e-10)
X_test_log = np.log(X_test + 1e-10)

y_train = y_train.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

epsilon = 1e-10  
y_train_log = np.log(y_train + epsilon)
y_val_log = np.log(y_val + epsilon)
y_test_log = np.log(y_test + epsilon)

y_train_original = y_train.copy()
y_val_original = y_val.copy()
y_test_original = y_test.copy()

sample_weights = 1.0 / (y_train + epsilon)
sample_weights = sample_weights / np.mean(sample_weights)  



print("Data normalized successfully")


batch_size = 256  

train_dataset = tf.data.Dataset.from_tensor_slices((X_train_log, y_train_log))
train_dataset = train_dataset.shuffle(buffer_size=len(X_train))  
train_dataset = train_dataset.batch(batch_size)  
train_dataset = train_dataset.cache()
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)  

val_dataset = tf.data.Dataset.from_tensor_slices((X_val_log, y_val_log))
val_dataset = val_dataset.batch(batch_size) 
val_dataset = val_dataset.cache()
val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)  

test_dataset = tf.data.Dataset.from_tensor_slices((X_test_log, y_test_log))
test_dataset = test_dataset.batch(batch_size)  
test_dataset = test_dataset.cache()
test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)  


# Run Optuna study
if __name__ == "__main__":
    try:
        print("Starting hyperparameter optimization with Optuna...")
        storage = optuna.storages.RDBStorage(
            f"sqlite:///optuna_study_{version}.db",
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
        study.optimize(objective, n_trials=50)  
        
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
            'initial_units': trial.params['initial_units'],
            'final_units': trial.params['final_units'],
            'l2_reg_1': trial.params['l2_reg_1'],
            'l1_reg_1': trial.params['l1_reg_1'],
            'l2_reg_2': trial.params['l2_reg_2'],
            'l1_reg_2': trial.params['l1_reg_2'],
            'l2_reg_3': trial.params['l2_reg_3'],
            'l1_reg_3': trial.params['l1_reg_3'],
            'learning_rate': trial.params['learning_rate'],
            'beta_1': trial.params['beta_1'],
            'beta_2': trial.params['beta_2'],
            'epsilon': trial.params['epsilon'],
            'clipnorm': trial.params['clipnorm'],
            'dropout_rate': trial.params['dropout_rate'],
            'Momentum': trial.params['Momentum'],
            'Epsilon': trial.params['Epsilon'],
            'alpha': trial.params['alpha'],
            'beta': trial.params['beta'],
            'gamma': trial.params['gamma'],
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
        
        model_checkpoint = ModelCheckpoint(
            os.path.join(model_dir, 'best_model.keras'),
            monitor='val_mae',
            save_best_only=True
        )
        
        csv_logger = CSVLogger(os.path.join(performance_dir, 'training_log.csv'))
        
        history = final_model.fit(
            train_dataset,
            validation_data=val_dataset,
            sample_weights=sample_weights,
            epochs=100,
            batch_size=1024,
            callbacks=[early_stopping, model_checkpoint, csv_logger],
            verbose=1
        )
        
        final_model = tf.keras.models.load_model(
            os.path.join(model_dir, 'best_model.keras'),
            custom_objects={'Polarization_Lineshape_Loss': Polarization_Lineshape_Loss}
        )
        
        y_test_pred = final_model.predict(test_dataset) 
        y_test_pred = np.exp(y_test_pred).flatten()
        y_test = y_test_original.flatten()
        residuals = y_test - y_test_pred
        
        rpe = np.abs((y_test - y_test_pred) / np.abs(y_test)) * 100
        
        plot_rpe_and_residuals(y_test, y_test_pred, performance_dir, version)
        plot_enhanced_results(y_test, y_test_pred, performance_dir, version)
                
        event_results_file = os.path.join(performance_dir, f'test_event_results_{version}.npy')

        results = {
            'Actual': y_test.flatten().round(6) * 100, 
            'Predicted': y_test_pred.flatten().round(6) * 100, 
            'Residuals': residuals.flatten().round(6) * 100,  
            'RPE': rpe.flatten().round(6)  
        }

        np.save(event_results_file, results)

        print(f"Test results saved to {event_results_file}")
        
        save_model_summary(final_model, performance_dir, version)
        
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.savefig(os.path.join(performance_dir, 'optuna_optimization_history.png'))
        
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.savefig(os.path.join(performance_dir, 'optuna_param_importances.png'))
        
        plot_training_history(history, performance_dir, version)

        print("Hyperparameter optimization and model training complete!")
    except Exception as e:
        print(f"Error during hyperparameter optimization: {e}")