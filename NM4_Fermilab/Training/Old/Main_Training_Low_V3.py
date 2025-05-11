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
tf.keras.mixed_precision.set_global_policy('float64') 
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"Using GPU: {physical_devices[0]}")

tf.keras.backend.set_floatx('float64')

# File paths and versioning
data_path = find_file("Deuteron_Low_Oversampled_1M.csv")  
version = 'Deuteron_Low_ResNet_Optuna_V2'  
performance_dir = f"Model Performance/{version}"  
model_dir = f"Models/{version}"  
os.makedirs(performance_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

### Defining a custom loss function here for precision
# @tf.keras.utils.register_keras_serializable() 
# class HighPrecisionLoss(tf.keras.losses.Loss):
#     def __init__(self, alpha=1.0, beta=10.0, gamma=5.0, epsilon=1e-10, 
#                  reduction='sum_over_batch_size', name="high_precision_loss"):
#         """
#         Args:
#             alpha (float): Weight for the relative error component
#             beta (float): Weight for the absolute error component
#             gamma (float): Weight for the log-space error component
#             epsilon (float): Small constant to prevent division by zero
#             reduction: Type of reduction to apply to the loss
#             name: Name of the loss function
#         """
#         super().__init__(reduction=reduction, name=name)
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma
#         self.epsilon = epsilon
        
#     def call(self, y_true, y_pred):

#         y_pred = tf.clip_by_value(y_pred, self.epsilon, float('inf'))
        
#         # Relative error component (scale-invariant)
#         relative_error = tf.abs(y_pred - y_true) / (y_true + self.epsilon)
        
#         # Absolute error component (important for very small values)
#         absolute_error = self.beta * tf.abs(y_pred - y_true)
        
#         # Log-space error component (emphasizes relative differences in small values)
#         log_predictions = tf.math.log(y_pred + self.epsilon)
#         log_targets = tf.math.log(y_true + self.epsilon)
#         log_space_error = self.gamma * tf.abs(log_predictions - log_targets)
        
#         combined_loss = (
#             self.alpha * relative_error + 
#             absolute_error + 
#             log_space_error
#         )
        
#         return combined_loss

    # def get_config(self):
    #     config = super().get_config()
    #     config.update({
    #         "alpha": self.alpha,
    #         "beta": self.beta,
    #         "gamma": self.gamma,
    #         "epsilon": self.epsilon
    #     })
    #     return config
    
    
    
@tf.keras.utils.register_keras_serializable() 
def Precision_Loss(y_true, y_pred):
    error = y_true - y_pred
    mean_loss = tf.reduce_mean(tf.abs(error))  # MAE in scaled space
    std_loss = tf.math.reduce_std(error)       # Penalize higher sigma
    
    # Calculate per-sample relative error
    relative_error = tf.abs(error / (y_true + 1e-8))  # Prevent division by zero
    total_relative_loss = tf.reduce_sum(relative_error) 

    return mean_loss + 5 * std_loss  + 5 * 1e-4 * total_relative_loss 


def residual_block(x, units, l1_reg=0.0, l2_reg=0.01, dropout_rate=0.2, Momentum=0.99, Epsilon=1e-5):
    y = layers.BatchNormalization(momentum=Momentum, epsilon=Epsilon)(x)
    y = layers.Dense(units, activation='swish', 
                    kernel_initializer=initializers.HeNormal(),
                    kernel_regularizer=regularizers.L1L2(l1=l1_reg, l2=l2_reg))(y)
    y = layers.Dropout(dropout_rate)(y)
    
    y = layers.Dense(units, activation='swish', 
                    kernel_initializer=initializers.HeNormal(),
                    kernel_regularizer=regularizers.L1L2(l1=l1_reg, l2=l2_reg))(y)
    y = layers.BatchNormalization(momentum=Momentum, epsilon=Epsilon)(y)

    if x.shape[-1] != units:
        x = layers.Dense(units, 
                        kernel_initializer=initializers.HeNormal(),
                        kernel_regularizer=regularizers.L1L2(l1=l1_reg, l2=l2_reg))(x)
    
    x = layers.Add()([x, y])
    x = layers.Activation('swish')(x)
    x = layers.Dropout(dropout_rate)(x)
    return x

def self_attention_block(x, units, num_heads=8, key_dim=64, l1_reg=0.0, l2_reg=0.01):
    """Self-attention block to identify important features in the signal"""
    # Reshape to sequence format for attention
    input_dim = tf.keras.backend.int_shape(x)[-1]
    x_reshaped = layers.Reshape((input_dim, 1))(x)
    
    # Multi-head self attention
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        kernel_regularizer=regularizers.L1L2(l1=l1_reg, l2=l2_reg)
    )(x_reshaped, x_reshaped)
    
    # Residual connection
    attention_output = layers.Add()([x_reshaped, attention_output])
    attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output)
    
    # Flatten back
    attention_output = layers.Reshape((input_dim,))(attention_output)
    
    # Projection
    output = layers.Dense(units, activation='swish', 
                         kernel_initializer=initializers.HeNormal(),
                         kernel_regularizer=regularizers.L1L2(l1=l1_reg, l2=l2_reg))(attention_output)
    
    return output

def feature_attention_module(x, units, l1_reg=0.0, l2_reg=0.01):
    """Feature attention module to assign importance weights to different parts of the signal"""
    # Get attention weights
    attention_weights = layers.Dense(units, activation='softmax', 
                                    kernel_initializer=initializers.HeNormal(),
                                    kernel_regularizer=regularizers.L1L2(l1=l1_reg, l2=l2_reg))(x)
    
    # Apply attention weights
    weighted_features = layers.Multiply()([x, attention_weights])
    
    # Optional: Create a visualization tensor for attention weights
    attention_viz = layers.Dense(1, activation='sigmoid', name='attention_viz')(attention_weights)
    
    return weighted_features, attention_viz

def signal_segmentation_attention(x, num_segments=10, segment_dim=50):
    """Process the signal in segments with attention to identify local features"""
    # Reshape from (batch, 500) to (batch, num_segments, segment_dim)
    # Make sure num_segments * segment_dim = 500
    x_reshaped = layers.Reshape((num_segments, segment_dim))(x)
    
    # Apply attention across segments
    attn_output = layers.MultiHeadAttention(
        num_heads=4, 
        key_dim=32
    )(x_reshaped, x_reshaped)
    
    # Residual connection and normalization
    x_reshaped = layers.Add()([x_reshaped, attn_output])
    x_reshaped = layers.LayerNormalization(epsilon=1e-6)(x_reshaped)
    
    # Process each segment with a shared dense layer
    segment_features = layers.TimeDistributed(
        layers.Dense(64, activation='swish', kernel_initializer=initializers.HeNormal())
    )(x_reshaped)
    
    # Flatten the segments
    output = layers.Flatten()(segment_features)
    
    return output

# def Precision_Loss(y_true, y_pred):
#     """Custom loss function for precision-focused training"""
#     # Mean squared error base
#     mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
    
#     # Add penalty for predictions outside the valid range [0, 1]
#     penalty = tf.reduce_mean(tf.maximum(0.0, tf.abs(y_pred) - 1.0))
    
#     return mse + 10.0 * penalty

def Polarization_Model(params):
    inputs = layers.Input(shape=(500,), dtype='float64')
    
    # Initial normalization
    x = layers.BatchNormalization(momentum=params['Momentum'], epsilon=params['Epsilon'])(inputs)
    
    # Apply noise if specified
    if params.get('input_noise_stddev', 0) > 0:
        x = layers.GaussianNoise(stddev=params.get('input_noise_stddev', 0))(x)
    
    # Option 1: Process the entire signal with self-attention
    if params.get('use_self_attention', False):
        x = self_attention_block(x, params['units_per_layer'][0], 
                                num_heads=params.get('num_attention_heads', 8),
                                key_dim=params.get('attention_key_dim', 64),
                                l1_reg=params['l1_reg'], l2_reg=params['l2_reg'])
    
    # Option 2: Process the signal in segments with attention
    if params.get('use_segment_attention', False):
        # Calculate segment_dim to ensure num_segments * segment_dim = 500
        num_segments = params.get('num_segments', 10)
        segment_dim = 500 // num_segments
        
        segment_features = signal_segmentation_attention(
            inputs, 
            num_segments=num_segments,
            segment_dim=segment_dim
        )
        
        # Combine segment features with previous features if self-attention was used
        if params.get('use_self_attention', False):
            x = layers.Concatenate()([x, segment_features])
        else:
            x = segment_features
    
    # Apply feature attention to identify important parts of the signal
    use_attention_visualization = False
    if params.get('use_feature_attention', False):
        x, attention_viz = feature_attention_module(x, x.shape[-1], 
                                                  l1_reg=params['l1_reg'], 
                                                  l2_reg=params['l2_reg'])
        use_attention_visualization = True
    
    # Apply hidden noise if specified
    if params.get('hidden_noise_stddev', 0) > 0:
        x = layers.GaussianNoise(stddev=params.get('hidden_noise_stddev', 0))(x)
    
    # Standard residual blocks
    for i in range(params['num_layers']):
        if i < len(params['units_per_layer']):
            units = params['units_per_layer'][i]
            x = residual_block(x, units, params['l1_reg'], params['l2_reg'], params['dropout_rate'], 
                              params['Momentum'], params['Epsilon'])
    
    # Apply output noise if specified
    if params.get('output_noise_stddev', 0) > 0:
        x = layers.GaussianNoise(stddev=params.get('output_noise_stddev', 0))(x)
    
    # Final regression output
    outputs = layers.Dense(1, activation='sigmoid', 
                          kernel_initializer=initializers.HeNormal())(x)
    
    # Create model with attention visualization if enabled
    if use_attention_visualization:
        model = tf.keras.Model(inputs=inputs, outputs=[outputs, attention_viz])
    else:
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    optimizer = optimizers.Nadam(
        learning_rate=params['learning_rate'],
        beta_1=params['beta_1'],
        beta_2=params['beta_2'],
        epsilon=params['epsilon'],
        clipnorm=params['clipnorm'],
    )

    # If using attention visualization, configure multiple outputs
    if use_attention_visualization:
        model.compile(
            optimizer=optimizer,
            loss={
                'dense': Precision_Loss,
                'attention_viz': 'mse'  # Just a placeholder, not actually used
            },
            loss_weights={
                'dense': 1.0,
                'attention_viz': 0.0  # Zero weight means it's not trained
            },
            metrics={'dense': tf.keras.metrics.MeanAbsoluteError(name='mae')}
        )
    else:
        model.compile(
            optimizer=optimizer,
            loss=Precision_Loss,
            metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')]
        )
    
    return model

def objective(trial):
    """Optuna objective function for hyperparameter optimization"""
    # Core hyperparameters
    num_layers = trial.suggest_int('num_layers', 1, 6)
    
    # Units per layer
    units_per_layer = []
    for i in range(num_layers):
        units = trial.suggest_int(f'units_{i}', 16, 512)
        units_per_layer.append(units)
    
    # Regularization parameters
    l2_reg = trial.suggest_float('l2_reg', 1e-6, 1e-1, log=True)
    l1_reg = trial.suggest_float('l1_reg', 1e-6, 1e-1, log=True)
    
    # Optimizer parameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    beta_1 = trial.suggest_float('beta_1', 0.8, 0.99)
    beta_2 = trial.suggest_float('beta_2', 0.9, 0.999)
    epsilon = trial.suggest_float('epsilon', 1e-8, 1e-6, log=True)
    clipnorm = trial.suggest_float('clipnorm', 0.1, 2.0)
    
    # Model parameters
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    Momentum = trial.suggest_float('Momentum', 0.7, 0.999)
    Epsilon = trial.suggest_float('Epsilon', 1e-6, 1e-2, log=True)
    
    # Noise parameters (optional)
    hidden_noise_stddev = trial.suggest_float('hidden_noise_stddev', 0.0, 0.1)
    output_noise_stddev = trial.suggest_float('output_noise_stddev', 0.0, 0.1)
    
    # Attention mechanism parameters - this is the new part
    use_self_attention = trial.suggest_categorical('use_self_attention', [True, False])
    use_segment_attention = trial.suggest_categorical('use_segment_attention', [True, False])
    use_feature_attention = trial.suggest_categorical('use_feature_attention', [True, False])
    
    # Only suggest these parameters if the corresponding attention is enabled
    attention_params = {}
    
    if use_self_attention:
        attention_params['num_attention_heads'] = trial.suggest_int('num_attention_heads', 1, 16)
        attention_params['attention_key_dim'] = trial.suggest_int('attention_key_dim', 8, 128)
    
    if use_segment_attention:
        # For signal segmentation, we need num_segments * segment_dim = 500
        # So we choose num_segments from factors of 500 or approximate factors
        possible_segments = [5, 10, 20, 25, 50, 100]
        attention_params['num_segments'] = trial.suggest_categorical('num_segments', possible_segments)
    
    # Combine all parameters
    params = {
        'num_layers': num_layers,
        'units_per_layer': units_per_layer,
        'l2_reg': l2_reg,
        'l1_reg': l1_reg,
        'learning_rate': learning_rate,
        'beta_1': beta_1,
        'beta_2': beta_2,
        'epsilon': epsilon,
        'clipnorm': clipnorm,
        'dropout_rate': dropout_rate,
        'Momentum': Momentum,
        'Epsilon': Epsilon,
        'hidden_noise_stddev': hidden_noise_stddev,
        'output_noise_stddev': output_noise_stddev,
        'use_self_attention': use_self_attention,
        'use_segment_attention': use_segment_attention,
        'use_feature_attention': use_feature_attention,
        **attention_params  # Add the attention parameters if they exist
    }
    
    # Create the model
    model = Polarization_Model(params)
    
    # Setup callbacks
    early_stopping = EarlyStopping(
        monitor='val_mae',
        patience=20,
        min_delta=1e-9,
        mode='min',
        restore_best_weights=True
    )
    
    pruning_callback = TFKerasPruningCallback(trial, 'val_mae')
    
    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=100,  
        batch_size=512,  
        callbacks=[early_stopping, pruning_callback],
        verbose=1  
    )
    
    # Clean up memory
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
scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train).astype('float64')
X_val = scaler.transform(X_val).astype('float64')
X_test = scaler.transform(X_test).astype('float64')

y_train = y_train.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

target_scaler = MinMaxScaler()
y_train_scaled = target_scaler.fit_transform(y_train)
y_val_scaled = target_scaler.transform(y_val)
y_test_scaled = target_scaler.transform(y_test)
print("Data normalized successfully")


batch_size = 256  

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train_scaled))
train_dataset = train_dataset.shuffle(buffer_size=len(X_train))  
train_dataset = train_dataset.batch(batch_size)  
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)  

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val_scaled))
val_dataset = val_dataset.batch(batch_size)  
val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)  

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test_scaled))
test_dataset = test_dataset.batch(batch_size)  
test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)  


# Run Optuna study
if __name__ == "__main__":
    try:
        print("Starting hyperparameter optimization with Optuna...")
        storage = optuna.storages.RDBStorage(
            f"sqlite:///optuna_study_attention_{version}.db",
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
        
        # Prepare best params dictionary
        best_params = {
            'num_layers': num_layers,
            'units_per_layer': units_per_layer,
            'l2_reg': trial.params['l2_reg'],
            'l1_reg': trial.params['l1_reg'],
            'learning_rate': trial.params['learning_rate'],
            'beta_1': trial.params['beta_1'],
            'beta_2': trial.params['beta_2'],
            'epsilon': trial.params['epsilon'],
            'clipnorm': trial.params['clipnorm'],
            'dropout_rate': trial.params['dropout_rate'],
            'Momentum': trial.params['Momentum'],
            'Epsilon': trial.params['Epsilon'],
            'hidden_noise_stddev': trial.params['hidden_noise_stddev'],
            'output_noise_stddev': trial.params['output_noise_stddev'],
            'use_self_attention': trial.params['use_self_attention'],
            'use_segment_attention': trial.params['use_segment_attention'],
            'use_feature_attention': trial.params['use_feature_attention'],
        }
        
        # Add attention-specific parameters if they were used
        if trial.params['use_self_attention']:
            best_params['num_attention_heads'] = trial.params['num_attention_heads']
            best_params['attention_key_dim'] = trial.params['attention_key_dim']
            
        if trial.params['use_segment_attention']:
            best_params['num_segments'] = trial.params['num_segments']
        
        # Save best parameters to file
        with open(os.path.join(performance_dir, 'best_params.json'), 'w') as f:
            json.dump(best_params, f, indent=4)
        
        print("Training final model with best parameters...")
        final_model = Polarization_Model(best_params)
        
        # Setup callbacks for final training
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
        
        # # Prepare dataset objects
        # batch_size = 256
        
        # train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train_scaled))
        # train_dataset = train_dataset.shuffle(buffer_size=len(X_train))  
        # train_dataset = train_dataset.batch(batch_size)  
        # train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)  
        
        # val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val_scaled))
        # val_dataset = val_dataset.batch(batch_size)  
        # val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)  
        
        # test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test_scaled))
        # test_dataset = test_dataset.batch(batch_size)  
        # test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        
        # Train final model
        history = final_model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=100,
            callbacks=[early_stopping, reduce_lr, model_checkpoint, csv_logger],
            verbose=1
        )
        
        # Load best model from checkpoint
        final_model = tf.keras.models.load_model(
            os.path.join(model_dir, 'best_model.keras'),
            custom_objects={'Precision_Loss': Precision_Loss}
        )
        
        # Check if model has multiple outputs (attention visualization)
        is_multi_output = isinstance(final_model.output, list) and len(final_model.output) > 1
        
        # Make predictions
        raw_predictions = final_model.predict(test_dataset)
        
        # Handle potential multiple outputs from attention model
        if is_multi_output:
            y_test_pred = scaler.inverse_transform(raw_predictions[0])  # First output is the prediction
            attention_maps = scaler.inverse_transform(raw_predictions[1])  # Second output is attention visualization
            
            # Save sample attention maps
            plt.figure(figsize=(12, 6))
            for i in range(min(5, len(attention_maps))):
                plt.subplot(1, 5, i+1)
                plt.plot(attention_maps[i])
                plt.title(f"Sample {i+1}")
            plt.suptitle("Attention Maps")
            plt.tight_layout()
            plt.savefig(os.path.join(performance_dir, 'attention_maps_samples.png'))
            plt.close()
        else:
            y_test_pred = scaler.inverse_transform(raw_predictions)
        
        # Calculate metrics
        residuals = y_test - y_test_pred
        rpe = np.abs((y_test - y_test_pred) / np.maximum(np.abs(y_test), 1e-10)) * 100
        
        # Plot results
        plot_rpe_and_residuals(y_test, y_test_pred, performance_dir, version)
        plot_enhanced_results(y_test, y_test_pred, performance_dir, version)
        
        # Save test results
        event_results_file = os.path.join(performance_dir, f'test_event_results_{version}.csv')
        test_results_df = pd.DataFrame({
            'Actual': y_test.flatten().round(6)*100,
            'Predicted': y_test_pred.flatten().round(6)*100,
            'Residuals': residuals.flatten().round(6)*100,
            'RPE': rpe.flatten().round(6)
        })
        test_results_df.to_csv(event_results_file, index=False)
        
        print(f"Test results saved to {event_results_file}")
        
        # Save model summary
        save_model_summary(final_model, performance_dir, version)
        
        # Plot Optuna visualizations
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.savefig(os.path.join(performance_dir, 'optuna_optimization_history.png'))
        
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.savefig(os.path.join(performance_dir, 'optuna_param_importances.png'))
        
        # Plot training history
        plot_training_history(history, performance_dir, version)
        

        attention_weights = final_model.get_layer('attention_viz').output
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(attention_weights)), attention_weights)
        plt.xlabel('Feature Index')
        plt.ylabel('Attention Weight')
        plt.title('Attention Weights for Deuteron Low')
        plt.savefig(os.path.join(performance_dir, 'attention_weights_bar_plot.png'))
        plt.close()
        

        print("Hyperparameter optimization and model training complete!")
    except Exception as e:
        print(f"Error during hyperparameter optimization: {e}")