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

tf.config.optimizer.set_jit(True)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'

physical_devices = tf.config.list_physical_devices('GPU')
try:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
except:
    print("Memory growth setting failed")


data_path = find_file("Deuteron_TE_60_Noisy_Shifted_100K.parquet")  
version = 'Deuteron_TE_60_Noisy_Shifted_100K_CNN_Attention_Separate_Models_V1'  
performance_dir = f"Model_Performance/{version}"  
model_dir = f"Models/{version}"  
os.makedirs(performance_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Define the polarization threshold for classification
P_THRESHOLD = 1.0

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

# Create classification targets
y_train_class = (y_train < P_THRESHOLD).astype(int)
y_val_class = (y_val < P_THRESHOLD).astype(int)
y_test_class = (y_test < P_THRESHOLD).astype(int)

# Split regression data based on threshold
X_train_low = X_train[y_train_class.flatten() == 1]
y_train_low = y_train[y_train_class.flatten() == 1]
X_train_high = X_train[y_train_class.flatten() == 0]
y_train_high = y_train[y_train_class.flatten() == 0]

X_val_low = X_val[y_val_class.flatten() == 1]
y_val_low = y_val[y_val_class.flatten() == 1]
X_val_high = X_val[y_val_class.flatten() == 0]
y_val_high = y_val[y_val_class.flatten() == 0]


def multi_scale_conv_block(x, filters):
    conv3 = layers.Conv1D(filters, 3, padding='same', activation='relu')(x)
    conv5 = layers.Conv1D(filters, 5, padding='same', activation='relu')(x)
    conv7 = layers.Conv1D(filters, 7, padding='same', activation='relu')(x)
    concat = layers.Concatenate()([conv3, conv5, conv7])
    concat = layers.BatchNormalization()(concat)
    return concat

def attention_block(x):
    # Enhanced attention mechanism with multi-head attention
    # For channel attention
    channel_avg_pool = layers.GlobalAveragePooling1D()(x)
    channel_max_pool = layers.GlobalMaxPooling1D()(x)
    
    shared_dense1 = layers.Dense(x.shape[-1] // 4, activation='relu')
    shared_dense2 = layers.Dense(x.shape[-1], activation='sigmoid')
    
    channel_avg_excitation = shared_dense1(channel_avg_pool)
    channel_avg_excitation = shared_dense2(channel_avg_excitation)
    
    channel_max_excitation = shared_dense1(channel_max_pool)
    channel_max_excitation = shared_dense2(channel_max_excitation)
    
    channel_excitation = layers.Add()([channel_avg_excitation, channel_max_excitation])
    channel_excitation = layers.Reshape((1, x.shape[-1]))(channel_excitation)
    
    # Apply attention
    attended = layers.Multiply()([x, channel_excitation])
    attended = layers.LayerNormalization()(attended)
    
    return attended

def residual_block(x, filters, dilation_rate=1):
    shortcut = x
    
    # Add normalization before activation (pre-activation pattern)
    x = layers.LayerNormalization()(x)
    
    # Use dilated convolutions for capturing long-range dependencies
    conv = layers.Conv1D(filters, 3, padding='same', activation='relu', dilation_rate=dilation_rate)(x)
    conv = layers.LayerNormalization()(conv)
    conv = layers.Conv1D(filters, 3, padding='same', dilation_rate=dilation_rate)(conv)
    
    # Add dropout for regularization
    conv = layers.SpatialDropout1D(0.1)(conv)
    
    # Add shortcut connection
    return layers.Add()([shortcut, conv])

def build_feature_extractor(params, input_shape=(500, 1)):
    inputs = Input(shape=input_shape)
    
    # Add explicit data type for higher precision
    inputs_cast = layers.Lambda(lambda x: tf.cast(x, tf.float64))(inputs)
    
    # Use more filters for initial feature extraction
    x = multi_scale_conv_block(inputs_cast, params['filters'])
    
    # Add more residual blocks with different dilation rates for better feature extraction
    for i in range(params['num_residual_blocks']):
        dilation_rate = 2**(i % 3)  # Exponential dilation rates: 1, 2, 4, 1, 2, 4...
        x = residual_block(x, x.shape[-1], dilation_rate=dilation_rate)
    
    # Add an additional attention mechanism for frequency importance
    x = attention_block(x)
    
    # Add a second attention block for refined feature importance
    x = attention_block(x)
    
    # Global pooling with additional context
    max_pool = layers.GlobalMaxPooling1D()(x)
    avg_pool = layers.GlobalAveragePooling1D()(x)
    x = layers.Concatenate()([max_pool, avg_pool])
    
    return inputs, x

def build_classifier_model(params, input_shape=(500, 1)):
    inputs, x = build_feature_extractor(params, input_shape)
    
    x = layers.Dense(params['classifier_units'], activation='relu')(x)
    x = layers.Dense(1, activation='sigmoid', name='classifier')(x)
    
    model = models.Model(inputs=inputs, outputs=x)
    return model

def build_regression_model(params, name_suffix, input_shape=(500, 1)):
    inputs, x = build_feature_extractor(params, input_shape)
    
    # Add more dense layers with higher precision for regression
    x = layers.Dense(params['reg_units'], activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(params['reg_units'] // 2, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Final layer with higher precision for better decimal accuracy
    x = layers.Dense(1, name=f'P_output_{name_suffix}', dtype='float64')(x)
    
    model = models.Model(inputs=inputs, outputs=x)
    return model

def create_dataset(X, y, batch_size, shuffle=True):
    X = tf.cast(X, tf.float32)
    y = tf.cast(y, tf.float32)
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def objective(trial):
    params = {
        'filters': trial.suggest_categorical('filters', [16, 32, 64]),  # Increased filter options
        'classifier_units': trial.suggest_categorical('classifier_units', [32, 64, 128, 256, 512]),  # Increased units
        'reg_units': trial.suggest_categorical('reg_units', [64, 128, 256, 512]),  # Increased units
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-6, 1e-3),  # Lower learning rate range
        'batch_size': trial.suggest_categorical('batch_size', [256, 512, 1024,2048]),
        'num_residual_blocks': trial.suggest_int('num_residual_blocks', 3, 6),  # More residual blocks
        'epochs': 200,  # Increased epochs
    }

    # Build and train classifier model
    classifier_model = build_classifier_model(params)
    train_dataset_class = create_dataset(X_train, y_train_class, params['batch_size'])
    val_dataset_class = create_dataset(X_val, y_val_class, params['batch_size'], shuffle=False)

    # More sophisticated learning rate schedule
    initial_learning_rate = params['learning_rate']
    decay_steps = params['epochs'] * (len(X_train) // params['batch_size'])
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=initial_learning_rate,
        first_decay_steps=decay_steps // 5,
        t_mul=2.0,
        m_mul=0.9,
        alpha=0.01
    )

    classifier_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Improved callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', 
        patience=20,  # Increased patience
        restore_best_weights=True,
        mode='max'
    )

    # Build regression models with custom loss functions
    reg_model_low = build_regression_model(params, "low")
    reg_model_high = build_regression_model(params, "high")
    
    # Create datasets
    train_dataset_low = create_dataset(X_train_low, y_train_low, params['batch_size'])
    val_dataset_low = create_dataset(X_val_low, y_val_low, params['batch_size'], shuffle=False)
    
    train_dataset_high = create_dataset(X_train_high, y_train_high, params['batch_size'])
    val_dataset_high = create_dataset(X_val_high, y_val_high, params['batch_size'], shuffle=False)
    
    # Use specialized loss functions for better precision
    reg_model_low.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8),
        loss=log_cosh_precision_loss,  # Custom loss for better small value precision
        metrics=['mae', scaled_mse]  # Additional metrics
    )
    
    reg_model_high.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8),
        loss=adaptive_weighted_huber_loss,  # Custom loss for high values
        metrics=['mae', scaled_mse]  # Additional metrics
    )

    # Train classifier
    classifier_history = classifier_model.fit(
        train_dataset_class,
        validation_data=val_dataset_class,
        epochs=params['epochs'],
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Train low region model
    low_reg_history = reg_model_low.fit(
        train_dataset_low,
        validation_data=val_dataset_low,
        epochs=params['epochs'],
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Train high region model
    high_reg_history = reg_model_high.fit(
        train_dataset_high,
        validation_data=val_dataset_high,
        epochs=params['epochs'],
        callbacks=[early_stopping],
        verbose=1
    )

    # Get validation predictions for evaluation
    val_class_pred = classifier_model.predict(X_val, verbose=1)
    val_pred_low = reg_model_low.predict(X_val, verbose=1)
    val_pred_high = reg_model_high.predict(X_val, verbose=1)
    
    # Combine predictions based on classifier output
    val_pred = np.where(val_class_pred > 0.5, val_pred_low, val_pred_high)
    
    # Calculate validation MAE
    val_mae = np.mean(np.abs(val_pred - y_val))
    
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
    
    study.optimize(objective, n_trials=10) 

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (val_mae): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    with open(f"{performance_dir}/best_params.json", "w") as f:
        import json
        json.dump(trial.params, f, indent=4)

    # Get best hyperparameters
    best_params = trial.params
    best_params['epochs'] = 200
    
    # Build final models
    classifier_model = build_classifier_model(best_params)
    reg_model_low = build_regression_model(best_params, "low")
    reg_model_high = build_regression_model(best_params, "high")
    
    # Create datasets for final training
    train_dataset_class = create_dataset(X_train, y_train_class, best_params['batch_size'])
    val_dataset_class = create_dataset(X_val, y_val_class, best_params['batch_size'], shuffle=False)
    
    train_dataset_low = create_dataset(X_train_low, y_train_low, best_params['batch_size'])
    val_dataset_low = create_dataset(X_val_low, y_val_low, best_params['batch_size'], shuffle=False)
    
    train_dataset_high = create_dataset(X_train_high, y_train_high, best_params['batch_size'])
    val_dataset_high = create_dataset(X_val_high, y_val_high, best_params['batch_size'], shuffle=False)

    # Configure improved learning rate decay for final models
    initial_learning_rate = best_params['learning_rate']
    decay_steps = best_params['epochs'] * (len(X_train) // best_params['batch_size'])
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=initial_learning_rate,
        first_decay_steps=decay_steps // 5,
        t_mul=2.0,
        m_mul=0.9,
        alpha=0.01
    )

    # Compile models with improved loss functions
    classifier_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    reg_model_low.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8),
        loss=log_cosh_precision_loss,
        metrics=['mae', scaled_mse, tf.keras.metrics.RootMeanSquaredError()]
    )
    
    reg_model_high.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8),
        loss=adaptive_weighted_huber_loss,
        metrics=['mae', scaled_mse, tf.keras.metrics.RootMeanSquaredError()]
    )
    
    # Create callback for logging
    classifier_log_file = f"{performance_dir}/training_log_classifier_model.csv"
    low_reg_log_file = f"{performance_dir}/training_log_low_reg_model.csv"
    high_reg_log_file = f"{performance_dir}/training_log_high_reg_model.csv"
    
    classifier_csv_logger = tf.keras.callbacks.CSVLogger(classifier_log_file, append=True, separator=',')
    low_reg_csv_logger = tf.keras.callbacks.CSVLogger(low_reg_log_file, append=True, separator=',')
    high_reg_csv_logger = tf.keras.callbacks.CSVLogger(high_reg_log_file, append=True, separator=',')
    
    # Train classifier model
    print("Training classifier model...")
    classifier_history = classifier_model.fit(
        train_dataset_class,
        validation_data=val_dataset_class,
        epochs=best_params['epochs'],
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', 
                patience=20, 
                restore_best_weights=True,
                mode='max'
            ),
            classifier_csv_logger
        ],
        verbose=1
    )
    
    # Train low region model
    print("Training low region regression model...")
    low_reg_history = reg_model_low.fit(
        train_dataset_low,
        validation_data=val_dataset_low,
        epochs=best_params['epochs'],
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_mae', 
                patience=20, 
                restore_best_weights=True,
                mode='min'
            ),
            low_reg_csv_logger
        ],
        verbose=1
    )
    
    # Train high region model
    print("Training high region regression model...")
    high_reg_history = reg_model_high.fit(
        train_dataset_high,
        validation_data=val_dataset_high,
        epochs=best_params['epochs'],
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_mae', 
                patience=20, 
                restore_best_weights=True,
                mode='min'
            ),
            high_reg_csv_logger
        ],
        verbose=1
    )

    # Save models
    classifier_model.save(f"{model_dir}/classifier_model.keras")
    reg_model_low.save(f"{model_dir}/low_reg_model.keras")
    reg_model_high.save(f"{model_dir}/high_reg_model.keras")

    # Evaluate on test set
    print("Evaluating models on test set...")
    test_class_pred = classifier_model.predict(X_test)
    test_pred_low = reg_model_low.predict(X_test)
    test_pred_high = reg_model_high.predict(X_test)
    
    # Combine predictions based on classifier output
    test_pred = np.where(test_class_pred > 0.5, test_pred_low, test_pred_high)
    
    # Calculate and print test metrics
    test_mae = np.mean(np.abs(test_pred - y_test))
    print(f"Test MAE: {test_mae}")
    
    # Flatten for plotting
    y_test_flat = y_test.flatten()
    y_pred_flat = test_pred.flatten()

    # Plot the performance metrics and results
    plot_enhanced_performance_metrics(y_test_flat, y_pred_flat, snr_test, performance_dir, version)
    plot_enhanced_results(y_test_flat, y_pred_flat, performance_dir, version)
    
    # Create a function to save a figure with training metrics
    def plot_combined_training_history():
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot classifier accuracy
        axs[0, 0].plot(classifier_history.history['accuracy'])
        axs[0, 0].plot(classifier_history.history['val_accuracy'])
        axs[0, 0].set_title('Classifier Model Accuracy')
        axs[0, 0].set_ylabel('Accuracy')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].legend(['train', 'val'], loc='upper left')
        
        # Plot classifier loss
        axs[0, 1].plot(classifier_history.history['loss'])
        axs[0, 1].plot(classifier_history.history['val_loss'])
        axs[0, 1].set_title('Classifier Model Loss')
        axs[0, 1].set_ylabel('Loss')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].legend(['train', 'val'], loc='upper left')
        
        # Plot regression model MAE
        axs[1, 0].plot(low_reg_history.history['mae'])
        axs[1, 0].plot(low_reg_history.history['val_mae'])
        axs[1, 0].plot(high_reg_history.history['mae'])
        axs[1, 0].plot(high_reg_history.history['val_mae'])
        axs[1, 0].set_title('Regression Models MAE')
        axs[1, 0].set_ylabel('MAE')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].legend(['low train', 'low val', 'high train', 'high val'], loc='upper left')
        
        # Plot regression model loss
        axs[1, 1].plot(low_reg_history.history['loss'])
        axs[1, 1].plot(low_reg_history.history['val_loss'])
        axs[1, 1].plot(high_reg_history.history['loss'])
        axs[1, 1].plot(high_reg_history.history['val_loss'])
        axs[1, 1].set_title('Regression Models Loss')
        axs[1, 1].set_ylabel('Loss')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].legend(['low train', 'low val', 'high train', 'high val'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f"{performance_dir}/training_history.png")
        plt.close()
    
    # Plot training history
    plot_combined_training_history()
    
    # Create a confusion matrix for the classifier
    def plot_classifier_confusion_matrix():
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        
        test_class_pred_binary = (test_class_pred > 0.5).astype(int)
        cm = confusion_matrix(y_test_class, test_class_pred_binary)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['High P', 'Low P'])
        disp.plot(ax=ax)
        plt.title('Classifier Confusion Matrix')
        plt.savefig(f"{performance_dir}/classifier_confusion_matrix.png")
        plt.close()
    
    # Plot classifier confusion matrix
    plot_classifier_confusion_matrix()
    
    # Print completion message
    print("Training and evaluation complete. Results saved to:", performance_dir)



