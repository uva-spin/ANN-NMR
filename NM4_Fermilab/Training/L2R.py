import sys  
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, regularizers, initializers, optimizers
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
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
    # tf.keras.mixed_precision.set_global_policy('mixed_float64')

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"Using GPU: {physical_devices[0]}")


tf.keras.backend.set_floatx('float64')

# File paths and versioning
data_path = find_file("Shifted_low.csv")  
version = 'Deuteron_Shifted_low_V1'  
performance_dir = f"Model Performance/{version}"  
model_dir = f"Models/{version}"  
os.makedirs(performance_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

print("Loading data...")
data = pd.read_csv(data_path)

train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=1/3, random_state=42)

X_train = train_data.drop(columns=["P", 'SNR']).astype('float64').values
y_train = train_data["P"].astype('float64').values
X_val = val_data.drop(columns=["P", 'SNR']).astype('float64').values
y_val = val_data["P"].astype('float64').values
X_test = test_data.drop(columns=["P", 'SNR']).astype('float64').values
y_test = test_data["P"].astype('float64').values

# Normalize Data
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train).astype('float64')
X_val = scaler.transform(X_val).astype('float64')
X_test = scaler.transform(X_test).astype('float64')

y_test_scaled = np.log(y_test + 1e-10)
y_train_scaled = np.log(y_train + 1e-10)
y_val_scaled = np.log(y_val + 1e-10)



def create_pair_dataset(X, y, num_pairs=10000, focus_on_small=True):
    """
    Create training pairs for learning to rank.
    Each pair consists of two samples where one has a higher target value than the other.
    
    Args:
        X: Input features
        y: Target values
        num_pairs: Number of pairs to generate
        focus_on_small: Whether to focus on small value range
    
    Returns:
        X pairs and binary labels (1 if first sample has higher value)
    """
    n = len(X)
    X_pairs = []
    y_pairs = []
    
    # Create indices for different value ranges if focusing on small values
    if focus_on_small:
        small_indices = np.where(y < 0.01)[0]
        other_indices = np.where(y >= 0.01)[0]
    
    for _ in range(num_pairs):
        if focus_on_small and len(small_indices) > 0:
            # 70% chance to include a small value in the pair
            if np.random.random() < 0.7:
                i = np.random.choice(small_indices)
                j = np.random.choice(n)
            else:
                i = np.random.choice(n)
                j = np.random.choice(n)
        else:
            # Random selection
            i = np.random.choice(n)
            j = np.random.choice(n)
            
        # Skip if the same index
        if i == j:
            continue
        
        # Create pair
        X_pairs.append([X[i], X[j]])
        # Label: 1 if y[i] > y[j], 0 otherwise
        y_pairs.append(1 if y[i] > y[j] else 0)
    
    return np.array(X_pairs), np.array(y_pairs)

def residual_block(x, units, dropout_rate=0.2):
    """
    A PreResNet block with BatchNormalization, activation, Dense layers, and Dropout.
    """
    # Store input for residual connection
    residual = x
    
    # Pre-activation: BN -> ReLU -> Dense (first layer)
    y = tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=0.99)(x)
    y = tf.keras.layers.Activation('swish')(y)
    # Apply dropout after activation but before dense layer
    y = tf.keras.layers.Dropout(dropout_rate)(y)
    y = tf.keras.layers.Dense(units=units, activation=None, use_bias=True,
                              kernel_initializer=tf.keras.initializers.HeNormal(), 
                              bias_initializer=tf.keras.initializers.Zeros(),
                              kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-05, l2=0.0001))(y)
    
    # Pre-activation: BN -> ReLU -> Dense (second layer)
    y = tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=0.99)(y)
    y = tf.keras.layers.Activation('swish')(y)
    # Apply dropout after activation but before dense layer
    y = tf.keras.layers.Dropout(dropout_rate)(y)
    y = tf.keras.layers.Dense(units=units, activation=None, use_bias=True,
                             kernel_initializer=tf.keras.initializers.HeNormal(), 
                             bias_initializer=tf.keras.initializers.Zeros(),
                             kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-05, l2=0.0001))(y)
    
    # Shortcut connection (projection if dimensions don't match)
    if x.shape[-1] != units:
        residual = tf.keras.layers.Dense(units, kernel_initializer=tf.keras.initializers.HeNormal(),
                                  kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-05, l2=0.0001))(x)
    
    # Add residual connection
    output = tf.keras.layers.Add()([residual, y])
    
    return output

def create_scoring_model(input_shape=(500,)):
    """
    Create a scoring model that outputs a scalar score
    """
    inputs = layers.Input(shape=input_shape, dtype='float64')
    
    x = layers.Dense(units=512, activation='swish', use_bias=True,
                    kernel_initializer=tf.keras.initializers.HeNormal(), 
                    bias_initializer=tf.keras.initializers.Zeros(),
                    kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-05, l2=0.0001))(inputs)
    
    x = residual_block(x, units=256, dropout_rate=0.2)
    x = residual_block(x, units=256, dropout_rate=0.2)
    x = residual_block(x, units=256, dropout_rate=0.2)
    
    x = residual_block(x, units=128, dropout_rate=0.2)
    x = residual_block(x, units=128, dropout_rate=0.2)
    
    x = residual_block(x, units=64, dropout_rate=0.2)
    x = residual_block(x, units=64, dropout_rate=0.2)
    
    # Focusing on small values with a specialized branch
    small_values_branch = layers.Dense(units=32, activation='swish')(x)
    small_values_branch = layers.Dense(units=16, activation='swish')(small_values_branch)
    small_values_score = layers.Dense(units=1, activation='sigmoid')(small_values_branch)
    
    # Main scoring branch
    main_score = layers.Dense(units=1, activation='sigmoid')(x)
    
    # Combine scores with attention
    attention = layers.Dense(units=2, activation='softmax')(x)
    final_score = layers.Dot(axes=1)([
        attention, 
        tf.keras.layers.Concatenate(axis=1)([main_score, small_values_score])
    ])
    
    model = tf.keras.Model(inputs=inputs, outputs=final_score)
    return model

def create_pairwise_model(scoring_model):
    """
    Create a pairwise ranking model using the scoring model
    """
    input_1 = layers.Input(shape=(500,), dtype='float64')
    input_2 = layers.Input(shape=(500,), dtype='float64')
    
    # Score each input
    score_1 = scoring_model(input_1)
    score_2 = scoring_model(input_2)
    
    # Compare scores
    diff = layers.Subtract()([score_1, score_2])
    probability = layers.Activation('sigmoid')(diff)
    
    # Create model
    model = tf.keras.Model(inputs=[input_1, input_2], outputs=probability)
    
    return model

def PolarizationRankingModel():
    """
    Create a learning to rank model for the polarization task
    """
    # Create the scoring model
    scoring_model = create_scoring_model()
    
    # Create the pairwise model
    pairwise_model = create_pairwise_model(scoring_model)
    
    # Compile the pairwise model
    cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-4,
        decay_steps=1000,
        alpha=0.0001
    )
    
    optimizer = tf.keras.optimizers.Nadam(
        learning_rate=cosine_decay,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        clipnorm=1.0
    )
    
    pairwise_model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return pairwise_model, scoring_model

# Function to train the model
def train_ranking_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=32, num_pairs=50000):
    # Create pairwise and scoring models
    pairwise_model, scoring_model = PolarizationRankingModel()
    
    # Create training pairs
    X_train_pairs, y_train_pairs = create_pair_dataset(X_train, y_train, num_pairs=num_pairs, focus_on_small=True)
    X_val_pairs, y_val_pairs = create_pair_dataset(X_val, y_val, num_pairs=num_pairs//5, focus_on_small=True)
    
    # Reshape pairs for model input
    X_train_1 = np.array([pair[0] for pair in X_train_pairs])
    X_train_2 = np.array([pair[1] for pair in X_train_pairs])
    X_val_1 = np.array([pair[0] for pair in X_val_pairs])
    X_val_2 = np.array([pair[1] for pair in X_val_pairs])
    
    # Train model
    history = pairwise_model.fit(
        [X_train_1, X_train_2],
        y_train_pairs,
        validation_data=([X_val_1, X_val_2], y_val_pairs),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
        ]
    )
    
    return pairwise_model, scoring_model, history

# Calibrate the scoring model to return values in the correct range
def calibrate_scoring_model(scoring_model, X_cal, y_cal):
    """
    Calibrate the scoring model to output values in the same range as the target
    """
    # Get raw scores
    raw_scores = scoring_model.predict(X_cal).flatten()
    
    # Fit a simple quantile mapping
    from sklearn.isotonic import IsotonicRegression
    
    # Train the isotonic regression model
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(raw_scores, y_cal)
    
    # Create a calibrated prediction function
    def predict_calibrated(X):
        raw_scores = scoring_model.predict(X).flatten()
        return iso_reg.predict(raw_scores)
    
    return predict_calibrated, iso_reg

# Evaluate the model focusing on small values
def evaluate_ranking_model(predict_calibrated, X_test, y_test):
    """
    Evaluate the calibrated model with focus on small values
    """
    # Make predictions
    y_pred = predict_calibrated(X_test)
    
    # Calculate overall metrics
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(np.mean(np.square(y_test - y_pred)))
    
    # Calculate metrics for small values
    small_mask = y_test < 0.01
    small_y_test = y_test[small_mask]
    small_y_pred = y_pred[small_mask]
    
    small_mae = np.mean(np.abs(small_y_test - small_y_pred))
    small_rmse = np.sqrt(np.mean(np.square(small_y_test - small_y_pred)))
    
    # Calculate Spearman correlation (ranking quality)
    from scipy.stats import spearmanr
    corr, _ = spearmanr(y_test, y_pred)
    
    # Calculate small values Spearman correlation
    small_corr, _ = spearmanr(small_y_test, small_y_pred)
    
    # Print results
    print(f"Overall MAE: {mae:.6f}")
    print(f"Overall RMSE: {rmse:.6f}")
    print(f"Small values MAE: {small_mae:.6f}")
    print(f"Small values RMSE: {small_rmse:.6f}")
    print(f"Overall Spearman correlation: {corr:.4f}")
    print(f"Small values Spearman correlation: {small_corr:.4f}")
    
    # Calculate relative error buckets for small values
    rel_errors = np.abs((small_y_test - small_y_pred) / (small_y_test + 1e-10))
    print(f"% within 10% relative error: {100 * np.mean(rel_errors < 0.1):.2f}%")
    print(f"% within 25% relative error: {100 * np.mean(rel_errors < 0.25):.2f}%")
    print(f"% within 50% relative error: {100 * np.mean(rel_errors < 0.5):.2f}%")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'small_mae': small_mae,
        'small_rmse': small_rmse,
        'corr': corr,
        'small_corr': small_corr
    }

# Complete pipeline
def train_and_evaluate():
    # Create and train the models
    pairwise_model, scoring_model, history = train_ranking_model(X_train, y_train, X_val, y_val)
    
    # Calibrate the scoring model
    predict_calibrated, iso_reg = calibrate_scoring_model(scoring_model, X_val, y_val)
    
    # Evaluate the model
    metrics = evaluate_ranking_model(predict_calibrated, X_test, y_test)
    
    return pairwise_model, scoring_model, predict_calibrated, iso_reg, metrics

# Save the models and calibration
def save_models(scoring_model, iso_reg, save_path="ranking_model"):
    # Save the scoring model
    scoring_model.save(f"{save_path}_scoring")
    
    # Save the isotonic regression model
    import joblib
    joblib.dump(iso_reg, f"{save_path}_calibration.joblib")
    
    print(f"Models saved to {save_path}")

# Load the models and calibration
def load_models(load_path="ranking_model"):
    # Load the scoring model
    scoring_model = tf.keras.models.load_model(f"{load_path}_scoring")
    
    # Load the isotonic regression model
    import joblib
    iso_reg = joblib.load(f"{load_path}_calibration.joblib")
    
    # Create the calibrated prediction function
    def predict_calibrated(X):
        raw_scores = scoring_model.predict(X).flatten()
        return iso_reg.predict(raw_scores)
    
    return scoring_model, predict_calibrated, iso_reg

pairwise_model, scoring_model, predict_calibrated, iso_reg, metrics = train_and_evaluate()

# For new data
predictions = predict_calibrated(X_test)

y_test_pred = predictions.flatten()
y_test = y_test.flatten()
residuals = y_test - y_test_pred
rpe = np.abs((y_test - y_test_pred) / np.abs(y_test)) * 100  

plot_rpe_and_residuals(y_test, y_test_pred, performance_dir, version)

plot_enhanced_results(y_test, y_test_pred, performance_dir, version)

# plot_training_history(history, performance_dir, version)

event_results_file = os.path.join(performance_dir, f'test_event_results_{version}.csv')
test_results_df = pd.DataFrame({
    'Actual': y_test.round(6),
    'Predicted': y_test_pred.round(6),
    'Residuals': residuals.round(6),
    'RPE' : rpe.round(6)
})
test_results_df.to_csv(event_results_file, index=False)

print(f"Test results saved to {event_results_file}")

save_model_summary(pairwise_model, performance_dir, version)