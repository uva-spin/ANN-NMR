import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers, initializers, optimizers
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import random

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import custom scripts
from Custom_Scripts.Misc_Functions import *
from Custom_Scripts.Loss_Functions import *
from Custom_Scripts.Lineshape import *
from Plotting.Plot_Script import *

### Let's set a specific seed for benchmarking
random.seed(42)


# Set environment variables and configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices --tf_xla_auto_jit=2'
tf.config.optimizer.set_jit(True)
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# GPU configuration
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"Using GPU: {physical_devices[0]}")

tf.keras.backend.set_floatx('float32')

# File paths and versioning
version = 'Deuteron_NeuralDecisionTree_V1'
performance_dir = f"Model Performance/{version}"
model_dir = f"Models/{version}"
os.makedirs(performance_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

#---------------- Neural Decision Tree Implementation ----------------#

class SoftDecisionNode(layers.Layer):
    def __init__(self, units=1, regularization=0.001, **kwargs):
        super(SoftDecisionNode, self).__init__(**kwargs)
        self.units = units
        self.regularization = regularization

    def build(self, input_shape):
        # Create feature selection weights with regularization
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            regularizer=tf.keras.regularizers.l1_l2(l1=0.0, l2=self.regularization),
            trainable=True,
            name='feature_weights'
        )
        # Initialize thresholds from uniform distribution
        self.thresholds = self.add_weight(
            shape=(self.units,),
            initializer=tf.random_uniform_initializer(-0.5, 0.5),
            trainable=True,
            name='thresholds'
        )
        # Temperature parameter for soft decisions
        self.temperature = self.add_weight(
            shape=(1,),
            initializer=tf.constant_initializer(1.0),
            trainable=True,
            name='temperature'
        )
        self.built = True
    
    def call(self, inputs):
        # Compute the weighted sum of features
        feature_output = tf.matmul(inputs, self.w)
        
        # Apply the sigmoid function with temperature scaling
        # to get the probability of going right at this decision node
        decision = tf.sigmoid((feature_output - self.thresholds) / self.temperature)
        
        return decision

    # Add get_config for serialization
    def get_config(self):
        config = super(SoftDecisionNode, self).get_config()
        config.update({
            'units': self.units,
            'regularization': self.regularization
        })
        return config

class NeuralDecisionTreeRegression(Model):
    """
    Deep Neural Decision Tree model for regression tasks
    Maps from a high-dimensional feature space to a continuous output value
    Focuses on high precision for polarization prediction (0-100%)
    """
    def __init__(self, input_dim=500, depth=5, num_leaves=32, hidden_units=[256, 128, 64]):
        super(NeuralDecisionTreeRegression, self).__init__()
        
        self.input_dim = input_dim
        self.depth = depth
        self.num_leaves = num_leaves
        self.hidden_units = hidden_units
        
        # Input normalization layer
        self.normalization = layers.LayerNormalization()
        
        self.projection_layer = layers.Dense(64)

        
        # Feature extraction layers
        self.feature_layers = []
        prev_dim = input_dim
        for units in hidden_units:
            self.feature_layers.append(layers.Dense(
                units, 
                activation='swish',  # Using swish for better performance
                kernel_initializer='he_normal',
                kernel_regularizer=tf.keras.regularizers.l2(0.0001)
            ))
            self.feature_layers.append(layers.BatchNormalization())
            self.feature_layers.append(layers.Dropout(0.1))  # Add dropout to prevent overfitting
            prev_dim = units
        
        # Decision nodes - create one for each internal node in the tree
        self.decision_nodes = []
        for _ in range(self.num_leaves - 1):  # Internal nodes in a binary tree
            self.decision_nodes.append(SoftDecisionNode(regularization=0.0001))
        
        # Leaf node values (regression predictions) - unconstrained for regression
        self.leaf_values = self.add_weight(
            shape=(num_leaves, 1),  # Each leaf outputs a scalar
            initializer='glorot_normal',  # Better for regression
            trainable=True,
            name='leaf_values'
        )
        
        # Final prediction refinement layers with residual connections
        self.refinement_layers = []
        prev_units = 64  # Starting refinement size
        for units in [32, 16]:
            self.refinement_layers.append(
                self._create_residual_block(prev_units, units)
            )
            prev_units = units
            
        # Final output layer - linear activation for regression
        self.final_layer = layers.Dense(
            1, 
            activation=None,
            kernel_initializer=initializers.RandomNormal(stddev=1e-4)
        )

    def _create_residual_block(self, input_units, output_units):
        inputs = layers.Input(shape=(input_units,))
        x = layers.Dense(
            output_units, 
            activation='swish',
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(1e-5)
        )(inputs)
        x = layers.LayerNormalization()(x)
        
        # Add projection if dimensions don't match
        if input_units != output_units:
            shortcut = layers.Dense(output_units, kernel_initializer="he_normal")(inputs)
        else:
            shortcut = inputs
            
        outputs = layers.Add()([x, shortcut])
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def compute_path_probabilities(self, decision_outputs):
        """Compute probabilities of reaching each leaf node"""
        leaf_probs = []
        
        for leaf_idx in range(self.num_leaves):
            # Convert leaf index to binary decision path
            binary_path = format(leaf_idx, f'0{self.depth}b')
            
            # Start with probability 1
            path_prob = tf.ones_like(decision_outputs[0])
            
            # For each decision in the path
            for i, bit in enumerate(binary_path):
                if i < len(decision_outputs):  # Ensure we don't go out of bounds
                    if bit == '0':
                        # Probability of going left
                        path_prob *= (1 - decision_outputs[i])
                    else:
                        # Probability of going right
                        path_prob *= decision_outputs[i]
            
            leaf_probs.append(path_prob)
        
        # Stack all leaf probabilities
        return tf.stack(leaf_probs, axis=1)

    def call(self, inputs):
        # Normalize inputs
        x = self.normalization(inputs)
        
        # Feature extraction
        for layer in self.feature_layers:
            x = layer(x)
        
        # Get decision node outputs
        decision_outputs = [node(x) for node in self.decision_nodes]
        
        # Compute path probabilities to each leaf
        leaf_probabilities = self.compute_path_probabilities(decision_outputs)
        
        # Remove the extra dimension for proper matrix multiplication
        leaf_probabilities = tf.squeeze(leaf_probabilities, axis=-1)
        
        # Compute weighted sum of leaf values
        prediction = tf.matmul(leaf_probabilities, self.leaf_values)
        
        # Extract features from the decision process
        decision_features = tf.concat([tf.reshape(d, [-1, 1]) for d in decision_outputs], axis=1)
        
        # Combine with initial prediction
        combined = tf.concat([prediction, decision_features], axis=1)
        
        # Pass through refinement layers (residual blocks)
        refined = combined
        current_shape = combined.shape[-1]
        
        # Apply refinement with residual blocks
        for i, residual_block in enumerate(self.refinement_layers):
            if i == 0:
                # For the first block, we need to project to the expected input size
                projection = self.projection_layer(refined)
                refined = residual_block(projection)
            else:
                refined = residual_block(refined)
        
        # Final prediction
        final_pred = self.final_layer(refined)
        
        return final_pred

    # Add get_config for serialization
    def get_config(self):
        config = super(NeuralDecisionTreeRegression, self).get_config()
        config.update({
            'input_dim': self.input_dim,
            'depth': self.depth,
            'num_leaves': self.num_leaves,
            'hidden_units': self.hidden_units
        })
        return config

#---------------- Loss Functions ----------------#

def log_cosh_precision_loss(y_true, y_pred):
    """
    Custom loss combining LogCosh (smooth L1/L2) with MSE for precision tasks
    Handles both large errors and enforces precision on small differences
    """
    # Log-cosh for smooth L1/L2 characteristics
    log_cosh = tf.math.log(tf.math.cosh(y_pred - y_true))
    
    # MSE component for precision
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Relative error penalty to enforce precision across the value range
    epsilon = 1e-6  # Small value to avoid division by zero
    relative_error = tf.abs((y_true - y_pred) / (tf.abs(y_true) + epsilon))
    relative_penalty = tf.reduce_mean(relative_error)
    
    # Combine with different weights
    return tf.reduce_mean(log_cosh) + 0.5 * mse + 0.3 * relative_penalty

def create_polarization_model(input_dim=500, depth=4, num_leaves=16, hidden_units=[256, 128, 64]):
    """
    Create and compile a Neural Decision Tree model for polarization prediction
    """
    # Calculate number of leaves based on depth for a binary tree if not specified
    if num_leaves is None:
        num_leaves = 2**depth
    
    model = NeuralDecisionTreeRegression(
        input_dim=input_dim,
        depth=depth,
        num_leaves=num_leaves,
        hidden_units=hidden_units
    )
    
    # Use Nadam optimizer with warmup and decay for better convergence
    optimizer = optimizers.Nadam(
        learning_rate=5e-5,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        clipnorm=1.0  # Gradient clipping to prevent exploding gradients
    )
    
    model.compile(
        optimizer=optimizer,
        loss=log_cosh_precision_loss,
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name='mae'),
            tf.keras.metrics.RootMeanSquaredError(name='rmse')
        ]
    )
    
    # Build model with dummy data to initialize weights
    dummy_input = tf.zeros((1, input_dim))
    _ = model(dummy_input)
    
    return model

#---------------- Visualization Functions ----------------#

def plot_rpe_and_residuals(y_true, y_pred, performance_dir, version):
    """Plot residuals and relative percentage error"""
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Residuals
    residuals = y_true - y_pred
    axs[0, 0].scatter(y_true, residuals, alpha=0.5)
    axs[0, 0].axhline(y=0, color='r', linestyle='-')
    axs[0, 0].set_xlabel('True Value')
    axs[0, 0].set_ylabel('Residuals')
    axs[0, 0].set_title('Residuals vs True Value')
    
    # Histogram of residuals
    axs[0, 1].hist(residuals, bins=50)
    axs[0, 1].set_xlabel('Residual')
    axs[0, 1].set_ylabel('Frequency')
    axs[0, 1].set_title('Distribution of Residuals')
    
    # True vs Predicted
    axs[1, 0].scatter(y_true, y_pred, alpha=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axs[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--')
    axs[1, 0].set_xlabel('True Value')
    axs[1, 0].set_ylabel('Predicted Value')
    axs[1, 0].set_title('True vs Predicted Values')
    
    # Relative Percentage Error
    rpe = np.abs((y_true - y_pred) / np.abs(y_true)) * 100
    axs[1, 1].scatter(y_true, rpe, alpha=0.5)
    axs[1, 1].set_xlabel('True Value')
    axs[1, 1].set_ylabel('Relative Percentage Error (%)')
    axs[1, 1].set_title('Relative Percentage Error vs True Value')
    axs[1, 1].set_yscale('log')  # Log scale for better visualization
    
    plt.tight_layout()
    plt.savefig(os.path.join(performance_dir, f'{version}_residuals_rpe.png'))
    plt.close()

def plot_training_history(history, performance_dir, version):
    """Plot training history metrics"""
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axs[0].plot(history.history['loss'])
    axs[0].plot(history.history['val_loss'])
    axs[0].set_title('Model Loss')
    axs[0].set_ylabel('Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['Train', 'Validation'], loc='upper right')
    
    # Plot MAE
    axs[1].plot(history.history['mae'])
    axs[1].plot(history.history['val_mae'])
    axs[1].set_title('Mean Absolute Error')
    axs[1].set_ylabel('MAE')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(performance_dir, f'{version}_training_history.png'))
    plt.close()

def save_model_summary(model, performance_dir, version):
    """Save model summary to file"""
    with open(os.path.join(performance_dir, f'{version}_model_summary.txt'), 'w') as f:
        # Redirect stdout to the file
        original_stdout = sys.stdout
        sys.stdout = f
        model.summary()
        sys.stdout = original_stdout

def cosine_decay_with_warmup(epoch, lr):
    """Learning rate scheduler with warmup and cosine decay"""
    warmup_epochs = 5
    total_epochs = 1000
    if epoch < warmup_epochs:
        return lr * (epoch + 1) / warmup_epochs
    else:
        return lr * 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

def evaluate_decimal_precision(model, X_test, y_test):
    """Evaluate model precision at different decimal places"""
    y_pred = model.predict(X_test).flatten()
    
    # Calculate precision at different decimal places
    precision_results = {}
    for decimal_places in range(1, 7):  # Check up to 6 decimal places
        scale = 10 ** decimal_places
        y_true_rounded = np.round(y_test * scale) / scale
        y_pred_rounded = np.round(y_pred * scale) / scale
        precision = np.mean(y_true_rounded == y_pred_rounded)
        precision_results[decimal_places] = precision
    
    return precision_results, y_pred

#---------------- Main Execution ----------------#

if __name__ == "__main__":
    # Load data
    try:
        # Try to find data file - adjust this function based on your environment
        # data_path = os.environ.get('DATA_PATH') or "Deuteron_Low_No_Noise_500K.csv"
        data_path = find_file("Deuteron_Low_No_Noise_500K.csv")
        
        print(f"Loading data from {data_path}...")
        data = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure the data file exists and path is correct.")
        sys.exit(1)

    # Split data
    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=1/3, random_state=42)

    # Prepare features and targets
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

    # Save scaler for production use
    import joblib
    joblib.dump(scaler, os.path.join(model_dir, 'feature_scaler.pkl'))

    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_mae',
        patience=50,
        min_delta=1e-9,
        mode='min',
        restore_best_weights=True
    )

    callbacks_list = [
        early_stopping,
        ModelCheckpoint(
            os.path.join(model_dir, 'best_model.keras'),
            monitor='val_mae',
            save_best_only=True
        ),
        ReduceLROnPlateau(
            monitor='val_mae',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
        LearningRateScheduler(cosine_decay_with_warmup),
        CSVLogger(os.path.join(performance_dir, 'training_log.csv'))
    ]

    # Create and train model
    print("\nCreating Neural Decision Tree model for polarization prediction...")
    model = create_polarization_model(
        input_dim=X_train.shape[1],
        depth=4,  # Reduced depth to prevent overfitting
        num_leaves=16,  # Fewer leaves
        hidden_units=[256, 128, 64, 32]  # Added more layers for feature extraction
    )

    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,  # Increased epochs with early stopping
        batch_size=128,  # Reduced batch size for better generalization
        callbacks=callbacks_list,
        verbose=2
    )

    # Load best model
    model = tf.keras.models.load_model(os.path.join(model_dir, 'best_model.keras'), 
                                       custom_objects={'SoftDecisionNode': SoftDecisionNode,
                                                      'NeuralDecisionTreeRegression': NeuralDecisionTreeRegression,
                                                      'log_cosh_precision_loss': log_cosh_precision_loss})

    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_mae, test_rmse = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    print(f"Test RMSE: {test_rmse:.6f}")

    # Detailed precision evaluation
    precision_results, y_test_pred = evaluate_decimal_precision(model, X_test, y_test)
    print("\nPrecision at different decimal places:")
    for places, precision in precision_results.items():
        print(f"{places} decimal places: {precision:.4f}")

    # Show sample predictions
    print("\nSample predictions (Actual vs. Predicted):")
    for i in range(10):
        print(f"Sample {i+1}: Actual = {y_test[i]:.6f}, Predicted = {y_test_pred[i]:.6f}, "
              f"Absolute Error = {abs(y_test[i] - y_test_pred[i]):.6f}")

    # Generate visualization plots
    residuals = y_test - y_test_pred
    plot_rpe_and_residuals(y_test, y_test_pred, performance_dir, version)
    plot_training_history(history, performance_dir, version)

    # Save test results to CSV
    print("\nSaving test results...")
    test_results_df = pd.DataFrame({
        'Actual': y_test.round(6),
        'Predicted': y_test_pred.round(6),
        'Absolute_Error': np.abs(y_test - y_test_pred).round(6),
        'Relative_Error_Pct': (np.abs(y_test - y_test_pred) / np.abs(y_test) * 100).round(4)
    })
    test_results_file = os.path.join(performance_dir, f'test_results_{version}.csv')
    test_results_df.to_csv(test_results_file, index=False)
    print(f"Test results saved to {test_results_file}")

    # Save model summary
    save_model_summary(model, performance_dir, version)
    print(f"\nModel performance analysis saved to {performance_dir}")