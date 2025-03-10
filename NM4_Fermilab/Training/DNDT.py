import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from Custom_Scripts.Misc_Functions import *
from Custom_Scripts.Loss_Functions import *
from Custom_Scripts.Lineshape import *
from Plotting.Plot_Script import *
import random
import pickle

### Let's set a specific seed for benchmarking
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# GPU Configuration
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"Using GPU: {physical_devices[0]}")

# Use float32 instead of float16 to avoid numerical instability
tf.keras.backend.set_floatx('float32')

# File paths and versioning
data_path = find_file("Deuteron_Low_No_Noise_500K.csv")  
version = 'Deuteron_Low_DNDT_V1'  
performance_dir = f"Model Performance/{version}"  
model_dir = f"Models/{version}"  
os.makedirs(performance_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

class DNDT_Regression:
    def __init__(self, 
                 num_features=500, 
                 num_cutpoints=1, 
                 feature_subset_size=50,
                 num_trees=10,
                 learning_rate=0.01,
                 batch_size=64,
                 epochs=100):
        """
        Deep Neural Decision Tree for Regression
        
        Args:
            num_features: Dimensionality of input features
            num_cutpoints: Number of cutpoints per feature (branching factor)
            feature_subset_size: Number of features to randomly select for each tree
            num_trees: Number of trees in the ensemble
            learning_rate: Learning rate for Adam optimizer
            batch_size: Batch size for training
            epochs: Number of training epochs
        """
        self.num_features = num_features
        self.num_cutpoints = num_cutpoints
        self.feature_subset_size = feature_subset_size
        self.num_trees = num_trees
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.trees = []
        self.feature_subsets = []
        
        # Initialize trees and feature subsets
        for i in range(num_trees):
            # Randomly select features for this tree
            feature_subset = np.random.choice(
                num_features, feature_subset_size, replace=False)
            self.feature_subsets.append(feature_subset)
            
            # Create and compile model for this tree
            model = self._build_tree_model(feature_subset_size, i)
            self.trees.append(model)
    
    def _build_tree_model(self, num_features_subset, tree_idx):
        """Build a single tree model"""
        inputs = keras.layers.Input(shape=(self.num_features,))
        
        # Extract the subset of features for this tree
        feature_subset = self.feature_subsets[tree_idx]
        
        # Define the gather function separately to avoid referencing self
        def gather_features(x, indices=feature_subset):
            return tf.gather(x, indices, axis=1)
        
        features_extracted = keras.layers.Lambda(gather_features)(inputs)
        
        # Apply soft binning to each feature
        binned_features = []
        for i in range(num_features_subset):
            # Extract individual feature - define function outside to avoid capturing self
            def extract_feature(x, idx=i):
                return x[:, idx:idx+1]
                
            feature_input = keras.layers.Lambda(extract_feature)(features_extracted)
            
            # Initialize cutpoints with proper constraints
            cutpoints_init = keras.initializers.RandomUniform(0.0, 1.0)
            
            def create_ones(x):
                return x * 0 + 1
                
            cutpoints = keras.layers.Dense(
                self.num_cutpoints, use_bias=False, 
                kernel_initializer=cutpoints_init, 
                name=f'tree{tree_idx}_cutpoints_{i}')(
                keras.layers.Lambda(create_ones)(feature_input))
            
            # Sort cutpoints to ensure they're monotonically increasing
            def sort_tensor(x):
                return tf.sort(x, axis=1)
                
            cutpoints = keras.layers.Lambda(sort_tensor)(cutpoints)
            
            # Create binning based on these cutpoints
            bins = keras.layers.Concatenate()([feature_input, cutpoints])
            
            # Define soft_binning outside the Lambda to avoid capturing self
            def soft_binning(x, num_cutpoints=self.num_cutpoints):
                """
                Stable soft binning function
                Input tensor has shape [batch_size, 1+num_cutpoints]
                where x[:,0] is the feature value, and x[:,1:] are the cutpoints.
                """
                feature_value = x[:, 0:1]  # [batch_size, 1]
                cutpoints = x[:, 1:]      # [batch_size, num_cutpoints]
                
                # Create bin indicators based on the feature value and cutpoints
                num_bins = num_cutpoints + 1
                indicators = []
                
                # First bin: feature < cutpoints[0]
                indicators.append(tf.sigmoid(-(feature_value - cutpoints[:, 0:1]) * 10))
                
                # Middle bins
                for i in range(1, num_cutpoints):
                    left_activation = tf.sigmoid((feature_value - cutpoints[:, i-1:i]) * 10)
                    right_activation = tf.sigmoid(-(feature_value - cutpoints[:, i:i+1]) * 10)
                    indicators.append(left_activation * right_activation)
                
                # Last bin: feature >= cutpoints[-1]
                indicators.append(tf.sigmoid((feature_value - cutpoints[:, -1:]) * 10))
                
                # Concatenate all bin indicators
                all_bins = keras.layers.Concatenate()(indicators)
                
                # Apply softmax with appropriate temperature
                tau = 1.0  # More stable temperature
                soft_bins = tf.nn.softmax(all_bins / tau)
                
                return soft_bins
            
            bins = keras.layers.Lambda(lambda x: soft_binning(x))(bins)
            binned_features.append(bins)
            
        # Combine all binned features
        concat_binned = keras.layers.Concatenate()(binned_features)
        
        # Apply regularization to avoid overfitting
        concat_binned = keras.layers.Dropout(0.2)(concat_binned)
        
        # A series of dense layers to approximate the decision tree structure
        dense1 = keras.layers.Dense(
            256, activation='relu', 
            kernel_regularizer=keras.regularizers.l2(1e-4))(concat_binned)
        dense1 = keras.layers.BatchNormalization()(dense1)
        dense1 = keras.layers.Dropout(0.2)(dense1)
        
        dense2 = keras.layers.Dense(
            128, activation='relu', 
            kernel_regularizer=keras.regularizers.l2(1e-4))(dense1)
        dense2 = keras.layers.BatchNormalization()(dense2)
        dense2 = keras.layers.Dropout(0.2)(dense2)
        
        dense3 = keras.layers.Dense(
            64, activation='relu', 
            kernel_regularizer=keras.regularizers.l2(1e-4))(dense2)
        dense3 = keras.layers.BatchNormalization()(dense3)
        
        # Final regression output with restricted range
        output = keras.layers.Dense(1)(dense3)
        
        model = keras.Model(inputs=inputs, outputs=output)
        
        # Use MAE along with MSE for training stability
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def fit(self, X, y, X_val=None, y_val=None, callbacks=None):
        """Train the DNDT ensemble on the given data"""
        histories = []
        
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # If there's a ModelCheckpoint callback, we need to handle it separately
        tree_callbacks = []
        model_checkpoint = None
        if callbacks:
            for callback in callbacks:
                if isinstance(callback, tf.keras.callbacks.ModelCheckpoint):
                    model_checkpoint = callback
                else:
                    tree_callbacks.append(callback)
        
        best_val_mae = float('inf')
        best_tree_idx = -1
        
        # Ensure model directory exists
        model_dir = os.path.dirname(callbacks[0].filepath) if callbacks and hasattr(callbacks[0], 'filepath') else '.'
        os.makedirs(model_dir, exist_ok=True)
        
        for i, tree in enumerate(self.trees):
            print(f"Training tree {i+1}/{self.num_trees}")
            
            # Train this tree
            h = tree.fit(
                X, y,
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=1,
                validation_data=validation_data,
                callbacks=tree_callbacks
            )
            histories.append(h.history)
            
            # Check if this tree is the best so far
            if validation_data and 'val_mae' in h.history:
                val_mae = h.history['val_mae'][-1]
                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_tree_idx = i
                    # Save the best tree
                    if model_checkpoint:
                        tree_path = os.path.join(model_dir, f'best_tree_{i}.keras')
                        tree.save(tree_path)
                        print(f"Saved best tree {i} with val_mae: {val_mae}")
            
            # Print intermediate results
            val_loss = h.history['val_loss'][-1] if validation_data else "N/A"
            val_mae = h.history['val_mae'][-1] if validation_data else "N/A"
            print(f"Tree {i+1} - Val Loss: {val_loss}, Val MAE: {val_mae}")
        
        # Save the entire ensemble model using pickle
        self.save(os.path.join(model_dir, 'full_ensemble.pkl'))
        
        return histories
    
    def predict(self, X):
        """Make predictions using the DNDT ensemble"""
        predictions = []
        
        for tree in self.trees:
            pred = tree.predict(X, verbose=0)
            predictions.append(pred)
        
        # Average predictions from all trees
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred
    
    def evaluate(self, X, y):
        """Evaluate the model on test data"""
        y_pred = self.predict(X)
        mse = np.mean((y_pred.flatten() - y) ** 2)
        mae = np.mean(np.abs(y_pred.flatten() - y))
        print(f"MSE: {mse:.6f}, MAE: {mae:.6f}")
        return mse, mae
    
    def analyze_feature_importance(self):
        """
        Analyze feature importance based on trained trees
        Returns an array of feature importance scores
        """
        importance = np.zeros(self.num_features)
        
        # Count how many times each feature is used across trees
        for feature_subset in self.feature_subsets:
            importance[feature_subset] += 1
        
        # Normalize
        if np.sum(importance) > 0:
            importance = importance / np.sum(importance)
            
        return importance
    
    def save(self, filepath):
        """Save the ensemble model using pickle"""
        # Create a dictionary with the necessary information
        model_dict = {
            'params': {
                'num_features': self.num_features,
                'num_cutpoints': self.num_cutpoints,
                'feature_subset_size': self.feature_subset_size,
                'num_trees': self.num_trees,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs': self.epochs
            },
            'feature_subsets': self.feature_subsets
        }
        
        # Save each tree separately
        tree_paths = []
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        for i, tree in enumerate(self.trees):
            tree_path = f"{filepath.replace('.pkl', '')}_tree_{i}.keras"
            tree.save(tree_path)
            tree_paths.append(tree_path)
        
        model_dict['tree_paths'] = tree_paths
        
        # Save the model dictionary
        with open(filepath, 'wb') as f:
            pickle.dump(model_dict, f)
            
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load an ensemble model from pickle"""
        with open(filepath, 'rb') as f:
            model_dict = pickle.load(f)
        
        # Create a new instance with the saved parameters
        model = cls(
            num_features=model_dict['params']['num_features'],
            num_cutpoints=model_dict['params']['num_cutpoints'],
            feature_subset_size=model_dict['params']['feature_subset_size'],
            num_trees=model_dict['params']['num_trees'],
            learning_rate=model_dict['params']['learning_rate'],
            batch_size=model_dict['params']['batch_size'],
            epochs=model_dict['params']['epochs']
        )
        
        # Replace feature subsets with the saved ones
        model.feature_subsets = model_dict['feature_subsets']
        
        # Load each tree
        model.trees = []
        for tree_path in model_dict['tree_paths']:
            if os.path.exists(tree_path):
                model.trees.append(keras.models.load_model(tree_path))
            else:
                print(f"Warning: Tree model {tree_path} not found. Skipping.")
        
        return model

# Load and prepare data
print("Loading data...")
data = pd.read_csv(data_path)

train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=1/3, random_state=42)

# Convert to float32 instead of float16 for numerical stability
X_train = train_data.drop(columns=["P", 'SNR']).astype('float32').values
y_train = train_data["P"].astype('float32').values
X_val = val_data.drop(columns=["P", 'SNR']).astype('float32').values
y_val = val_data["P"].astype('float32').values
X_test = test_data.drop(columns=["P", 'SNR']).astype('float32').values
y_test = test_data["P"].astype('float32').values

# Scale the data
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train).astype('float32')
X_val = scaler.transform(X_val).astype('float32')
X_test = scaler.transform(X_test).astype('float32')

# Save the scaler for later use
with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

# Check for NaN values
print(f"NaN in X_train: {np.isnan(X_train).any()}")
print(f"NaN in y_train: {np.isnan(y_train).any()}")

# Setup callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_mae',  
    patience=10,        
    min_delta=1e-4,     
    mode='min',         
    restore_best_weights=True  
)

def cosine_decay_with_warmup(epoch, lr):
    warmup_epochs = 5
    total_epochs = 100
    if epoch < warmup_epochs:
        return lr * (epoch + 1) / warmup_epochs
    else:
        return lr * 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

callbacks_list = [
    early_stopping,
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(model_dir, 'best_tree.keras'),  # This will be handled specially
        monitor='val_mae',
        save_best_only=True),
    tf.keras.callbacks.LearningRateScheduler(cosine_decay_with_warmup),
    tf.keras.callbacks.CSVLogger(os.path.join(performance_dir, 'training_log.csv')),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_mae', 
        factor=0.5, 
        patience=5, 
        min_lr=1e-6)
]

# Define and train the model
dndt = DNDT_Regression(
    num_features=500,
    num_cutpoints=5,
    feature_subset_size=100,
    num_trees=15,
    learning_rate=0.001,  # Start with a lower learning rate
    batch_size=256,       # Smaller batch size for better convergence
    epochs=100            # Allow more epochs with early stopping
)

# Train with validation data
history = dndt.fit(X_train, y_train, X_val, y_val, callbacks=callbacks_list)

# Evaluate on test data
test_mse, test_mae = dndt.evaluate(X_test, y_test)
print(f"Test MSE: {test_mse:.6f}, Test MAE: {test_mae:.6f}")

# Plot predictions vs actual
y_pred = dndt.predict(X_test)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('DNDT Predictions vs Actual Values')
plt.savefig(os.path.join(performance_dir, 'predictions_vs_actual.png'))
plt.close()

# Get feature importance
importance = dndt.analyze_feature_importance()
plt.figure(figsize=(12, 6))
plt.bar(range(500), importance)
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.savefig(os.path.join(performance_dir, 'feature_importance.png'))
plt.close()

print("Training and evaluation completed.")

# Example of how to load the model later:
"""
# Load scaler
with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
    loaded_scaler = pickle.load(f)

# Load model
loaded_model = DNDT_Regression.load(os.path.join(model_dir, 'full_ensemble.pkl'))

# Process and predict on new data
new_data = loaded_scaler.transform(new_data).astype('float32')
new_data = np.clip(new_data, -5, 5)
predictions = loaded_model.predict(new_data)
"""

residuals = y_test - y_pred

rpe = np.abs((y_test - y_pred) / np.abs(y_test)) * 100  

### Plotting the results
# plot_rpe_and_residuals_over_range(y_test, y_test_pred, performance_dir, version)

plot_rpe_and_residuals(y_test, y_pred, performance_dir, version)


plot_training_history(history, performance_dir, version)

event_results_file = os.path.join(performance_dir, f'test_event_results_{version}.csv')
test_results_df = pd.DataFrame({
    'Actual': y_test.round(6),
    'Predicted': y_pred.round(6),
    'Residuals': residuals.round(6)
})
test_results_df.to_csv(event_results_file, index=False)

print(f"Test results saved to {event_results_file}")

save_model_summary(dndt, performance_dir, version)