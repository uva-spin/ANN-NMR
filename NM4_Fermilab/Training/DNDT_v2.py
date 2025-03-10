import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model, constraints
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

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

class NeuralDecisionTreeRegression(Model):
    """
    Deep Neural Decision Tree model for regression tasks
    Maps from a high-dimensional feature space to a continuous output value
    """
    def __init__(self, input_dim=500, depth=5, num_leaves=32, hidden_units=[256, 128, 64]):
        super(NeuralDecisionTreeRegression, self).__init__()
        
        self.input_dim = input_dim
        self.depth = depth
        self.num_leaves = num_leaves
        
        # Feature extraction layers
        self.feature_layers = []
        prev_dim = input_dim
        for units in hidden_units:
            self.feature_layers.append(layers.Dense(units, activation='relu',
                                                  kernel_regularizer=tf.keras.regularizers.l2(0.001)))
            self.feature_layers.append(layers.BatchNormalization())
            prev_dim = units
        
        # Decision nodes - create one for each internal node in the tree
        self.decision_nodes = []
        for _ in range(self.num_leaves - 1):  # Internal nodes in a binary tree
            self.decision_nodes.append(SoftDecisionNode())
        
        # Leaf node values (regression predictions) - unconstrained for regression
        self.leaf_values = self.add_weight(
            shape=(num_leaves, 1),  # Each leaf outputs a scalar
            initializer='glorot_normal',  # Better for regression
            trainable=True,
            name='leaf_values'
        )
        
        # Final prediction refinement layers
        self.refinement_layers = [
            layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            # Linear activation for regression
            layers.Dense(1, activation=None)
        ]

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
        # Feature extraction
        x = inputs
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
        
        # Refine prediction
        for layer in self.refinement_layers:
            prediction = layer(prediction)
        
        return prediction

# Custom loss function that handles both large and small differences well
def regression_huber_mse_loss(y_true, y_pred):
    # Combine MSE (sensitive to small differences) with Huber (robust to outliers)
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Huber loss to handle potential outliers
    delta = 1.0
    huber = tf.keras.losses.Huber(delta=delta)
    huber_loss = huber(y_true, y_pred)
    
    return 0.7 * mse + 0.3 * huber_loss

# Function to create and compile the model
def create_neural_decision_tree_regression(
    input_dim=500,
    depth=5,
    num_leaves=None,
    hidden_units=[256, 128, 64],
    learning_rate=0.001
):
    # If num_leaves not specified, calculate based on depth for a binary tree
    if num_leaves is None:
        num_leaves = 2**depth
    
    model = NeuralDecisionTreeRegression(
        input_dim=input_dim,
        depth=depth,
        num_leaves=num_leaves,
        hidden_units=hidden_units
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss=regression_huber_mse_loss,
        metrics=['mae', 'mse', tf.keras.metrics.RootMeanSquaredError()]
    )
    
    # Build model with dummy data to initialize weights
    dummy_input = tf.zeros((1, input_dim))
    _ = model(dummy_input)
    
    return model

# Example usage
def train_regression_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    model = create_neural_decision_tree_regression(
        input_dim=X_train.shape[1],  # Automatically determine from data
        depth=5,  # Tree depth
        hidden_units=[256, 128, 64]  # Feature extraction layers
    )
    
    # Define callbacks for model training
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            mode='min',
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_ndt_regression_model.keras',
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        )
    ]
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

# Example of how to generate synthetic data for testing
def generate_synthetic_data(num_samples=1000, input_dim=500):
    X = np.random.normal(0, 1, (num_samples, input_dim))
    
    # Create a non-linear function that requires high precision
    beta = np.random.normal(0, 1, (input_dim,))
    beta = beta / np.linalg.norm(beta)  # Normalize weights
    
    # Generate target values between 0 and 1 with high precision requirements
    y_linear = np.dot(X, beta)
    y = 1.0 / (1.0 + np.exp(-y_linear))  # Sigmoid to bound between 0 and 1
    
    # Add noise at the 5th decimal place
    y += np.random.normal(0, 0.00001, y.shape)
    
    # Ensure values stay within [0, 1]
    y = np.clip(y, 0, 1)
    
    return X, y.reshape(-1, 1)

# Function to evaluate model precision at different decimal places
def evaluate_decimal_precision(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    # Calculate precision at different decimal places
    precision_results = {}
    for decimal_places in range(1, 6):
        scale = 10 ** decimal_places
        y_true_rounded = np.round(y_test * scale) / scale
        y_pred_rounded = np.round(y_pred * scale) / scale
        precision = np.mean(y_true_rounded == y_pred_rounded)
        precision_results[decimal_places] = precision
    
    return precision_results, y_pred

# Function to visualize prediction accuracy
def plot_prediction_results(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    
    # Plot actual vs predicted values
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted')
    
    # Plot prediction error
    plt.subplot(1, 2, 2)
    errors = np.abs(y_test - y_pred)
    plt.hist(errors, bins=50)
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.title('Prediction Error Distribution')
    
    plt.tight_layout()
    return plt

# Example usage with synthetic data
if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Generate synthetic data
    print("Generating synthetic data...")
    X, y = generate_synthetic_data(num_samples=10000, input_dim=500)
    
    # Split into train, validation, and test sets
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Validation data shape: {X_val.shape}, {y_val.shape}")
    print(f"Test data shape: {X_test.shape}, {y_test.shape}")
    
    # Train the model
    print("\nTraining model...")
    model, history = train_regression_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=64)
    
    # Evaluate the model
    print("\nEvaluating model...")
    test_loss, test_mae, test_precision = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    print(f"Test Decimal Precision (5 places): {test_precision:.4f}")
    
    # Detailed precision evaluation
    precision_results, y_pred = evaluate_decimal_precision(model, X_test, y_test)
    print("\nPrecision at different decimal places:")
    for places, precision in precision_results.items():
        print(f"{places} decimal places: {precision:.4f}")
    
    # Show some example predictions
    print("\nSample predictions (Actual vs. Predicted):")
    for i in range(10):
        print(f"Sample {i+1}: Actual = {y_test[i][0]:.6f}, Predicted = {y_pred[i][0]:.6f}, "
              f"Absolute Error = {abs(y_test[i][0] - y_pred[i][0]):.6f}")
    
    # Plot results
    plt = plot_prediction_results(y_test, y_pred)
    plt.suptitle("Neural Decision Tree Model Performance")
    plt.show()
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Mean Absolute Error')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['decimal_precision_metric'])
    plt.plot(history.history['val_decimal_precision_metric'])
    plt.title('5-Decimal Precision')
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.tight_layout()
    plt.show()
