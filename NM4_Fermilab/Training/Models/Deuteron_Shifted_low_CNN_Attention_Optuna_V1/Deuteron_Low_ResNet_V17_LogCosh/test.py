import tensorflow as tf
import keras
from keras import ops
from keras import backend as K
from keras.layers import Activation
import json
import os

# Register the swish activation function
@keras.saving.register_keras_serializable()
def swish(x):
    return x * K.sigmoid(x)

@keras.saving.register_keras_serializable()
def log_cosh_precision_loss(y_true, y_pred):
    error = y_true - y_pred
    precision_weights = tf.math.exp(-10.0 * y_true) + 1e-6  
    return tf.reduce_mean(precision_weights * tf.math.log(cosh(error)))


# Load the model
model = keras.models.load_model('Models/Deuteron_Low_ResNet_V17_LogCosh/best_model.keras', 
                                 custom_objects={'swish': swish, 'log_cosh_precision_loss': log_cosh_precision_loss})

model.summary()
# print(model.get_config())

# Get model configuration
config = model.get_config()

# Prepare layer configurations
layer_configs = {}
for i, layer in enumerate(model.layers):
    layer_config = layer.get_config()
    layer_configs[layer.name] = layer_config

# Combine model config and layer configs into a single dictionary
output = {
    'model_config': config,
    'layer_configs': layer_configs
}

# Define the output file path
output_file_path = 'Models/Deuteron_Low_ResNet_V17_LogCosh/model_summary.json'

# Write the combined output to a JSON file
with open(output_file_path, 'w') as json_file:
    json.dump(output, json_file, indent=4)

print(f"Model configuration and layer configurations saved to {output_file_path}")

