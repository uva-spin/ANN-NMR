import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

# Load real data
real_data = pd.read_csv('Noise_Data/Noise_RawSignal.csv')

# Drop the first column
real_data = real_data.iloc[:, 1:]

def normalize_data(data):
    real_data = data
    real_data_min = data.min()
    real_data_max = data.max()
    data = (real_data - real_data_min) / (real_data_max - real_data_min)
    return data
# Normalize real data
real_data_normalized = normalize_data(real_data)

# Create a quadratic activation function
def cubic(w):
  return -4.33836842e-04*w**3 -6.36998936e-03*w**2 + 1.74780124e+00*w -3.52343591e+01


# Define the VAE model
def build_vae(input_shape, latent_dim):
    encoder_inputs = tf.keras.Input(shape=input_shape)
    x = layers.Dense(32, activation='relu')(encoder_inputs)
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)
    
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.keras.backend.random_normal(shape=(tf.keras.backend.shape(z_mean)[0], latent_dim))
        return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon
    
    z = layers.Lambda(sampling)([z_mean, z_log_var])
    
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
    
    decoder_inputs = tf.keras.Input(shape=(latent_dim,))
    x = layers.Dense(32, activation='relu')(decoder_inputs)
    outputs = layers.Dense(input_shape[0], activation=cubic)(x)  # Use linear activation function
    
    decoder = Model(decoder_inputs, outputs, name='decoder')
    
    outputs = decoder(encoder(encoder_inputs)[2])
    vae = Model(encoder_inputs, outputs, name='vae')
    
    reconstruction_loss = tf.reduce_mean(tf.square(encoder(encoder_inputs)[0] - decoder(encoder(encoder_inputs)[2])))
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    vae_loss = reconstruction_loss + kl_loss
    vae.add_loss(vae_loss)
    
    return vae

# Set hyperparameters
latent_dim = 500  # Adjust this based on your requirements
num_samples = 100

# Build the VAE model
input_shape = (real_data_normalized.shape[1],)
vae = build_vae(input_shape, latent_dim)

# Compile the model
vae.compile(optimizer='adam')

# Train the VAE model
vae.fit(real_data_normalized, epochs=5, batch_size=256)

# Generate synthetic data
synthetic_data_normalized = vae.predict(real_data_normalized.sample(num_samples))

synthetic_data_normalized = synthetic_data_normalized.reshape((-1, real_data_normalized.shape[1]))

# Denormalize synthetic data to be between the smallest value and 0
min_val = real_data.min().min()
synthetic_data = -abs(synthetic_data_normalized)

# Convert synthetic data back to a DataFrame
synthetic_df = pd.DataFrame(synthetic_data, columns=real_data_normalized.columns)

# Save synthetic data to CSV file
synthetic_df.to_csv('Noise_Data/Synthetic_Data.csv', index=False, header=False)

