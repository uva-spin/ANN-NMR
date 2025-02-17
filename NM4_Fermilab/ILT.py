import numpy as np
import matplotlib.pyplot as plt
import mpmath
from Lineshape import *

# Polarization
P = 0.5

# Generate the frequency axis (in MHz)
f_min = -3  # Minimum frequency in MHz
f_max = 3  # Maximum frequency in MHz
n_points = 500  # Number of points
frequencies_MHz = np.linspace(f_min, f_max, n_points)  # Frequency axis in MHz

# Generate the frequency-domain signal using your function
signal, Iplus, Iminus = GenerateLineshape(P, frequencies_MHz)

# Define the Laplace transform function numerically (Laplace of f(t) = integral e^(-st) * f(t) dt)
def laplace_transform(f_t, s):
    """
    Numerically computes the Laplace transform of a function f(t) at a given s using integration.
    """
    # Set the integral bounds: from 0 to a large value (e.g., 10^6) for numerical stability
    return mpmath.quad(lambda t: mpmath.exp(-s * t) * f_t(t), [0, 1e6])

# Define the time-domain function for your signal
# Assuming signal as a function of time t (replace this with the actual function)
def f_t(t):
    # For example, if `signal` corresponds to an exponential decay or similar function, define it here
    # Example: a simple exponential decay (modify as needed)
    return mpmath.exp(-t)

# Perform the Laplace Transform on the signal using the numerical integration method
mpmath.mp.dps = 15  # Set decimal places for precision

laplace_transformed_signal = []
for s in frequencies_MHz:
    # Compute the Laplace transform at each frequency point
    laplace_transformed_signal.append(laplace_transform(f_t, s))

# Convert the results into a numpy array for plotting
laplace_transformed_signal = np.array(laplace_transformed_signal, dtype=complex)

# Plot the frequency-domain signal and the Laplace-transformed signal
plt.figure(figsize=(12, 6))

# Plot the original frequency-domain signal
plt.subplot(2, 1, 1)
plt.plot(frequencies_MHz, np.abs(signal))  # Plot magnitude for clarity
plt.title('Frequency-Domain Signal (MHz)')
plt.xlabel('Frequency [MHz]')
plt.ylabel('Amplitude')

# Plot the Laplace-transformed signal
plt.subplot(2, 1, 2)
plt.plot(frequencies_MHz, np.abs(laplace_transformed_signal))
plt.title('Laplace Transformed Signal')
plt.xlabel('Frequency [MHz]')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.savefig("Laplace_Transform.png")