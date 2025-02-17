import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from Lineshape import *

# Polarization
P = 0.5

# Generate the frequency axis (in MHz)
f_min = 212  # Minimum frequency in MHz
f_max = 214   # Maximum frequency in MHz
n_points = 500  # Number of points
frequencies_MHz = np.linspace(f_min, f_max, n_points)  # Frequency axis in MHz

# Generate the frequency-domain signal using your function
signal, Iplus, Iminus = GenerateLineshape(P, frequencies_MHz)

# Ensure Hermitian symmetry
def enforce_hermitian_symmetry(signal):
    n = len(signal)
    symmetric_signal = np.zeros_like(signal, dtype=complex)
    symmetric_signal[:n//2] = signal[:n//2]
    symmetric_signal[n//2:] = np.conj(signal[:n//2][::-1])
    return symmetric_signal

signal = enforce_hermitian_symmetry(signal)

# Perform the Inverse Fourier Transform (IFFT) using TensorFlow
signal_tf = tf.convert_to_tensor(signal, dtype=tf.complex64)
time_domain_signal_tf = tf.signal.ifft(signal_tf)

# Convert to numpy array and take the real part
time_domain_signal = np.real(time_domain_signal_tf.numpy())

# Generate the time axis
delta_f = frequencies_MHz[1] - frequencies_MHz[0]  # Frequency spacing in MHz
sampling_rate = 1 / delta_f  # Sampling rate in MHz
nyquist_limit = sampling_rate / 2  # Nyquist limit in MHz
print(f"Nyquist Limit: {nyquist_limit} MHz")
time_axis = np.fft.fftfreq(n_points, d=delta_f)  # Time axis in microseconds (μs)

# Plot the frequency-domain signal and the time-domain signal
plt.figure(figsize=(12, 6))

# Plot the frequency-domain signal
plt.subplot(2, 1, 1)
plt.plot(frequencies_MHz, np.abs(signal))  # Plot magnitude for clarity
plt.title('Frequency-Domain Signal (MHz)')
plt.xlabel('Frequency [MHz]')
plt.ylabel('Amplitude')

# Plot the time-domain signal
plt.subplot(2, 1, 2)
plt.plot(time_axis, time_domain_signal)
plt.title('Time-Domain Signal (IFFT Result)')
plt.xlabel('Time [μs]')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.savefig("IFFT.png")