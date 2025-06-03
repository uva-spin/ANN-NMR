import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.font_manager as font_manager
from scipy.signal import hilbert
from Lineshape import *

# --- System and lineshape setup ---
g = 0.05
s = 0.04
bigy = np.sqrt(3 - s)
labelfontsize = 30
P = 0.50  # Input polarization
r = (np.sqrt(4 - 3 * P**2) + P) / (2 - 2 * P)

# --- Phase for detection (degrees) ---
phi_deg = 0  # <-- Change this value for different phase settings
phi_rad = np.deg2rad(phi_deg)

def cosal(x, eps):
    return (1 - eps * x - s) / bigxsquare(x, eps)

def c(x):
    return np.sqrt(np.sqrt(g**2 + (1 - x - s)**2))

def bigxsquare(x, eps):
    return np.sqrt(g**2 + (1 - eps * x - s)**2)

def mult_term(x, eps):
    return 1 / (2 * np.pi * np.sqrt(bigxsquare(x, eps)))

def cosaltwo(x, eps):
    return np.sqrt((1 + cosal(x, eps)) / 2)

def sinaltwo(x, eps):
    return np.sqrt((1 - cosal(x, eps)) / 2)

def termone(x, eps):
    return np.pi / 2 + np.arctan((bigy**2 - bigxsquare(x, eps)) / (2 * bigy * np.sqrt(bigxsquare(x, eps)) * sinaltwo(x, eps)))

def termtwo(x, eps):
    return np.log((bigy**2 + bigxsquare(x, eps) + 2 * bigy * np.sqrt(bigxsquare(x, eps)) * cosaltwo(x, eps)) /
                  (bigy**2 + bigxsquare(x, eps) - 2 * bigy * np.sqrt(bigxsquare(x, eps)) * cosaltwo(x, eps)))

def icurve(x, eps):
    return mult_term(x, eps) * (2 * cosaltwo(x, eps) * termone(x, eps) + sinaltwo(x, eps) * termtwo(x, eps))

# --- Generate x and absorptive signal ---
xvals = np.linspace(-3, 3, 500)
yvals_absorp1 = icurve(xvals, 1) / 10        # χ''₊
yvals_absorp2 = icurve(-xvals, 1) / 10       # χ''₋

# --- Get dispersive part via Hilbert transform (numerical Kramers–Kronig) ---
yvals_disp1 = np.imag(hilbert(yvals_absorp1))  # χ'₊
yvals_disp2 = np.imag(hilbert(yvals_absorp2))  # χ'₋

# --- Plotting setup ---
fig = plt.figure(figsize=(16, 16))
ax = fig.add_subplot(1, 1, 1)

# --- Generate and plot signals for different phase values ---
phi_values = np.linspace(0, 360, 500)  # 36 values from 0 to 360 degrees
P_values = np.linspace(0, 1, 500)

U = 2.4283
eta = 1.04e-2
# self.phi = 6.1319
Cstray = 10**(-20)
shift = 0
Cknob = 0.220
cable = 6/2
center_freq = 32.32

noise = np.random.normal(0, 0.1, len(xvals))

for i, phi_deg, P in enumerate(phi_values, P_values):
    phi_rad = np.deg2rad(phi_deg)
    
    # Phase-sensitive linear combination
    signal1 = r * (yvals_absorp1 * np.sin(phi_rad) + yvals_disp1 * np.cos(phi_rad))
    signal2 = yvals_absorp2 * np.sin(phi_rad) + yvals_disp2 * np.cos(phi_rad)

    baseline = Baseline(xvals, U, Cknob, eta, shift, Cstray, phi_deg, 0)

    total_signal = signal1 + signal2 + baseline + noise
    
    # Use a colormap to create a gradient of colors
    color = plt.cm.plasma(i / len(phi_values))
    
    # Plot with decreasing opacity for better visualization
    alpha = 0.7 - (i / len(phi_values)) * 0.5  # Fade out as phi increases
    
    plt.plot(xvals, total_signal, color=color, alpha=alpha, linewidth=2)
    # plt.plot(xvals, signal1, color='red', alpha=alpha, linewidth=2)
    # plt.plot(xvals, signal2, color='green', alpha=alpha, linewidth=2)


norm = plt.Normalize(0, 360)
sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.1)
cbar.set_label('φ (degrees)', fontsize=labelfontsize, fontweight='bold')
cbar.ax.tick_params(labelsize=labelfontsize-8)

# --- Plot formatting ---
axisFontSize = 38
legendFontSize = 38

plt.xlabel('R', fontsize=axisFontSize)
plt.ylabel('Signal [$C_E$ mV]', fontsize=axisFontSize)

# Set up major and minor ticks
major_ticks_x = np.arange(-4, 4, 0.5)  # Major ticks every 1 unit
minor_ticks_x = np.arange(-4, 4, 0.5)  # Minor ticks every 0.5 units
major_ticks_y = np.arange(-1.2, 1.2, 0.2)  # Major ticks every 0.2 units
minor_ticks_y = np.arange(-1.2, 1.2, 0.1)  # Minor ticks every 0.1 units



# Set the ticks
ax.set_xticks(major_ticks_x)
ax.set_xticks(minor_ticks_x, minor=True)
ax.set_yticks(major_ticks_y)
ax.set_yticks(minor_ticks_y, minor=True)

# Set the grid
ax.grid(True, which='major', linestyle='-', linewidth=1.5, alpha=0.7)
ax.grid(True, which='minor', linestyle='--', linewidth=1, alpha=0.4)

ax.set_facecolor('black')

# Format tick labels
plt.xticks(fontsize=axisFontSize)
plt.yticks(fontsize=axisFontSize)

# Format tick labels - show only every other major tick
for label in ax.xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
for label in ax.yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)

ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.set_axisbelow(True)

plt.xlim(-3.5, 3.5)

# Add title
plt.title('Phase-Sensitive Signal Variation (φ = 0° to 360°)', fontsize=28, pad=20)


# Save figure
fig.set_size_inches(24, 16)
fig.savefig('signal_phase_variation.jpeg', dpi=600, bbox_inches='tight')
# plt.show()

def create_signal_heatmap(signal1, signal2, xvals, phi_values):
    """
    Create a heatmap visualization of real vs imaginary signal variations.
    
    Args:
        signal1: Array of signal1 values
        signal2: Array of signal2 values
        xvals: x-axis values
        phi_values: Array of phase values
    """
    # Create figure for real vs imaginary plot only
    fig, ax2 = plt.subplots(1, 1, figsize=(10, 10))
    
    # Calculate signal values for each combination
    Z1 = np.zeros((len(phi_values), len(xvals)))
    real_part = np.zeros_like(Z1)
    imag_part = np.zeros_like(Z1)
    for i, phi_deg in enumerate(phi_values):
        phi_rad = np.deg2rad(phi_deg)
        baseline = Baseline(xvals, U, Cknob, eta, shift, Cstray, phi_deg, 0)
        real_part[i, :] = (r *yvals_disp1 + yvals_disp2 ) * np.cos(phi_rad) + baseline + noise
        imag_part[i, :] = (r * yvals_absorp1 + yvals_absorp2) * np.sin(phi_rad) + baseline + noise
    
    # --- Real vs Imag, colored by phase ---
    # For each R (column), plot a line/points in (imag, real) space, colored by phase
    for j in range(len(xvals)):
        ax2.plot(imag_part[:, j], real_part[:, j], color='gray', alpha=0.2, linewidth=0.5)
    # Now scatter all points, colored by phase
    scatter = ax2.scatter(imag_part.flatten(), real_part.flatten(), c=np.repeat(phi_values, len(xvals)), 
                          cmap='hsv', s=8, alpha=0.8)
    cbar2 = plt.colorbar(scatter, ax=ax2, orientation='horizontal', pad=0.1)
    cbar2.set_label('φ (degrees)', fontsize=labelfontsize, fontweight='bold')
    cbar2.ax.tick_params(labelsize=labelfontsize-8)
    ax2.set_title('Real vs Imaginary Signal', fontsize=24, pad=20)
    ax2.set_xlabel('Imaginary Signal', fontsize=32)
    ax2.set_ylabel('Real Signal', fontsize=32)
    ax2.tick_params(axis='both', labelsize=24)
    ax2.set_xlim(imag_part.min(), imag_part.max())
    ax2.set_ylim(real_part.min(), real_part.max())
    ax2.grid(True, which='major', linestyle='-', linewidth=1.5, alpha=0.7)
    ax2.grid(True, which='minor', linestyle='--', linewidth=1, alpha=0.4)
    
    ax2.set_facecolor('black')


    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.set_size_inches(24, 16)
    fig.savefig('signal_heatmap_real_imag_only.jpeg', dpi=600, bbox_inches='tight')
    # plt.show()

# Add this after your main plot code:
create_signal_heatmap(yvals_absorp1, yvals_disp1, xvals, phi_values)

# Create array to store total signal for heatmap
signal_heatmap = np.zeros((len(phi_values), len(xvals)))

# Fill the array with total signal values for each phase and R value
for i, phi_deg in enumerate(phi_values):
    phi_rad = np.deg2rad(phi_deg)
    
    # Phase-sensitive linear combination
    signal1 = r * (yvals_absorp1 * np.sin(phi_rad) + yvals_disp1 * np.cos(phi_rad))
    signal2 = yvals_absorp2 * np.sin(phi_rad) + yvals_disp2 * np.cos(phi_rad)
    baseline = Baseline(xvals, U, Cknob, eta, shift, Cstray, phi_deg, 0)

    signal_heatmap[i, :] = signal1 + signal2 + baseline + noise

# Create heatmap
plt.figure(figsize=(16, 10))
im = plt.imshow(
    signal_heatmap,
    aspect='auto',
    extent=[xvals[0], xvals[-1], phi_values[0], phi_values[-1]],
    origin='lower',
    cmap='hsv'
)

# Add colorbar
cbar = plt.colorbar(im, orientation='horizontal', pad=0.15)
cbar.set_label('Signal Amplitude', fontsize=labelfontsize, fontweight='bold')
cbar.ax.tick_params(labelsize=labelfontsize-8)

# Add labels and title
plt.xlabel('R', fontsize=24)
plt.ylabel('φ (degrees)', fontsize=24)
plt.title('Phase-Sensitive Signal Variation', fontsize=28, pad=20)

# Format ticks
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.tight_layout()
plt.savefig('signal_phase_heatmap.jpeg', dpi=600, bbox_inches='tight')



real_signal = np.zeros((len(phi_values), len(xvals)))
imag_signal = np.zeros((len(phi_values), len(xvals)))

for i, phi_deg in enumerate(phi_values):
    phi_rad = np.deg2rad(phi_deg)
    baseline = Baseline(xvals, U, Cknob, eta, shift, Cstray, phi_deg, 0)
    # Real component (cosine terms)
    real_signal[i, :] = r * yvals_disp1 * np.cos(phi_rad) + yvals_disp2 * np.cos(phi_rad) + baseline + noise
    # Imaginary component (sine terms)
    imag_signal[i, :] = r * yvals_absorp1 * np.sin(phi_rad) + yvals_absorp2 * np.sin(phi_rad) + baseline + noise




# Create heatmap
fig = plt.figure(figsize=(24, 10))

# Create a GridSpec to handle the subplots and colorbar
gs = plt.GridSpec(1, 2, width_ratios=[1, 1], height_ratios=[1], wspace=0.3)

# Real component heatmap
ax1 = plt.subplot(gs[0])
im1 = ax1.imshow(
    real_signal,
    aspect='auto',
    extent=[xvals[0], xvals[-1], phi_values[0], phi_values[-1]],
    origin='lower',
    cmap='hsv'
)
ax1.set_xlabel('R', fontsize=24)
ax1.set_ylabel('φ (degrees)', fontsize=24)
ax1.set_title('Real Component', fontsize=28, pad=20)
ax1.tick_params(axis='both', labelsize=16)

# Imaginary component heatmap
ax2 = plt.subplot(gs[1])
im2 = ax2.imshow(
    imag_signal,
    aspect='auto',
    extent=[xvals[0], xvals[-1], phi_values[0], phi_values[-1]],
    origin='lower',
    cmap='hsv'
)
ax2.set_xlabel('R', fontsize=24)
ax2.set_ylabel('φ (degrees)', fontsize=24)
ax2.set_title('Imaginary Component', fontsize=28, pad=20)
ax2.tick_params(axis='both', labelsize=16)

# Add a shared colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(im1, cax=cbar_ax)
cbar.set_label('Signal Amplitude', fontsize=labelfontsize, fontweight='bold')
cbar.ax.tick_params(labelsize=labelfontsize-8)

plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make room for colorbar
plt.savefig('signal_phase_heatmap_real_imag.jpeg', dpi=600, bbox_inches='tight')
# plt.show()

# Create side-by-side 2D histograms for real and imaginary components
plt.figure(figsize=(24, 10))

# Real component histogram
plt.subplot(1, 2, 1)
plt.hist2d(imag_signal.flatten(), real_signal.flatten(), 
          bins=125, cmap='hsv', density=True)
plt.colorbar(label='Density')
plt.xlabel('Imaginary Signal', fontsize=axisFontSize)
plt.ylabel('Real Signal', fontsize=axisFontSize)
plt.title('Real Component Phase Space', fontsize=28, pad=20)
plt.xticks(fontsize=axisFontSize-8)
plt.yticks(fontsize=axisFontSize-8)
plt.grid(True, alpha=0.3)

# Imaginary component histogram
plt.subplot(1, 2, 2)
plt.hist2d(imag_signal.flatten(), real_signal.flatten(), 
          bins=125, cmap='hsv', density=True)
plt.colorbar(label='Density')
plt.xlabel('Imaginary Signal', fontsize=axisFontSize)
plt.ylabel('Real Signal', fontsize=axisFontSize)
plt.title('Imaginary Component Phase Space', fontsize=28, pad=20)
plt.xticks(fontsize=axisFontSize-8)
plt.yticks(fontsize=axisFontSize-8)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('phase_space_components_side_by_side.jpeg', dpi=300, bbox_inches='tight')
# plt.show()

def calculate_total_signal(x, P, phi_deg):
    """
    Calculate the total signal for given x, polarization P, and phase angle phi.
    
    Parameters:
    -----------
    x : float or array-like
        The x-coordinate value(s)
    P : float
        Input polarization (between 0 and 1)
    phi_deg : float
        Phase angle in degrees
        
    Returns:
    --------
    float or array-like
        The total signal value(s)
    """
    # System parameters
    g = 0.05
    s = 0.04
    bigy = np.sqrt(3 - s)
    
    # Calculate r from P
    r = (np.sqrt(4 - 3 * P**2) + P) / (2 - 2 * P)
    
    # Convert phase to radians
    phi_rad = np.deg2rad(phi_deg)
    
    # Calculate absorptive signals
    yvals_absorp1 = icurve(x, 1) / 10        # χ''₊
    yvals_absorp2 = icurve(-x, 1) / 10       # χ''₋
    
    # Calculate dispersive signals using Hilbert transform
    yvals_disp1 = np.imag(hilbert(yvals_absorp1))  # χ'₊
    yvals_disp2 = np.imag(hilbert(yvals_absorp2))  # χ'₋
    
    # Calculate phase-sensitive linear combination
    signal1 = r * (yvals_absorp1 * np.sin(phi_rad) + yvals_disp1 * np.cos(phi_rad))
    signal2 = yvals_absorp2 * np.sin(phi_rad) + yvals_disp2 * np.cos(phi_rad)
    
    # Return total signal
    return signal1 + signal2
