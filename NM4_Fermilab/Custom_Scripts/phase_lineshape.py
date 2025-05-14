import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.font_manager as font_manager
from scipy.signal import hilbert

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
phi_values = np.linspace(0, 360, 360)  # 36 values from 0 to 360 degrees

for i, phi_deg in enumerate(phi_values):
    phi_rad = np.deg2rad(phi_deg)
    
    # Phase-sensitive linear combination
    signal1 = r * (yvals_absorp1 * np.sin(phi_rad) + yvals_disp1 * np.cos(phi_rad))
    signal2 = yvals_absorp2 * np.sin(phi_rad) + yvals_disp2 * np.cos(phi_rad)
    total_signal = signal1 + signal2
    
    # Use a colormap to create a gradient of colors
    color = plt.cm.plasma(i / len(phi_values))
    
    # Plot with decreasing opacity for better visualization
    alpha = 0.7 - (i / len(phi_values)) * 0.5  # Fade out as phi increases
    
    plt.plot(xvals, total_signal, color=color, alpha=alpha, linewidth=2)


norm = plt.Normalize(0, 360)
sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm)
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
plt.show()

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
        real_part[i, :] = (r *yvals_disp1 + yvals_disp2 ) * np.cos(phi_rad)
        imag_part[i, :] = (r * yvals_absorp1 + yvals_absorp2) * np.sin(phi_rad)
    
    # --- Real vs Imag, colored by phase ---
    # For each R (column), plot a line/points in (imag, real) space, colored by phase
    for j in range(len(xvals)):
        ax2.plot(imag_part[:, j], real_part[:, j], color='gray', alpha=0.2, linewidth=0.5)
    # Now scatter all points, colored by phase
    scatter = ax2.scatter(imag_part.flatten(), real_part.flatten(), c=np.repeat(phi_values, len(xvals)), 
                          cmap='plasma', s=8, alpha=0.8)
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
    plt.show()

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
    signal_heatmap[i, :] = signal1 + signal2

# Create heatmap
plt.figure(figsize=(16, 10))
im = plt.imshow(
    signal_heatmap,
    aspect='auto',
    extent=[xvals[0], xvals[-1], phi_values[0], phi_values[-1]],
    origin='lower',
    cmap='plasma'
)

# Add colorbar
cbar = plt.colorbar(im, orientation='horizontal', pad=0.1)
cbar.set_label('Signal Amplitude', fontsize=labelfontsize, fontweight='bold')
cbar.ax.tick_params(labelsize=labelfontsize-8)

# Add labels and title
plt.xlabel('R', fontsize=axisFontSize)
plt.ylabel('φ (degrees)', fontsize=axisFontSize)
plt.title('Phase-Sensitive Signal Variation', fontsize=28, pad=20)

# Format ticks
plt.xticks(fontsize=axisFontSize-8)
plt.yticks(fontsize=axisFontSize-8)

plt.tight_layout()
plt.savefig('signal_phase_heatmap.jpeg', dpi=300, bbox_inches='tight')
plt.show()

# Create arrays for real and imaginary components
real_signal = np.zeros((len(phi_values), len(xvals)))
imag_signal = np.zeros((len(phi_values), len(xvals)))

# Calculate real and imaginary components for each phase
for i, phi_deg in enumerate(phi_values):
    phi_rad = np.deg2rad(phi_deg)
    
    # Real component (cosine terms)
    real_signal[i, :] = r * yvals_disp1 * np.cos(phi_rad) + yvals_disp2 * np.cos(phi_rad)
    
    # Imaginary component (sine terms)
    imag_signal[i, :] = r * yvals_absorp1 * np.sin(phi_rad) + yvals_absorp2 * np.sin(phi_rad)

# Create 2D histogram (heatmap) of the phase space
plt.figure(figsize=(24, 16))
plt.hist2d(imag_signal.flatten(), real_signal.flatten(), 
          bins=125, cmap='seismic', density=True)

# Add colorbar
plt.colorbar(label='Density')

# Add labels and title
plt.xlabel('Imaginary Signal', fontsize=axisFontSize)
plt.ylabel('Real Signal', fontsize=axisFontSize)
plt.title('Phase Space Density Plot', fontsize=28, pad=20)

# Format ticks
plt.xticks(fontsize=axisFontSize-8)
plt.yticks(fontsize=axisFontSize-8)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('phase_space_density.jpeg', dpi=300, bbox_inches='tight')
plt.show()
