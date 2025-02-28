import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Constants
g = 0.05
s = 0.04
bigy = (3 - s)**0.5

def GenerateLineshape(P, x):
    def cosal(x, eps):
        return (1 - eps * x - s) / bigxsquare(x, eps)

    def c(x):
        return ((g**2 + (1 - x - s)**2)**0.5)**0.5

    def bigxsquare(x, eps):
        return (g**2 + (1 - eps * x - s)**2)**0.5

    def mult_term(x, eps):
        return float(1) / (2 * np.pi * np.sqrt(bigxsquare(x, eps)))

    def cosaltwo(x, eps):
        return ((1 + cosal(x, eps)) / 2)**0.5

    def sinaltwo(x, eps):
        return ((1 - cosal(x, eps)) / 2)**0.5

    def termone(x, eps):
        return np.pi / 2 + np.arctan((bigy**2 - bigxsquare(x, eps)) / ((2 * bigy * (bigxsquare(x, eps))**0.5) * sinaltwo(x, eps)))

    def termtwo(x, eps):
        return np.log((bigy**2 + bigxsquare(x, eps) + 2 * bigy * (bigxsquare(x, eps)**0.5) * cosaltwo(x, eps)) / (bigy**2 + bigxsquare(x, eps) - 2 * bigy * (bigxsquare(x, eps)**0.5) * cosaltwo(x, eps)))

    def icurve(x, eps):
        return mult_term(x, eps) * (2 * cosaltwo(x, eps) * termone(x, eps) + sinaltwo(x, eps) * termtwo(x, eps))

    r = (np.sqrt(4 - 3 * P**(2)) + P) / (2 - 2 * P)
    Iplus = r * icurve(x, 1) / 10
    Iminus = icurve(x, -1) / 10
    signal = Iplus + Iminus
    return signal, Iplus, Iminus

n = 10000  
num_bins = 500  
data_min = -3  
data_max = 3 
polarization_values = np.linspace(0.001, .8, 100)  

x_values = np.linspace(data_min, data_max, n) 
bin_edges = np.linspace(data_min, data_max, num_bins + 1)  # Bin edges
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Compute bin centers for plotting

binned_errors = np.zeros((len(polarization_values), num_bins))

for idx, P in enumerate(polarization_values):
    signal, Iplus, Iminus = GenerateLineshape(P, x_values)
    
    # Bin the data
    bin_indices = np.digitize(x_values, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)  
    
    for i in range(num_bins):
        mask = (bin_indices == i)
        if np.any(mask):
            binned_errors[idx, i] = np.std(signal[mask])  

# Normalize the error
min_error = np.min(binned_errors)
max_error = np.max(binned_errors)
normalized_errors = (binned_errors - min_error) / (max_error - min_error)


# plt.style.use('dark_background')

# 2D Heatmap
plt.figure(figsize=(14, 8))
plt.imshow(normalized_errors, aspect='auto', extent=[data_min, data_max, 0, 1], origin='lower', cmap='inferno')
cbar = plt.colorbar(label='Normalized Error', pad=0.02)
cbar.set_label('Normalized Error', fontsize=14, rotation=270, labelpad=20)
plt.xlabel('Frequency Bin', fontsize=14)
plt.ylabel('Polarization', fontsize=14)
plt.title('Normalized Error as a Function of Polarization and Frequency Bins (2D Heatmap)', fontsize=16, pad=20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.savefig('Normalized_Error_Heatmap_Pretty.png', dpi=300, bbox_inches='tight')
plt.show()

# 3D Surface Plot
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(bin_centers, polarization_values)
surf = ax.plot_surface(X, Y, normalized_errors, cmap='plasma', linewidth=0, antialiased=True, alpha=0.9)
cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, pad=0.1)
cbar.set_label('Normalized Error', fontsize=14, rotation=270, labelpad=20)
ax.set_xlabel('Frequency Bin', fontsize=14, labelpad=10)
ax.set_ylabel('Polarization', fontsize=14, labelpad=10)
ax.set_zlabel('Normalized Error', fontsize=14, labelpad=10)
ax.set_title('Normalized Error as a Function of Polarization and Frequency Bins (3D Surface Plot)', fontsize=16, pad=20)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.grid(True, linestyle='--', alpha=0.3)
ax.view_init(elev=30, azim=45)  # Adjust viewing angle
plt.savefig('Normalized_Error_3D_Plot_Pretty.png', dpi=300, bbox_inches='tight')
plt.show()