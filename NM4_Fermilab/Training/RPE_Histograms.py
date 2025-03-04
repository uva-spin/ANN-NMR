import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

try:
    import seaborn as sns
    sns.set_style("whitegrid")  # Set default style
except ImportError:
    print("Warning: Seaborn not found, continuing without it...")
    pass  # Continue without seaborn as it's not critical for the core functionality

def create_basic_error_plots(actual, predicted, save_dir='.', prefix=''):
    """
    Create basic error analysis plots (histograms and scatter plots).
    
    Parameters:
    -----------
    actual : array-like
        The actual/true values
    predicted : array-like
        The predicted values
    save_dir : str
        Directory to save the plots
    prefix : str
        Prefix for saved files
    
    Returns:
    --------
    dict
        Dictionary containing the calculated statistics
    """
    # Calculate residuals and relative percent error
    residuals = predicted - actual
    relative_percent_error = (abs((predicted - actual)) / actual) * 100

    print(f"Total data points: {len(residuals)}")

    # Create figure with 4 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

    # Plot 1: Histogram of Residuals with Gaussian fit
    mu_res, std_res = norm.fit(residuals)
    x_res = np.linspace(residuals.min(), residuals.max(), 100)
    p_res = norm.pdf(x_res, mu_res, std_res)

    ax1.hist(residuals, bins=100, density=True, alpha=0.7, color='skyblue')
    ax1.plot(x_res, p_res, 'r-', lw=2, label=f'Gaussian fit\nμ={mu_res:.2e}\nσ={std_res:.2e}')
    ax1.set_xlabel('Residuals (Predicted - Actual)')
    ax1.set_ylabel('Density')
    ax1.set_title('Histogram of Residuals')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Histogram of RPE with Gaussian fit
    mu_rpe, std_rpe = norm.fit(relative_percent_error)
    x_rpe = np.linspace(relative_percent_error.min(), relative_percent_error.max(), 100)
    p_rpe = norm.pdf(x_rpe, mu_rpe, std_rpe)

    ax2.hist(relative_percent_error, bins=100, density=True, alpha=0.7, color='skyblue')
    ax2.plot(x_rpe, p_rpe, 'r-', lw=2, label=f'Gaussian fit\nμ={mu_rpe:.2f}%\nσ={std_rpe:.2f}%')
    ax2.set_xlabel('Relative Percent Error')
    ax2.set_ylabel('Density')
    ax2.set_title('Histogram of Relative Percent Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Scatter plot of RPE vs Actual Polarization
    ax3.scatter(actual, relative_percent_error, alpha=0.5, s=1)
    ax3.set_xlabel('Actual Polarization')
    ax3.set_ylabel('Relative Percent Error (%)')
    ax3.set_title('RPE vs Actual Polarization')
    ax3.grid(True, alpha=0.3)

    # Save the first set of plots
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f'{prefix}error_analysis_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'residuals_mean': mu_res,
        'residuals_std': std_res,
        'rpe_mean': mu_rpe,
        'rpe_std': std_rpe
    }

def create_rpe_range_plots(actual, predicted, save_dir='.', prefix='', threshold=0.2):
    """
    Create RPE vs Polarization plots for different ranges.
    
    Parameters:
    -----------
    actual : array-like
        The actual/true values
    predicted : array-like
        The predicted values
    save_dir : str
        Directory to save the plots
    prefix : str
        Prefix for saved files
    threshold : float
        Threshold value for splitting the data (default: 0.2)
    """
    relative_percent_error = (abs((predicted - actual)) / actual) * 100

    # Create figure for RPE vs Polarization plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: RPE vs low polarization
    mask_low = (actual <= threshold)
    ax1.scatter(actual[mask_low], relative_percent_error[mask_low], alpha=0.5, s=1)
    ax1.set_xlabel('Actual Polarization')
    ax1.set_ylabel('Relative Percent Error (%)')
    ax1.set_title(f'RPE vs Actual Polarization (P ≤ {threshold})')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}%'))
    ax1.axvline(x=0.05, color='red', linestyle='--', alpha=0.5, label='P = 0.05%')
    ax1.legend()

    # Right plot: RPE vs higher polarization
    mask_high = (actual > threshold)
    ax2.scatter(actual[mask_high], relative_percent_error[mask_high], alpha=0.5, s=1)
    ax2.set_xlabel('Actual Polarization')
    ax2.set_ylabel('Relative Percent Error (%)')
    ax2.set_title(f'RPE vs Actual Polarization (P > {threshold})')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}%'))

    # Save the RPE vs Polarization plots
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f'{prefix}rpe_vs_polarization_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_model_errors(actual, predicted, save_dir='.', prefix='', threshold=0.2):
    """
    Comprehensive analysis of model errors, creating all plots and returning statistics.
    
    Parameters:
    -----------
    actual : array-like
        The actual/true values
    predicted : array-like
        The predicted values
    save_dir : str
        Directory to save the plots
    prefix : str
        Prefix for saved files
    threshold : float
        Threshold value for splitting the data in range plots
    
    Returns:
    --------
    dict
        Dictionary containing the calculated statistics
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate basic error plots and get statistics
    stats = create_basic_error_plots(actual, predicted, save_dir, prefix)
    
    # Generate RPE range plots
    create_rpe_range_plots(actual, predicted, save_dir, prefix, threshold)
    
    # Print statistics
    print(f"\nResiduals Statistics:")
    print(f"Mean: {stats['residuals_mean']:.2e}")
    print(f"Standard Deviation: {stats['residuals_std']:.2e}")
    print(f"\nRelative Percent Error Statistics:")
    print(f"Mean: {stats['rpe_mean']:.2f}%")
    print(f"Standard Deviation: {stats['rpe_std']:.2f}%")
    
    return stats

# # Load the data
# df = pd.read_csv('Model Performance/Deuteron_0_10_HighPrecision_V1/test_results_Deuteron_0_10_HighPrecision_V1.csv')

# # Calculate residuals and relative percent error
# residuals = df['Predicted'] - df['Actual']
# relative_percent_error = (abs((df['Predicted'] - df['Actual'])) / df['Actual']) * 100

# # Remove the outlier filtering function and filtered data creation
# print(f"Total data points: {len(residuals)}")

# # Create figure with 4 subplots
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

# # Plot 1: Histogram of Residuals with Gaussian fit
# mu_res, std_res = norm.fit(residuals)  # Using unfiltered residuals
# x_res = np.linspace(residuals.min(), residuals.max(), 100)
# p_res = norm.pdf(x_res, mu_res, std_res)

# ax1.hist(residuals, bins=100, density=True, alpha=0.7, color='skyblue')
# ax1.plot(x_res, p_res, 'r-', lw=2, label=f'Gaussian fit\nμ={mu_res:.2e}\nσ={std_res:.2e}')
# ax1.set_xlabel('Residuals (Predicted - Actual)')
# ax1.set_ylabel('Density')
# ax1.set_title('Histogram of Residuals')  # Removed "(Outliers Removed)"
# ax1.legend()
# ax1.grid(True, alpha=0.3)

# # Plot 2: Histogram of Relative Percent Error with Gaussian fit
# mu_rpe, std_rpe = norm.fit(relative_percent_error)  # Using unfiltered RPE
# x_rpe = np.linspace(relative_percent_error.min(), relative_percent_error.max(), 100)
# p_rpe = norm.pdf(x_rpe, mu_rpe, std_rpe)

# ax2.hist(relative_percent_error, bins=100, density=True, alpha=0.7, color='skyblue')
# ax2.plot(x_rpe, p_rpe, 'r-', lw=2, label=f'Gaussian fit\nμ={mu_rpe:.2f}%\nσ={std_rpe:.2f}%')
# ax2.set_xlabel('Relative Percent Error')
# ax2.set_ylabel('Density')
# ax2.set_title('Histogram of Relative Percent Error')  # Removed "(Outliers Removed)"
# ax2.legend()
# ax2.grid(True, alpha=0.3)

# # Plot 3: Scatter plot of RPE vs Actual Polarization (using unfiltered data)
# ax3.scatter(df['Actual'], relative_percent_error, alpha=0.5, s=1)
# ax3.set_xlabel('Actual Polarization')
# ax3.set_ylabel('Relative Percent Error (%)')
# ax3.set_title('RPE vs Actual Polarization')  # Removed "(Outliers Removed)"
# ax3.grid(True, alpha=0.3)

# # Adjust layout and save
# plt.tight_layout()
# plt.savefig('error_analysis_plots_filtered.png', dpi=300, bbox_inches='tight')
# plt.close()

# # Create new figure for RPE vs Polarization plots
# fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# # Left plot: RPE vs low polarization (P ≤ 0.05%)
# mask_low = (df['Actual'] <= 0.2)
# ax1.scatter(df['Actual'][mask_low], relative_percent_error[mask_low], alpha=0.5, s=1)
# ax1.set_xlabel('Actual Polarization')
# ax1.set_ylabel('Relative Percent Error (%)')
# ax1.set_title('RPE vs Actual Polarization (P ≤ 0.05%)')  # Removed "(Outliers Removed)"
# ax1.grid(True, alpha=0.3)
# ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}%'))
# ax1.axvline(x=0.05, color='red', linestyle='--', alpha=0.5, label='P = 0.05%')
# ax1.legend()

# # Right plot: RPE vs higher polarization (P > 0.05%)
# mask_high = (df['Actual'] > 0.2)
# ax2.scatter(df['Actual'][mask_high], relative_percent_error[mask_high], alpha=0.5, s=1)
# ax2.set_xlabel('Actual Polarization')
# ax2.set_ylabel('Relative Percent Error (%)')
# ax2.set_title('RPE vs Actual Polarization (P > 0.05%)')  # Removed "(Outliers Removed)"
# ax2.grid(True, alpha=0.3)
# ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}%'))

# # Save the RPE vs Polarization plots
# plt.tight_layout()
# plt.savefig('rpe_vs_polarization_plots.png', dpi=300, bbox_inches='tight')
# plt.close(fig2)

# # Print statistics
# print(f"\nResiduals Statistics:")
# print(f"Mean: {mu_res:.2e}")
# print(f"Standard Deviation: {std_res:.2e}")
# print(f"\nRelative Percent Error Statistics:")
# print(f"Mean: {mu_rpe:.2f}%")
# print(f"Standard Deviation: {std_rpe:.2f}%")
