import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import datetime

def perform_lineshape_fitting(output_dir=None, polarization_center=0.0005, polarization_range=0.0001, num_steps=10):
    """
    Perform lineshape fitting with parameter optimization and residual analysis.
    
    Parameters:
    -----------
    output_dir : str, optional
        Directory to save results files. Defaults to current directory if None.
    polarization_center : float, optional
        Center value for polarization parameter range. Default is 0.0005.
    polarization_range : float, optional
        Range around center value for polarization parameter. Default is 0.0001.
    num_steps : int, optional
        Number of polarization values to test. Default is 10.
    
    Returns:
    --------
    dict
        Dictionary containing results of the fitting process.
    """
    # Set default output directory to current directory if not provided
    if output_dir is None:
        output_dir = os.getcwd()
        
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from Custom_Scripts.Lineshape import GenerateLineshape
    
    
    # Define X domain
    X = np.linspace(-3, 3, 500)
    
    # Generate polarization values
    P_values = np.linspace(polarization_center - polarization_range, 
                           polarization_center + polarization_range, 
                           num_steps)
    
    # Generate true lineshapes
    Lineshapes_True = []
    for P in P_values:
        Line, _, _ = GenerateLineshape(P, X)
        Lineshapes_True.append(Line)
    
    # Set initial parameters and bounds
    initial_params = [0.00038, 0.00042, 0.00046, 0.000532, 0.0005443, 
                      0.000556, 0.000568, 0.00058, 0.00062, 0.00062]
    
    # Adjust if num_steps doesn't match length of initial_params
    if len(initial_params) != num_steps:
        initial_params = np.linspace(P_values[0] * 0.9, P_values[-1] * 1.1, num_steps)
    
    lower_bounds = [p - 0.00005 for p in initial_params]
    upper_bounds = [p + 0.00005 for p in initial_params]
    param_bounds = (lower_bounds, upper_bounds)
    
    # Define fitting function
    def Baseline(x, P):
        Sig, _, _ = GenerateLineshape(P, x)
        return Sig
    
    # Perform curve fitting
    covs = []
    popts = []
    residuals = []
    
    for i, (lower_bound, upper_bound) in enumerate(zip(lower_bounds, upper_bounds)):
        popt, pcov = curve_fit(Baseline, X, Lineshapes_True[i], 
                               p0=initial_params[i], 
                               bounds=(lower_bound, upper_bound))
        covs.append(pcov)
        popts.append(popt)
        
        # Calculate residuals
        P_true = P_values[i]
        P_opt = popt[0] if hasattr(popt, '__iter__') else popt
        residual = (P_true - P_opt) * 100  # Converting to percentage
        residuals.append(residual)
        
    # Calculate statistics for residuals
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)
    
    # Plot histogram of residuals with Gaussian fit
    plt.figure(figsize=(10, 6))
    
    # Create histogram
    n, bins, patches = plt.hist(residuals, bins=min(10, num_steps), 
                               color='skyblue', edgecolor='black', 
                               alpha=0.7, density=True)
    
    # Fit a Gaussian curve to the histogram
    def gaussian(x, mean, std, amplitude):
        return amplitude * np.exp(-((x - mean) ** 2) / (2 * std ** 2))
    
    # Create x values for the Gaussian curve
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    
    # Initial guess for the fit
    p0 = [residual_mean, residual_std, 1.0/(residual_std * np.sqrt(2 * np.pi))]
    
    try:
        # Try to fit a Gaussian to the histogram data
        params, covs_gauss = curve_fit(gaussian, bin_centers, n, p0=p0)
        fitted_mean, fitted_std, amplitude = params
        
        # Create smooth curve for plotting
        x_smooth = np.linspace(min(bins), max(bins), 100)
        y_smooth = gaussian(x_smooth, fitted_mean, fitted_std, amplitude)
        
        # Plot the fitted Gaussian curve
        plt.plot(x_smooth, y_smooth, 'r-', linewidth=2, 
                 label=f'Gaussian Fit\nMean = {fitted_mean:.5f}\nStd Dev = {fitted_std:.5f}')
        
        # Store the fitted parameters
        gauss_fit_params = {
            'mean': fitted_mean,
            'std_dev': fitted_std,
            'amplitude': amplitude
        }
    except:
        # If fitting fails, use the calculated statistics
        plt.axvline(residual_mean, color='r', linestyle='--', 
                   label=f'Mean = {residual_mean:.5f}')
        plt.axvline(residual_mean + residual_std, color='g', linestyle=':', 
                   label=f'±Std Dev = {residual_std:.5f}')
        plt.axvline(residual_mean - residual_std, color='g', linestyle=':')
        
        gauss_fit_params = {
            'mean': residual_mean,
            'std_dev': residual_std,
            'amplitude': None,
            'note': 'Gaussian fitting failed, using sample statistics'
        }
    
    plt.xlabel('Residual (P_true - P_optimized) × 100')
    plt.ylabel('Probability Density')
    plt.title('Histogram of Fitting Residuals with Gaussian Fit')
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Save histogram with Gaussian fit
    histogram_path = os.path.join(output_dir, "residuals_histogram_with_gaussian.png")
    plt.savefig(histogram_path)
    plt.close()
    
    # Save results to file
    results_path = os.path.join(output_dir, "Chi2_Results.txt")
    with open(results_path, 'w') as f:
        # Write header
        f.write("="*50 + "\n")
        f.write("OPTIMIZATION RESULTS\n")
        f.write("="*50 + "\n\n")
        
        # Write optimized parameters
        f.write("OPTIMIZED PARAMETERS:\n")
        f.write("-"*50 + "\n")
        for i, (P_true, param_value) in enumerate(zip(P_values, popts)):
            # Handle both single value and array cases
            if hasattr(param_value, '__iter__') and not isinstance(param_value, str):
                formatted_value = f"{param_value[0]:.8f}"
            else:
                formatted_value = f"{param_value:.8f}"
            
            f.write(f"Parameter {i+1}: {P_true} = {formatted_value}\n")
        f.write("-"*50 + "\n\n")
        
        # Write residuals
        f.write("RESIDUALS (P_true - P_optimized):\n")
        f.write("-"*50 + "\n")
        for i, (P_true, opt_val) in enumerate(zip(P_values, popts)):
            # Handle both single value and array cases for optimized values
            if hasattr(opt_val, '__iter__') and not isinstance(opt_val, str):
                opt_val = opt_val[0]
            
            # Calculate residual
            residual = P_true - opt_val
            percent_error = (residual / P_true) * 100 if P_true != 0 else float('inf')
            
            f.write(f"Parameter {i+1}: {P_true}\n")
            f.write(f"    True value:      {P_true:.8f}\n")
            f.write(f"    Optimized value: {opt_val:.8f}\n")
            f.write(f"    Residual:        {residual:.8f}\n")
            f.write(f"    Percent error:   {percent_error:.4f}%\n\n")
        f.write("-"*50 + "\n\n")
        
        # Write covariance matrices
        f.write("COVARIANCE MATRICES:\n")
        f.write("-"*50 + "\n")
        for i, cov in enumerate(covs):
            f.write(f"\nMatrix {i+1} (Parameter: {P_values[i]}):\n")
            # Check the type of cov and handle accordingly
            if np.isscalar(cov):
                # If it's a scalar value
                f.write(f"    {cov:.8e}\n")
            elif isinstance(cov, (list, np.ndarray)):
                # If it's an array or matrix
                if hasattr(cov, 'shape') and len(cov.shape) == 2:
                    # For 2D matrices
                    for row in cov:
                        formatted_row = "    " + " ".join([f"{val:.4e}" for val in row])
                        f.write(formatted_row + "\n")
                else:
                    # For 1D arrays or other structures
                    f.write(f"    {cov}\n")
            else:
                # Fallback for any other type
                f.write(f"    {cov}\n")
        f.write("-"*50 + "\n")
        
        f.write("\n\nFile generated on: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    print(f"Results saved to '{results_path}'")
    print(f"Histogram saved to '{histogram_path}'")
    
    # Return results
    return {
        "P_values": P_values,
        "optimized_params": popts,
        "covariance_matrices": covs,
        "residuals": residuals,
        "results_file": results_path,
        "histogram_file": histogram_path
    }

if __name__ == "__main__":
    # Set the output directory to be the same as the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    perform_lineshape_fitting(output_dir=script_dir)


# plt.plot(x_full_bins, y)
# plt.show()


# initial_params = [0.00052]


# param_bounds = (0.0002, 0.0007) 

# def Baseline(x, P):
#     Sig, _, _ = GenerateLineshape(P, x)
#     return Sig


# popt, pcov = curve_fit(Baseline, x_full_bins, y, p0=initial_params, bounds=param_bounds)

# print("Covariance Matrix:")
# print(pcov)

# print("Optimized Parameters:")
# print(popt)

# Fitted_Sig = Baseline(x_full_bins, *popt)

# plt.errorbar(x_full_bins, y, yerr=yerr_full, fmt='o', markersize=1, label='Data', color='black')  # Plot full data range
# plt.plot(x_full_bins, Fitted_Sig, label='Fit', color='red', linewidth=2)  # Fit for selected domain
# plt.xlabel('Frequency (MHz)')
# plt.ylabel('Dependent Variable')
# plt.legend()
# plt.title('Fit of Data between Frequencies')
# plt.show()
