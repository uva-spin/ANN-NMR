import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from scipy.stats import norm
    
def plot_rpe_and_residuals_over_range(y_true, y_pred, performance_dir, version):
    # Create bins for different polarization ranges
    bins = [
        (0.001, 0.01),  # 0.1-1%
        (0.01, 0.1),    # 1-10%
        (0.1, 0.8)      # 10-80%
    ]

    valid_bins = 0  # Count of valid bins with data
    plt.figure(figsize=(18, 12))
    plt.suptitle(f'Range-Specific Metrics - {version}', y=1.02, fontsize=20, weight='bold')

    for i, (lower, upper) in enumerate(bins):
        # Filter data for current range
        mask = (y_true >= lower) & (y_true < upper)
        y_true_range = y_true[mask]
        y_pred_range = y_pred[mask]
        
        if len(y_true_range) == 0:
            continue
        
        valid_bins += 1  # Increment valid bin count

        # Calculate metrics
        residuals = y_true_range - y_pred_range
        rpe = np.abs((residuals / y_true_range)) * 100  # Relative Percent Error
        mae = np.mean(np.abs(residuals))
        mse = np.mean(residuals**2)

        # Create subplot for histogram of residuals using Seaborn
        ax = plt.subplot(2, 3, valid_bins)  # Use valid_bins for positioning
        sns.histplot(residuals, bins=30, kde=True, stat="density", color='blue', edgecolor='black', ax=ax, alpha=0.6)
        
        # Fit a Gaussian to the residuals data
        mu_res, sigma_res = norm.fit(residuals)
        x = np.linspace(min(residuals), max(residuals), 100)
        y_res = norm.pdf(x, mu_res, sigma_res)
        ax.plot(x, y_res, '--', color='red', linewidth=2, label=f'Gaussian Fit: μ={mu_res:.6f}, σ={sigma_res:.6f}')

        # Set titles and labels for residuals histogram
        ax.set_title(f'Residuals Histogram: {lower*100:.6f}% to {upper*100:.6f}%', fontsize=16)
        ax.set_xlabel('Residuals', fontsize=14)
        ax.set_ylabel('Density', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)

        # Create subplot for RPE vs. Polarization using Seaborn
        ax_rpe = plt.subplot(2, 3, valid_bins + 3)  # Positioning in the second row
        sns.scatterplot(x=y_true_range, y=rpe, marker='o', color='purple', ax=ax_rpe)
        ax_rpe.set_title(f'RPE vs. Polarization: {lower*100:.4f}% to {upper*100:.4f}%', fontsize=16)
        ax_rpe.set_xlabel('Polarization Values', fontsize=14)
        ax_rpe.set_ylabel('Relative Percent Error (%)', fontsize=14)
        ax_rpe.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout based on the number of valid bins
    if valid_bins > 0:
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the title
    else:
        plt.close()  # Close the plot if no valid bins

    range_metrics_path = os.path.join(performance_dir, f'{version}_Combined_Metrics.png')
    plt.savefig(range_metrics_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Combined metrics plot saved to {range_metrics_path}")
    

    
def plot_training_history(history, performance_dir, version):
    """
    Plots training and validation loss and accuracy from the training history.

    Parameters:
    - history: A History object returned by the fit method of a Keras model.
    """
    plt.figure(figsize=(14, 6))

    # Plot training and validation loss using Seaborn
    plt.subplot(1, 2, 1)
    sns.lineplot(data=history.history['loss'], label='Training Loss', color='blue', marker='o')
    sns.lineplot(data=history.history['val_loss'], label='Validation Loss', color='orange', marker='o')
    plt.title('Training and Validation Loss', fontsize=18)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plot training and validation accuracy if available using Seaborn
    if 'accuracy' in history.history and 'val_accuracy' in history.history:
        plt.subplot(1, 2, 2)
        sns.lineplot(data=history.history['accuracy'], label='Training Accuracy', color='green', marker='o')
        sns.lineplot(data=history.history['val_accuracy'], label='Validation Accuracy', color='red', marker='o')
        plt.title('Training and Validation Accuracy', fontsize=18)
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(performance_dir, f'{version}_Training_History.png'), dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {os.path.join(performance_dir, f'{version}_Training_History.png')}")

def plot_rpe_and_residuals(y_true, y_pred, performance_dir, version):
    """
    Plots the Relative Percent Error (RPE) and residuals for given true and predicted values.

    Parameters:
    - y_true: Array of true values.
    - y_pred: Array of predicted values.
    - performance_dir: Directory to save the plot.
    - version: Version identifier for the plot filename.
    """
    # Calculate metrics
    residuals = (y_true - y_pred) * 100
    rpe = np.abs((residuals / y_true))  # Relative Percent Error

    plt.figure(figsize=(18, 6))

    # Plot residuals
    plt.subplot(1, 3, 1)
    sns.histplot(residuals, bins=30, kde=True, stat="density", color='blue', edgecolor='black', alpha=0.6)
    mu_res, sigma_res = norm.fit(residuals)
    x_res = np.linspace(min(residuals), max(residuals), 100)
    y_res = norm.pdf(x_res, mu_res, sigma_res)
    plt.plot(x_res, y_res, '--', color='red', linewidth=2, label=f'Gaussian Fit: μ={mu_res:.6f}, σ={sigma_res:.6f}')
    plt.title('Residuals Histogram', fontsize=16)
    plt.xlabel('Residuals', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plot RPE
    plt.subplot(1, 3, 2)
    sns.scatterplot(x=y_true, y=rpe, marker='o', color='purple')
    plt.title('Relative Percent Error vs. True Values', fontsize=16)
    plt.xlabel('True Values', fontsize=14)
    plt.ylabel('Relative Percent Error (%)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plot RPE Histogram
    plt.subplot(1, 3, 3)
    sns.histplot(rpe, bins=30, kde=True, stat="density", color='green', edgecolor='black', alpha=0.6)
    mu_rpe, sigma_rpe = norm.fit(rpe)
    x_rpe = np.linspace(min(rpe), max(rpe), 100)
    y_rpe = norm.pdf(x_rpe, mu_rpe, sigma_rpe)
    plt.plot(x_rpe, y_rpe, '--', color='orange', linewidth=2, label=f'Gaussian Fit: μ={mu_rpe:.6f}, σ={sigma_rpe:.6f}')
    plt.title('RPE Histogram', fontsize=16)
    plt.xlabel('Relative Percent Error (%)', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    range_metrics_path = os.path.join(performance_dir, f'{version}_RPE_and_Residuals.png')
    plt.savefig(range_metrics_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"RPE and Residuals plot saved to {range_metrics_path}")

    