import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_metrics(history1, history2, performance_dir, version):
    # Create a 2x2 grid of subplots
    plt.figure(figsize=(20, 16))
    plt.suptitle(f'Training Metrics - {version}', y=1.02, fontsize=16)
    
    # Phase 1 Metrics
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(history1.history['loss'], label='Training Loss')
    ax1.plot(history1.history['val_loss'], label='Validation Loss')
    ax1.set_title('Phase 1: Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2 = plt.subplot(2, 2, 2)
    loss_diff_phase1 = np.array(history1.history['loss']) - np.array(history1.history['val_loss'])
    ax2.plot(loss_diff_phase1, marker='o', label="Loss Difference")
    ax2.axhline(0, color='red', linestyle='--', label="Zero Difference")
    ax2.set_title('Phase 1: Training-Validation Loss Difference')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss Difference')
    ax2.legend()
    ax2.grid(True)
    
    # Phase 2 Metrics
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(history2.history['loss'], label='Training Loss')
    ax3.plot(history2.history['val_loss'], label='Validation Loss')
    ax3.set_title('Phase 2: Loss Curves')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True)
    
    ax4 = plt.subplot(2, 2, 4)
    loss_diff_phase2 = np.array(history2.history['loss']) - np.array(history2.history['val_loss'])
    ax4.plot(loss_diff_phase2, marker='o', label="Loss Difference")
    ax4.axhline(0, color='red', linestyle='--', label="Zero Difference")
    ax4.set_title('Phase 2: Training-Validation Loss Difference')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss Difference')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    metrics_plot_path = os.path.join(performance_dir, f'{version}_Training_Metrics.png')
    plt.savefig(metrics_plot_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Training metrics plot saved to {metrics_plot_path}")
    
def plot_range_specific_metrics(y_true, y_pred, performance_dir, version):
    # Create bins for different polarization ranges
    bins = [
        (0, 0.001),    # 0-0.1%
        (0.001, 0.01),  # 0.1-1%
        (0.01, 0.1),    # 1-10%
        (0.1, 0.8)     # 10-80%
    ]

    plt.figure(figsize=(18, 12))
    plt.suptitle(f'Range-Specific Metrics - {version}', y=1.02, fontsize=16)

    for i, (lower, upper) in enumerate(bins):
        # Filter data for current range
        mask = (y_true >= lower) & (y_true < upper)
        y_true_range = y_true[mask]
        y_pred_range = y_pred[mask]
        
        if len(y_true_range) == 0:
            continue
        
        # Calculate metrics
        residuals = y_true_range - y_pred_range
        mae = np.mean(np.abs(residuals))
        mse = np.mean(residuals**2)
        
        # Create subplot
        ax = plt.subplot(2, 2, i+1)
        
        # Scatter plot of predictions vs true values
        ax.scatter(y_true_range, y_pred_range, alpha=0.5, label=f'Predictions (MAE: {mae:.2e}, MSE: {mse:.2e})')
        ax.plot([lower, upper], [lower, upper], 'r--', label='Ideal Prediction')
        ax.set_title(f'Polarization Range: {lower*100:.1f}% to {upper*100:.1f}%')
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predictions')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    range_metrics_path = os.path.join(performance_dir, f'{version}_Range_Specific_Metrics.png')
    plt.savefig(range_metrics_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Range-specific metrics plot saved to {range_metrics_path}")
    
    
def plot_histogram(data, title, xlabel, ylabel, color, ax, num_bins=100, plot_norm=True):
    n, bins, patches = plt.hist(data, num_bins, density=True, color=color, alpha=0.7)
    mu, sigma = norm.fit(data)

    if plot_norm:
        y = norm.pdf(bins, mu, sigma)
        plt.plot(bins, y, '--', color='black')

    plt.title(f"{title}: μ={mu:.4f}, σ={sigma:.4f}", fontsize = 16,weight='bold')
    plt.xlabel(xlabel, fontsize = 16,weight='bold')
    plt.ylabel(ylabel, fontsize = 16,weight='bold')
    ax.tick_params(axis='both', which='major', labelsize=12)  
    ax.tick_params(axis='both', which='minor', labelsize=12)  
    plt.grid(False)
    # plt.savefig(save_path)
    # plt.close()