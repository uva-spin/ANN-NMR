import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import pandas as pd
from matplotlib.ticker import MaxNLocator
    
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
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)

        # Create subplot for histogram of residuals using Seaborn
        ax = plt.subplot(2, 3, valid_bins)  # Use valid_bins for positioning
        sns.histplot(residuals, bins=30, kde=False, stat="density", color='blue', edgecolor='black', ax=ax, alpha=0.6)
        


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
    

    

def plot_training_history(history, performance_dir, version, figsize=(14, 8), dpi=600):
    """
    Plots training and validation loss and accuracy from the training history.

    Parameters:
    - history: A History object returned by the fit method of a Keras model.
    - performance_dir: Directory to save the plot.
    - version: Version identifier for the plot filename.
    - figsize: Size of the figure as a tuple (width, height).
    - dpi: Resolution of the saved figure.
    """
    # Convert history to DataFrame for better plotting with Seaborn
    history_df = pd.DataFrame(history.history)
    
    # Create the directory if it doesn't exist
    os.makedirs(performance_dir, exist_ok=True)
    
    # Determine available metrics for plotting
    metric_pairs = []
    if 'loss' in history_df and 'val_loss' in history_df:
        metric_pairs.append(('loss', 'val_loss', 'Loss'))
    if 'accuracy' in history_df and 'val_accuracy' in history_df:
        metric_pairs.append(('accuracy', 'val_accuracy', 'Accuracy'))
    if 'mae' in history_df and 'val_mae' in history_df:
        metric_pairs.append(('mae', 'val_mae', 'Mean Absolute Error'))
    if 'mse' in history_df and 'val_mse' in history_df:
        metric_pairs.append(('mse', 'val_mse', 'Mean Squared Error'))
    
    # Define plot aesthetics
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    markers = ['o', 's', 'D', '^']
    
    # Create subplots based on number of metrics
    n_metrics = len(metric_pairs)
    if n_metrics == 0:
        print("No valid metric pairs found in history.")
        return
    
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]  # Ensure axes is iterable even with one subplot
    
    # Plot each metric
    for i, (train_metric, val_metric, title) in enumerate(metric_pairs):
        ax = axes[i]
        
        # Get data for plotting
        epochs = np.arange(1, len(history_df) + 1)
        train_values = history_df[train_metric]
        val_values = history_df[val_metric]
        
        # Plot with error bands (assuming we have multiple runs)
        sns.lineplot(x=epochs, y=train_values, label=f'Training {title}', 
                     color=colors[0], marker=markers[0], ax=ax)
        sns.lineplot(x=epochs, y=val_values, label=f'Validation {title}', 
                     color=colors[1], marker=markers[1], ax=ax)
        
        # Find and annotate best epoch
        best_epoch = val_values.idxmin() if 'loss' in val_metric else val_values.idxmax()
        best_value = val_values[best_epoch]
        ax.axvline(x=best_epoch + 1, color='gray', linestyle='--', alpha=0.7)
        ax.annotate(f'Best: {best_value:.6f} (Epoch {best_epoch + 1})',
                    xy=(best_epoch + 1, best_value),
                    xytext=(best_epoch + 1 + 0.5, best_value + (0.1 * best_value)),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        
        # Set plot labels and styling
        ax.set_title(f'Training and Validation {title}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Epochs', fontsize=14)
        ax.set_ylabel(title, fontsize=14)
        ax.legend(loc='best', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Force integer x-axis
    
    # Add overall title
    fig.suptitle(f'Model Training Performance - {version}', fontsize=18, fontweight='bold', y=1.05)
    
    # Save plot with tight layout
    plt.tight_layout()
    plot_path = os.path.join(performance_dir, f'{version}_Training_History.png')
    plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Training history plot saved to {plot_path}")
    
def plot_rpe_and_residuals(y_true, y_pred, performance_dir, version, figsize=(18, 10), dpi=600):
    """
    Plots comprehensive performance metrics including Relative Percent Error (RPE), 
    residuals, and prediction vs actual for given true and predicted values.
    Also creates a separate focused histogram of RPE for data points where y_true is 
    around 0.05% ± 0.01% and a 3D histogram showing RPE distribution across true values.
    Displays mean and standard deviation statistics without Gaussian fits.

    Parameters:
    - y_true: Array of true values.
    - y_pred: Array of predicted values.
    - performance_dir: Directory to save the plot.
    - version: Version identifier for the plot filename.
    - figsize: Size of the figure as a tuple (width, height).
    - dpi: Resolution of the saved figure.
    """
    
    ### Debugging Message ###
    print(f"Plotting RPE and Residuals for {version}")
     
    # Ensure directory exists
    os.makedirs(performance_dir, exist_ok=True)
    
    y_true = np.array(y_true).flatten()*100
    y_pred = np.array(y_pred).flatten()*100
    
    # Calculate metrics
    residuals = y_true - y_pred
    percentage_error = (residuals / y_true) * 100
    rpe = np.abs(percentage_error)  # Relative Percent Error
    
    # Calculate statistical measures
    mae = np.mean(np.abs(residuals))
    mape = np.mean(rpe)
    rmse = np.sqrt(np.mean(residuals**2))
    
    # Calculate correlation coefficient
    corr_coef = np.corrcoef(y_true, y_pred)[0, 1]
    
    # Create a DataFrame for easy filtering and analysis
    results_df = pd.DataFrame({
        'True': y_true,
        'Predicted': y_pred,
        'Residuals': residuals,
        'Percentage_Error': percentage_error,
        'RPE': rpe
    })
    
    
    # Remove any potential infinity or NaN values
    results_df = results_df.replace([np.inf, -np.inf], np.nan).dropna()
    
    results_df.to_csv(os.path.join(performance_dir, f'{version}_results.csv'), index=False)
    
    # Style settings
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    # Create figure with 2x2 subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3)
    
    # 1. Actual vs Predicted Plot
    ax1 = fig.add_subplot(gs[0, 0])
    # Plot the scatter points
    sns.scatterplot(x=results_df['True'], y=results_df['Predicted'], alpha=0.6, 
                   color=colors[0], edgecolor='k', s=50, ax=ax1)
    
    # Add perfect prediction line
    min_val = min(results_df['True'].min(), results_df['Predicted'].min())
    max_val = max(results_df['True'].max(), results_df['Predicted'].max())
    margin = 0.1 * (max_val - min_val)
    ax1.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin], 
             '--', color='gray', linewidth=1.5, label='Perfect Prediction')
    
    # Add regression line
    sns.regplot(x=results_df['True'], y=results_df['Predicted'], 
                scatter=False, color=colors[1], line_kws={'linewidth': 2}, ax=ax1)
    
    # Add text annotation with metrics
    ax1.text(0.05, 0.95, f'Correlation: {corr_coef:.6f}\nMAE: {mae:.6f}\nRMSE: {rmse:.6f}',
             transform=ax1.transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax1.set_title('Actual vs. Predicted Values', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Actual Values', fontsize=12)
    ax1.set_ylabel('Predicted Values', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='best')
    
    # 2. Residuals Plot
    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(results_df['Residuals'], bins=30, kde=False, stat="density", 
                color=colors[2], edgecolor='black', alpha=0.6, ax=ax2)
    
    # Calculate and display mean and standard deviation
    residuals_mean = results_df['Residuals'].mean()
    residuals_std = results_df['Residuals'].std()
    
    # Add vertical line at the mean
    ax2.axvline(x=residuals_mean, color='red', linestyle='-', linewidth=1.5, 
               label=f'Mean: {residuals_mean:.6f}')
    
    # Shade the range within ±1 standard deviation
    ax2.axvspan(residuals_mean - residuals_std, residuals_mean + residuals_std, 
               alpha=0.2, color='red', 
               label=f'±1σ: [{residuals_mean-residuals_std:.6f}, {residuals_mean+residuals_std:.6f}]')
    
    ax2.set_title('Residuals Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Residuals', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 3. RPE Histogram
    ax3 = fig.add_subplot(gs[0, 2])
    sns.histplot(results_df['RPE'], bins=30, kde=False, stat="density", 
                color=colors[3], edgecolor='black', alpha=0.6, ax=ax3)
    
    # Calculate and display mean and standard deviation
    rpe_mean = results_df['RPE'].mean()
    rpe_std = results_df['RPE'].std()
    
    # Add vertical line at the mean
    ax3.axvline(x=rpe_mean, color='orange', linestyle='-', linewidth=1.5, 
               label=f'Mean: {rpe_mean:.6f}')
    
    # Shade the range within ±1 standard deviation
    ax3.axvspan(rpe_mean - rpe_std, rpe_mean + rpe_std, 
               alpha=0.2, color='orange', 
               label=f'±1σ: [{rpe_mean-rpe_std:.6f}, {rpe_mean+rpe_std:.6f}]')
    
    ax3.set_title('RPE Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Relative Percent Error (%)', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.legend(loc='best')
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # 4. Residuals vs. True Values
    ax4 = fig.add_subplot(gs[1, 0])
    sns.scatterplot(x=results_df['True'], y=results_df['Residuals'], 
                   alpha=0.6, color=colors[4], edgecolor='k', s=50, ax=ax4)
    ax4.axhline(y=0, color='r', linestyle='-', linewidth=1.5)
    
    # Add regression line to see trends
    sns.regplot(x=results_df['True'], y=results_df['Residuals'], 
                scatter=False, color='green', line_kws={'linewidth': 2}, ax=ax4)
    
    ax4.set_title('Residuals vs. True Values', fontsize=14, fontweight='bold')
    ax4.set_xlabel('True Values', fontsize=12)
    ax4.set_ylabel('Residuals', fontsize=12)
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    # 5. RPE vs. True Values
    ax5 = fig.add_subplot(gs[1, 1])
    sns.scatterplot(x=results_df['True'], y=results_df['RPE'], 
                   alpha=0.6, color='purple', edgecolor='k', s=50, ax=ax5)
    
    # Calculate and display the mean RPE
    mean_rpe = results_df['RPE'].mean()
    median_rpe = results_df['RPE'].median()
    ax5.axhline(y=mean_rpe, color='r', linestyle='-', linewidth=1.5, 
               label=f'Mean RPE: {mean_rpe:.2f}%')
    ax5.axhline(y=median_rpe, color='green', linestyle='--', linewidth=1.5, 
               label=f'Median RPE: {median_rpe:.2f}%')
    
    ax5.set_title('RPE vs. True Values', fontsize=14, fontweight='bold')
    ax5.set_xlabel('True Values', fontsize=12)
    ax5.set_ylabel('Relative Percent Error (%)', fontsize=12)
    ax5.legend(loc='best')
    ax5.grid(True, linestyle='--', alpha=0.7)
    
    # 6. Error Distribution
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Calculate error distribution statistics
    error_stats = pd.DataFrame({
        'Metric': ['Mean Abs Error', 'Mean RPE', 'RMSE', 'Correlation'],
        'Value': [mae, mape, rmse, corr_coef]
    })
    
    # Create horizontal bar chart
    sns.barplot(y='Metric', x='Value', data=error_stats, palette='viridis', ax=ax6)
    
    # Add value labels on bars
    for i, v in enumerate(error_stats['Value']):
        ax6.text(v + 0.01, i, f'{v:.6f}', va='center')
    
    ax6.set_title('Error Metrics Summary', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Value', fontsize=12)
    ax6.grid(True, linestyle='--', alpha=0.7, axis='x')
    
    # Add overall title and adjust layout
    fig.suptitle(f'Model Performance Evaluation - {version}', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(performance_dir, f'{version}_Model_Performance.png')
    plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Performance evaluation plots saved to {plot_path}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Metric': ['MAE', 'MAPE', 'RMSE', 'Correlation'],
        'Value': [mae, mape, rmse, corr_coef]
    })
    metrics_path = os.path.join(performance_dir, f'{version}_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Performance metrics saved to {metrics_path}")
    
    # Create a separate histogram for RPE where y_true is around 0.05% ± 0.01%
    target_true = 0.05  # 0.05%
    range_true = 0.01   # 0.01%
    
    # Filter for data points where y_true is in the target range
    filtered_df = results_df[(results_df['True'] >= target_true - range_true) & 
                            (results_df['True'] <= target_true + range_true)]
    
    # Create a new figure for the focused histogram
    plt.figure(figsize=(10, 6))
    
    if len(filtered_df) > 0:
        # Create histogram with KDE for the percentage errors of the filtered data
        ax = sns.histplot(filtered_df['Percentage_Error'], bins=50, kde=False, stat="density", 
                         color="#3498db", edgecolor='black', alpha=0.7)
        
        # Calculate mean and standard deviation
        mu = filtered_df['Percentage_Error'].mean()
        sigma = filtered_df['Percentage_Error'].std()
        
        # Add vertical line at the mean
        plt.axvline(x=mu, color='green', linestyle='-', linewidth=1.5, 
                   label=f'Mean: {mu:.8f}%')
        
        # Shade the range within ±1 sigma
        plt.axvspan(mu - sigma, mu + sigma, alpha=0.2, color='green', 
                   label=f'±1σ: [{mu-sigma:.8f}%, {mu+sigma:.8f}%]')
        
        # Add vertical line at zero (perfect prediction)
        plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5,
                   label='Perfect Prediction (0%)')
        
        # Add statistics annotation
        stats_text = (
            f'N: {len(filtered_df)}\n'
            f'Min: {filtered_df["Percentage_Error"].min():.8f}%\n'
            f'Max: {filtered_df["Percentage_Error"].max():.8f}%\n'
            f'Mean: {filtered_df["Percentage_Error"].mean():.8f}%\n'
            f'Median: {filtered_df["Percentage_Error"].median():.8f}%\n'
            f'Std Dev: {filtered_df["Percentage_Error"].std():.8f}%\n'
            f'Abs Mean: {np.abs(filtered_df["Percentage_Error"]).mean():.8f}%'
        )
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        plt.text(0.5, 0.5, f"No data points found in the range {target_true - range_true}% to {target_true + range_true}%",
                horizontalalignment='center', verticalalignment='center', fontsize=14)
    
    # Add title and labels with high precision
    plt.title(f'Distribution of Percentage Errors for True Values around {target_true:.6f}% ± {range_true:.6f}%', 
             fontsize=16, fontweight='bold')
    plt.xlabel('Percentage Error (%)', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Save the focused histogram separately
    focused_path = os.path.join(performance_dir, f'{version}_Focused_PE_Histogram_True_{target_true}.png')
    plt.savefig(focused_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Focused percentage error histogram for y_true ≈ {target_true}% saved to {focused_path}")
    
    # NEW ADDITION: Create a 3D histogram of RPE vs. True polarization values
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define number of bins for the 3D histogram
    true_bins = 20
    rpe_bins = 20
    
    # Calculate the histogram
    hist, x_edges, y_edges = np.histogram2d(
        results_df['True'], 
        results_df['RPE'], 
        bins=[true_bins, rpe_bins]
    )
    
    # Get the centers of the bins
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    
    # Create a mesh grid for the 3D plot
    X, Y = np.meshgrid(x_centers, y_centers)
    
    # Plot the 3D histogram as a surface
    surf = ax.plot_surface(
        X, Y, hist.T, 
        cmap='viridis',
        rstride=1, cstride=1, 
        alpha=0.8, 
        linewidth=0, 
        antialiased=True
    )
    
    # Add a color bar to show the mapping
    colorbar = fig.colorbar(surf, ax=ax, shrink=0.7, aspect=10)
    colorbar.set_label('Frequency', fontsize=12)
    
    # Set labels
    ax.set_xlabel('True Polarization Values (%)', fontsize=12)
    ax.set_ylabel('Relative Percent Error (%)', fontsize=12)
    ax.set_zlabel('Frequency', fontsize=12)
    
    # Set title
    ax.set_title('3D Distribution of RPE across True Polarization Values', fontsize=16, fontweight='bold')
    
    # Adjust the view angle for better visualization
    ax.view_init(elev=30, azim=45)
    
    # Save the 3D histogram
    plt.tight_layout()
    plot3d_path = os.path.join(performance_dir, f'{version}_3D_RPE_Distribution.png')
    plt.savefig(plot3d_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"3D RPE distribution histogram saved to {plot3d_path}")
    
    # Alternative 2D heatmap visualization of the same data (often easier to interpret)
    plt.figure(figsize=(12, 8))
    
    # Create pivot table for heatmap
    heatmap_data = pd.DataFrame({
        'True': results_df['True'],
        'RPE': results_df['RPE'],
        'Count': 1
    })
    
    # Create bins for True values and RPE
    true_bins_edges = np.linspace(results_df['True'].min(), results_df['True'].max(), 15)
    rpe_bins_edges = np.linspace(results_df['RPE'].min(), results_df['RPE'].max(), 15)
    
    # Assign data points to bins
    heatmap_data['True_bin'] = pd.cut(heatmap_data['True'], bins=true_bins_edges, labels=true_bins_edges[:-1])
    heatmap_data['RPE_bin'] = pd.cut(heatmap_data['RPE'], bins=rpe_bins_edges, labels=rpe_bins_edges[:-1])
    
    # Create pivot table
    pivot_data = heatmap_data.pivot_table(
        values='Count', 
        index='RPE_bin', 
        columns='True_bin', 
        aggfunc='count',
        fill_value=0
    )
    
    # Plot heatmap
    sns.heatmap(
        pivot_data, 
        cmap='viridis', 
        annot=False, 
        fmt='d', 
        linewidths=0,
        cbar_kws={'label': 'Frequency'}
    )
    
    plt.title('Heatmap of RPE Distribution across True Polarization Values', fontsize=16, fontweight='bold')
    plt.xlabel('True Polarization Values (%)', fontsize=12)
    plt.ylabel('Relative Percent Error (%)', fontsize=12)
    
    # Adjust tick frequency for readability
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # Save the heatmap
    plt.tight_layout()
    heatmap_path = os.path.join(performance_dir, f'{version}_RPE_Heatmap.png')
    plt.savefig(heatmap_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"RPE distribution heatmap saved to {heatmap_path}")
    
    return {
        'mae': mae,
        'mape': mape,
        'rmse': rmse,
        'correlation': corr_coef
    }
    
    
def plot_enhanced_results(y_true, y_pred, output_dir, version_name):
    """
    Create detailed, visually stunning plots for model evaluation as separate files
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    output_dir : str
        Directory to save the plots
    version_name : str
        Name of the model version for plot titles and filenames
    """
    
    ### Debugging Message ###
    print(f"Plotting Enhanced Results for {version_name}")
    
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    from matplotlib import cm
    import seaborn as sns
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set seaborn style for better aesthetics
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2, 
            rc={"lines.linewidth": 2.5, "font.sans-serif": ['Arial', 'DejaVu Sans']})
    
    # Prepare data
    y_true = y_true.flatten()*100
    y_pred = y_pred.flatten()*100
    residuals = (y_true - y_pred)
    rpe = np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-10))
    
    # Common settings for all plots
    plt.rcParams.update({'figure.autolayout': True})
    
    # Custom color palette
    custom_palette = sns.color_palette("viridis", n_colors=10)
    
    # 1. True vs Predicted Values Plot
    plt.figure(figsize=(10, 8))
    
    # Create a density heatmap using hexbin
    hb = plt.hexbin(y_true, y_pred, gridsize=50, cmap='viridis', mincnt=1, 
                    bins='log', alpha=0.9)
    
    # Add color bar
    cb = plt.colorbar(hb, label='log10(count)')
    cb.ax.tick_params(labelsize=10)
    
    # Add diagonal line
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    buffer = (max_val - min_val) * 0.05
    plt.plot([min_val-buffer, max_val+buffer], [min_val-buffer, max_val+buffer], 
             'r--', linewidth=1.5, alpha=0.8, label='Perfect Prediction')
    
    # Enhance aesthetics
    plt.xlabel('True Polarization (%)', fontsize=12, fontweight='bold')
    plt.ylabel('Predicted Polarization (%)', fontsize=12, fontweight='bold')
    plt.title(f'True vs Predicted Values - {version_name}', fontsize=16, fontweight='bold', pad=20)
    
    # Add annotations
    r2 = np.corrcoef(y_true, y_pred)[0, 1]**2
    plt.annotate(f'R² = {r2:.4f}', 
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", fc=custom_palette[0], ec="none", alpha=0.8),
                fontsize=12, color='white')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{version_name}_true_vs_pred.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Residuals Distribution Plot
    plt.figure(figsize=(10, 8))
    
    # Create a KDE plot for residuals
    sns.histplot(residuals, bins=100, kde=False, color=custom_palette[2], alpha=0.7, stat='density')
    
    mean = np.mean(residuals)
    std_dev = np.std(residuals)
    
    # Add reference line
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label='Zero Error')
    
    # Enhance aesthetics
    plt.xlabel('Residual Value (Percentage Points)', fontsize=12, fontweight='bold')
    plt.ylabel('Density', fontsize=12, fontweight='bold')
    plt.title(f'Residuals Distribution - {version_name}', fontsize=16, fontweight='bold', pad=20)
    
    # Add metrics text box
    stats_text = (f'Mean: {mean:.5f}\n'
                 f'Std Dev: {std_dev:.5f}\n'
                 f'MAE: {np.mean(np.abs(residuals)):.5f}')
    plt.text(0.05, 0.80, stats_text, transform=plt.gca().transAxes, fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", fc=custom_palette[0], ec="none", alpha=0.8),
            color='white')
    
    plt.legend(frameon=True, fancybox=True, facecolor='white', framealpha=0.9, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{version_name}_residuals_dist.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Residuals vs True Values Plot
    plt.figure(figsize=(10, 8))
    
    # Create a scatter plot with color based on density
    x = y_true
    y = residuals
    
    # Use kernel density estimate for coloring points by density
    from scipy.stats import gaussian_kde
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    
    scatter = plt.scatter(x, y, c=z, s=25, alpha=0.8, cmap='viridis', edgecolors='none')
    plt.colorbar(scatter, label='Density')
    
    # Add reference line
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=1.5)
    
    # Add trendline
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    plt.plot(x, intercept + slope*x, color=custom_palette[7], linewidth=2, 
             label=f'Trend: y={slope:.6f}x+{intercept:.6f}')
    
    # Enhance aesthetics
    plt.xlabel('True Polarization (%)', fontsize=12, fontweight='bold')
    plt.ylabel('Residual (Percentage Points)', fontsize=12, fontweight='bold')
    plt.title(f'Residuals vs True Values - {version_name}', fontsize=16, fontweight='bold', pad=20)
    
    plt.legend(frameon=True, fancybox=True, facecolor='white', framealpha=0.9, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{version_name}_residuals_vs_true.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. RPE Distribution Plot (excluding outliers)
    plt.figure(figsize=(10, 8))
    
    # Filter out extreme outliers for better visualization
    rpe_filtered = rpe[rpe < np.percentile(rpe, 99)]
    
    # Create a KDE plot for RPE
    sns.histplot(rpe_filtered, bins=100, kde=False, color=custom_palette[4], alpha=0.7, stat='density')
    
    mean_rpe = np.mean(rpe_filtered)
    std_dev_rpe = np.std(rpe_filtered)
    
    # Enhance aesthetics
    plt.xlabel('Relative Percentage Error (%)', fontsize=12, fontweight='bold')
    plt.ylabel('Density', fontsize=12, fontweight='bold')
    plt.title(f'RPE Distribution - {version_name}', fontsize=16, fontweight='bold', pad=20)
    
    # Add metrics text box
    stats_text = (f'Median: {np.median(rpe):.5f}%\n'
                 f'Mean: {mean_rpe:.5f}%\n'
                 f'Std Dev: {std_dev_rpe:.5f}%\n'
                 f'95th Percentile: {np.percentile(rpe, 95):.5f}%')
    plt.text(0.05, 0.80, stats_text, transform=plt.gca().transAxes, fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", fc=custom_palette[0], ec="none", alpha=0.8),
            color='white')
    
    plt.legend(frameon=True, fancybox=True, facecolor='white', framealpha=0.9, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{version_name}_rpe_dist.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. RPE vs True Values Plot
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot with log scale for y-axis
    plt.scatter(y_true, rpe, c=np.log1p(rpe), cmap='plasma', alpha=0.7, s=25, edgecolors='none')
    plt.colorbar(label='log(1+RPE)')
    
    # Set y-axis limit to exclude extreme outliers
    plt.ylim(0, np.percentile(rpe, 99))
    
    # Add a trend line
    from scipy import stats
    from sklearn.linear_model import LinearRegression
    
    # For plotting purposes, filter out outliers
    mask = rpe < np.percentile(rpe, 99)
    if np.sum(mask) > 0:  # Only fit if we have enough points
        lr = LinearRegression()
        lr.fit(y_true[mask].reshape(-1, 1), rpe[mask])
        x_plot = np.linspace(min(y_true), max(y_true), 100)
        plt.plot(x_plot, lr.predict(x_plot.reshape(-1, 1)), color='red', 
                linewidth=2, linestyle='-', label=f'Trend: y={lr.coef_[0]:.6f}x+{lr.intercept_:.6f}')
    
    # Enhance aesthetics
    plt.xlabel('True Polarization (%)', fontsize=12, fontweight='bold')
    plt.ylabel('Relative Percentage Error (%)', fontsize=12, fontweight='bold')
    plt.title(f'RPE vs True Values - {version_name}', fontsize=16, fontweight='bold', pad=20)
    
    plt.legend(frameon=True, fancybox=True, facecolor='white', framealpha=0.9, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{version_name}_rpe_vs_true.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Error Cumulative Distribution (Log Scale)
    plt.figure(figsize=(10, 8))
    
    # Sort errors and calculate cumulative percentage
    sorted_errors = np.sort(np.abs(residuals))
    cum_pct = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    
    # Create beautiful line plot
    plt.plot(sorted_errors, cum_pct, linewidth=3, color=custom_palette[6])
    plt.xscale('log')
    
    # Add grid with customization
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    
    # Add reference lines
    for err_val in [0.001, 0.01, 0.1, 1.0]:
        idx = np.searchsorted(sorted_errors, err_val)
        if idx < len(cum_pct):
            pct_below = cum_pct[idx]
            plt.axvline(x=err_val, color='gray', linestyle='--', alpha=0.5)
            plt.text(err_val*1.1, 50, f'{err_val}', rotation=90, alpha=0.8)
            plt.axhline(y=pct_below, color='gray', linestyle='--', alpha=0.5)
            plt.text(0.5*min(sorted_errors), pct_below+2, f'{pct_below:.1f}%', alpha=0.8)
    
    # Enhance aesthetics
    plt.xlabel('Absolute Error (log scale)', fontsize=12, fontweight='bold')
    plt.ylabel('Cumulative Percentage (%)', fontsize=12, fontweight='bold')
    plt.title(f'Error Cumulative Distribution - {version_name}', fontsize=16, fontweight='bold', pad=20)
    
    # Add metrics text box
    metrics_text = (
        f"MAE: {np.mean(np.abs(residuals)):.8f}\n"
        f"RMSE: {np.sqrt(np.mean(np.square(residuals))):.8f}\n"
        f"% within 0.001 abs error: {(np.abs(residuals) < 0.001).mean()*100:.4f}%\n"
        f"% within 0.1% rel error: {(rpe < 0.1).mean()*100:.4f}%"
    )
    plt.text(0.05, 0.25, metrics_text, transform=plt.gca().transAxes, fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", fc=custom_palette[0], ec="none", alpha=0.8),
            color='white')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{version_name}_error_cumulative.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Summary Metrics Plot (New visualization)
    plt.figure(figsize=(10, 6))
    
    # Calculate key metrics
    metrics = {
        'MAE': np.mean(np.abs(residuals)),
        'RMSE': np.sqrt(np.mean(np.square(residuals))),
        'Median RPE (%)': np.median(rpe),
        '95th Pct RPE (%)': np.percentile(rpe, 95),
        'R²': np.corrcoef(y_true, y_pred)[0, 1]**2
    }
    
    # Create a horizontal bar chart with custom colors
    bars = plt.barh(list(metrics.keys()), list(metrics.values()), 
                   color=custom_palette[:len(metrics)], alpha=0.8, height=0.6)
    
    # Add value labels to the end of each bar
    for i, bar in enumerate(bars):
        value = list(metrics.values())[i]
        plt.text(value + (max(metrics.values()) * 0.01), bar.get_y() + bar.get_height()/2, 
                f'{value:.6f}', va='center', fontsize=10, fontweight='bold')
    
    # Enhance aesthetics
    plt.xlabel('Value', fontsize=12, fontweight='bold')
    plt.title(f'Summary Metrics - {version_name}', fontsize=16, fontweight='bold', pad=20)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{version_name}_summary_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. Create an all-in-one metrics report (text file)
    with open(os.path.join(output_dir, f'{version_name}_metrics_report.txt'), 'w') as f:
        f.write(f"Model Evaluation Metrics - {version_name}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Basic Error Metrics:\n")
        f.write(f"MAE: {np.mean(np.abs(residuals)):.8f}\n")
        f.write(f"RMSE: {np.sqrt(np.mean(np.square(residuals))):.8f}\n")
        f.write(f"Max Absolute Error: {np.max(np.abs(residuals)):.8f}\n")
        f.write(f"R²: {np.corrcoef(y_true, y_pred)[0, 1]**2:.8f}\n\n")
        
        f.write("Relative Error Metrics:\n")
        f.write(f"Median RPE: {np.median(rpe):.5f}%\n")
        f.write(f"Mean RPE: {np.mean(rpe):.5f}%\n")
        f.write(f"90th Percentile RPE: {np.percentile(rpe, 90):.5f}%\n")
        f.write(f"95th Percentile RPE: {np.percentile(rpe, 95):.5f}%\n")
        f.write(f"99th Percentile RPE: {np.percentile(rpe, 99):.5f}%\n\n")
        
        f.write("Threshold Metrics:\n")
        f.write(f"% within 0.0001 abs error: {(np.abs(residuals) < 0.0001).mean()*100:.4f}%\n")
        f.write(f"% within 0.001 abs error: {(np.abs(residuals) < 0.001).mean()*100:.4f}%\n")
        f.write(f"% within 0.01 abs error: {(np.abs(residuals) < 0.01).mean()*100:.4f}%\n")
        f.write(f"% within 0.1 abs error: {(np.abs(residuals) < 0.1).mean()*100:.4f}%\n")
        f.write(f"% within 0.01% rel error: {(rpe < 0.01).mean()*100:.4f}%\n")
        f.write(f"% within 0.1% rel error: {(rpe < 0.1).mean()*100:.4f}%\n")
        f.write(f"% within 1% rel error: {(rpe < 1).mean()*100:.4f}%\n")
        f.write(f"% within 5% rel error: {(rpe < 5).mean()*100:.4f}%\n")

    
def plot_training_history(history, output_dir, version_name):
    """
    Plot training history with enhanced visualization and metrics
    
    Parameters:
    -----------
    history : dict
        Dictionary containing training metrics
    output_dir : str
        Directory to save the plots
    version_name : str
        Name of the model version for plot titles and filenames
    """
    
    ### Debugging Message ###
    print(f"Plotting Training History for {version_name}")
    
    history = history.history
    
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import MaxNLocator
    
    os.makedirs(output_dir, exist_ok=True)
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2,
            rc={"lines.linewidth": 2.5, "font.sans-serif": ['Arial', 'DejaVu Sans']})
    
    # Custom color palettes
    loss_palette = sns.color_palette("rocket", n_colors=4)
    metrics_palette = sns.color_palette("viridis", n_colors=6)
    
    # Get number of epochs
    epochs = range(1, len(history['loss']) + 1)
    
    # 1. Create Loss Plot (as a separate figure)
    plt.figure(figsize=(12, 8))
    
    # Plot loss curves with improved styling
    plt.plot(epochs, history['loss'], color=loss_palette[0], marker='o', markersize=4, 
             label='Training Loss', alpha=0.9, markevery=max(1, len(epochs)//20))
    plt.plot(epochs, history['val_loss'], color=loss_palette[2], marker='s', markersize=4, 
             label='Validation Loss', alpha=0.9, markevery=max(1, len(epochs)//20))
    
    # Add min/max markers
    min_val_loss_epoch = np.argmin(history['val_loss']) + 1
    min_val_loss = history['val_loss'][min_val_loss_epoch - 1]
    plt.scatter([min_val_loss_epoch], [min_val_loss], s=100, color=loss_palette[3], 
                marker='*', label=f'Best Val Loss: {min_val_loss:.6f}', zorder=5)
    
    # Set appropriate y-axis scale (log if the range is large)
    if max(history['loss']) / (min(history['loss']) + 1e-10) > 10:
        plt.yscale('log')
        plt.ylabel('Loss (log scale)', fontsize=12, fontweight='bold')
    else:
        plt.ylabel('Loss', fontsize=12, fontweight='bold')
    
    # Add annotations
    plt.annotate(f'Best: {min_val_loss:.6f} (Epoch {min_val_loss_epoch})',
                xy=(min_val_loss_epoch, min_val_loss), xytext=(min_val_loss_epoch + len(epochs)*0.05, min_val_loss*1.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8, alpha=0.7),
                fontsize=10, fontweight='bold')
    
    # Enhance aesthetics
    plt.title(f'Training and Validation Loss - {version_name}', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.9, fontsize=10)
    
    # Add final loss values
    final_train = history['loss'][-1]
    final_val = history['val_loss'][-1]
    loss_text = (f'Final Training Loss: {final_train:.6f}\n'
                f'Final Validation Loss: {final_val:.6f}\n'
                f'Best Validation Loss: {min_val_loss:.6f} (Epoch {min_val_loss_epoch})')
    
    plt.text(0.02, 0.02, loss_text, transform=plt.gca().transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=loss_palette[0], alpha=0.1))
    
    # Set x-axis to integers
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{version_name}_loss_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Create Metrics Plot (as a separate figure)
    plt.figure(figsize=(12, 8))
    
    # Plot metrics with improved styling
    plt.plot(epochs, history['mae'], color=metrics_palette[0], marker='o', markersize=4, 
             label='Training MAE', alpha=0.9, markevery=max(1, len(epochs)//20))
    plt.plot(epochs, history['val_mae'], color=metrics_palette[1], marker='s', markersize=4, 
             label='Validation MAE', alpha=0.9, markevery=max(1, len(epochs)//20))
    
    # Add min/max markers for MAE
    min_val_mae_epoch = np.argmin(history['val_mae']) + 1
    min_val_mae = history['val_mae'][min_val_mae_epoch - 1]
    plt.scatter([min_val_mae_epoch], [min_val_mae], s=100, color=metrics_palette[0], 
                marker='*', label=f'Best Val MAE: {min_val_mae:.6f}', zorder=5)
    
    # Add RMSE if available
    if 'rmse' in history:
        plt.plot(epochs, history['rmse'], color=metrics_palette[3], marker='o', linestyle='--',
                markersize=4, label='Training RMSE', alpha=0.9, markevery=max(1, len(epochs)//20))
        plt.plot(epochs, history['val_rmse'], color=metrics_palette[4], marker='s', linestyle='--',
                markersize=4, label='Validation RMSE', alpha=0.9, markevery=max(1, len(epochs)//20))
        
        # Add min/max markers for RMSE
        min_val_rmse_epoch = np.argmin(history['val_rmse']) + 1
        min_val_rmse = history['val_rmse'][min_val_rmse_epoch - 1]
        plt.scatter([min_val_rmse_epoch], [min_val_rmse], s=100, color=metrics_palette[4], 
                    marker='*', label=f'Best Val RMSE: {min_val_rmse:.6f}', zorder=5)
    
    # Enhance aesthetics
    plt.title(f'Training and Validation Metrics - {version_name}', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Metric Value', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.9, fontsize=10)
    
    # Add annotation for best metrics
    plt.annotate(f'Best MAE: {min_val_mae:.6f} (Epoch {min_val_mae_epoch})',
                xy=(min_val_mae_epoch, min_val_mae), xytext=(min_val_mae_epoch - len(epochs)*0.2, min_val_mae*1.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8, alpha=0.7),
                fontsize=10, fontweight='bold')
    
    if 'rmse' in history:
        plt.annotate(f'Best RMSE: {min_val_rmse:.6f} (Epoch {min_val_rmse_epoch})',
                    xy=(min_val_rmse_epoch, min_val_rmse), xytext=(min_val_rmse_epoch + len(epochs)*0.05, min_val_rmse*1.1),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8, alpha=0.7),
                    fontsize=10, fontweight='bold')
    
    # Set x-axis to integers
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{version_name}_metrics_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Create a Combined Overview Plot with Multiple Metrics
    plt.figure(figsize=(15, 10))
    
    # Custom colors for each metric
    colors = {
        'train_loss': loss_palette[0],
        'val_loss': loss_palette[2],
        'train_mae': metrics_palette[0],
        'val_mae': metrics_palette[1],
        'train_rmse': metrics_palette[3],
        'val_rmse': metrics_palette[4]
    }
    
    # Custom line styles
    styles = {
        'train_loss': '-',
        'val_loss': '-',
        'train_mae': '-',
        'val_mae': '-',
        'train_rmse': '--',
        'val_rmse': '--'
    }
    
    # Create 2x2 grid for combined plot
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[3, 1])
    
    # 1. Loss subplot
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(epochs, history['train_loss'], color=colors['train_loss'], linestyle=styles['train_loss'], 
             marker='o', markersize=3, label='Training Loss', markevery=max(1, len(epochs)//20))
    ax1.plot(epochs, history['val_loss'], color=colors['val_loss'], linestyle=styles['val_loss'], 
             marker='s', markersize=3, label='Validation Loss', markevery=max(1, len(epochs)//20))
    ax1.scatter([min_val_loss_epoch], [min_val_loss], s=80, color=colors['val_loss'], marker='*', zorder=5)
    
    # Improve aesthetics
    ax1.set_title('Loss History', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right', frameon=True)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 2. Metrics subplot
    ax2 = plt.subplot(gs[1, 0])
    ax2.plot(epochs, history['train_mae'], color=colors['train_mae'], linestyle=styles['train_mae'], 
             marker='o', markersize=3, label='Training MAE', markevery=max(1, len(epochs)//20))
    ax2.plot(epochs, history['val_mae'], color=colors['val_mae'], linestyle=styles['val_mae'], 
             marker='s', markersize=3, label='Validation MAE', markevery=max(1, len(epochs)//20))
    ax2.scatter([min_val_mae_epoch], [min_val_mae], s=80, color=colors['val_mae'], marker='*', zorder=5)
    
    if 'train_rmse' in history:
        ax2.plot(epochs, history['train_rmse'], color=colors['train_rmse'], linestyle=styles['train_rmse'], 
                marker='o', markersize=3, label='Training RMSE', markevery=max(1, len(epochs)//20))
        ax2.plot(epochs, history['val_rmse'], color=colors['val_rmse'], linestyle=styles['val_rmse'], 
                marker='s', markersize=3, label='Validation RMSE', markevery=max(1, len(epochs)//20))
        ax2.scatter([min_val_rmse_epoch], [min_val_rmse], s=80, color=colors['val_rmse'], marker='*', zorder=5)
    
    # Improve aesthetics
    ax2.set_title('Metrics History', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Metric Value', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right', frameon=True)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 3. Best Values Table
    ax3 = plt.subplot(gs[:, 1])
    ax3.axis('off')
    
    # Create a stylized table of best metrics
    best_metrics = {
        'Best Val Loss': (min_val_loss, min_val_loss_epoch),
        'Best Val MAE': (min_val_mae, min_val_mae_epoch),
    }
    
    if 'train_rmse' in history:
        best_metrics['Best Val RMSE'] = (min_val_rmse, min_val_rmse_epoch)
    
    # Add final values
    best_metrics['Final Train Loss'] = (history['loss'][-1], len(epochs))
    best_metrics['Final Val Loss'] = (history['val_loss'][-1], len(epochs))
    best_metrics['Final Train MAE'] = (history['mae'][-1], len(epochs))
    best_metrics['Final Val MAE'] = (history['val_mae'][-1], len(epochs))
    
    if 'train_rmse' in history:
        best_metrics['Final Train RMSE'] = (history['train_rmse'][-1], len(epochs))
        best_metrics['Final Val RMSE'] = (history['val_rmse'][-1], len(epochs))
    
    table_text = []
    for i, (metric, (value, epoch)) in enumerate(best_metrics.items()):
        table_text.append(f"{metric}: {value:.6f} (Epoch {epoch})")
    
    # Create a tabular-like display
    y_pos = 0.95
    for line in table_text:
        if 'Best' in line:
            ax3.text(0.1, y_pos, line, fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.3))
        else:
            ax3.text(0.1, y_pos, line, fontsize=10)
        y_pos -= 0.07
    
    # Add improvement metrics
    if y_pos > 0.1:
        ax3.text(0.1, y_pos, "Improvements:", fontsize=11, fontweight='bold')
        y_pos -= 0.06
        
        # Calculate improvements
        loss_improvement = (history['val_loss'][0] - min_val_loss) / history['val_loss'][0] * 100
        mae_improvement = (history['val_mae'][0] - min_val_mae) / history['val_mae'][0] * 100
        
        ax3.text(0.1, y_pos, f"Loss: {loss_improvement:.2f}%", fontsize=10)
        y_pos -= 0.05
        ax3.text(0.1, y_pos, f"MAE: {mae_improvement:.2f}%", fontsize=10)
        
        if 'train_rmse' in history:
            y_pos -= 0.05
            rmse_improvement = (history['val_rmse'][0] - min_val_rmse) / history['val_rmse'][0] * 100
            ax3.text(0.1, y_pos, f"RMSE: {rmse_improvement:.2f}%", fontsize=10)
    
    # Add a heading
    ax3.text(0.5, 1.02, "Training Summary", fontsize=14, fontweight='bold', ha='center')
    
    # Add border around the table
    border = plt.Rectangle((0.05, 0.03), 0.9, 0.94, fill=False, edgecolor='gray', 
                         linestyle='-', linewidth=1, alpha=0.5)
    ax3.add_patch(border)
    
    # Add title to the entire figure
    plt.suptitle(f'Training History Overview - {version_name}', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(output_dir, f'{version_name}_training_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # # 4. Create an interactive HTML report with embedded plots and metrics
    # try:
    #     html_content = f"""
    #     <!DOCTYPE html>
    #     <html>
    #     <head>
    #         <title>Training History Report - {version_name}</title>
    #         <style>
    #             body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
    #             .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    #             h1 {{ color: #333366; text-align: center; }}
    #             h2 {{ color: #333366; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
    #             .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
    #             .metric-card {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    #             .metric-value {{ font-size: 24px; font-weight: bold; color: #333366; margin: 5px 0; }}
    #             .metric-title {{ color: #666; margin-bottom: 5px; }}
    #             .plots {{ margin: 30px 0; text-align: center; }}
    #             .plot-img {{ max-width: 100%; height: auto; margin-bottom: 20px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.2); }}
    #             .footer {{ text-align: center; margin-top: 40px; color: #888; font-size: 12px; }}
    #         </style>
    #     </head>
    #     <body>
    #         <div class="container">
    #             <h1>Training History Report - {version_name}</h1>
                
    #             <h2>Key Metrics</h2>
    #             <div class="metrics-grid">
    #                 <div class="metric-card">
    #                     <div class="metric-title">Best Validation Loss</div>
    #                     <div class="metric-value">{min_val_loss:.6f}</div>
    #                     <div>Epoch {min_val_loss_epoch}</div>
    #                 </div>
                    
    #                 <div class="metric-card">
    #                     <div class="metric-title">Best Validation MAE</div>
    #                     <div class="metric-value">{min_val_mae:.6f}</div>
    #                     <div>Epoch {min_val_mae_epoch}</div>
    #                 </div>
                    
    #                 {'<div class="metric-card"><div class="metric-title">Best Validation RMSE</div><div class="metric-value">' + f"{min_val_rmse:.6f}" + '</div><div>Epoch ' + f"{min_val_rmse_epoch}" + '</div></div>' if 'train_rmse' in history else ''}
                    
    #                 <div class="metric-card">
    #                     <div class="metric-title">Final Training Loss</div>
    #                     <div class="metric-value">{history['train_loss'][-1]:.6f}</div>
    #                 </div>
                    
    #                 <div class="metric-card">
    #                     <div class="metric-title">Final Validation Loss</div>
    #                     <div class="metric-value">{history['val_loss'][-1]:.6f}</div>
    #                 </div>
                    
    #                 <div class="metric-card">
    #                     <div class="metric-title">Loss Improvement</div>
    #                     <div class="metric-value">{loss_improvement:.2f}%</div>
    #                 </div>
    #             </div>
                
    #             <h2>Training Plots</h2>
    #             <div class="plots">
    #                 <img src="{version_name}_loss_history.png" alt="Loss History" class="plot-img">
    #                 <img src="{version_name}_metrics_history.png" alt="Metrics History" class="plot-img">
    #                 <img src="{version_name}_training_overview.png" alt="Training Overview" class="plot-img">
    #             </div>
                
    #             <div class="footer">
    #                 Report generated at {import datetime; datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}
    #             </div>
    #         </div>
    #     </body>
    #     </html>
    #     """
        
    #     with open(os.path.join(output_dir, f'{version_name}_training_report.html'), 'w') as f:
    #         f.write(html_content)
    # except:
    #     # If HTML creation fails, silently continue (it's just a bonus feature)
    #     pass