#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.font_manager as font_manager

def load_tensor_data(file_path):
    """
    Load tensor polarization data from a Parquet file.
    
    Parameters:
    -----------
    file_path : str
        Path to the Parquet file
        
    Returns:
    --------
    tuple
        (signals, P_values, SNR_values, metadata)
        - signals: numpy array of shape (n_samples, frequency_bins, phi_bins)
        - P_values: numpy array of polarization values
        - SNR_values: numpy array of SNR values (if available)
        - metadata: dictionary containing data structure information
    """
    # Load the Parquet file
    df = pd.read_parquet(file_path)
    
    # Get metadata
    metadata = eval(df.attrs['metadata'])
    
    if metadata['polarization_type'] == 'tensor':
        # Get the signal column and drop P and SNR
        signal_df = df.drop(['P', 'SNR'] if 'SNR' in df.columns else ['P'], axis=1)
        
        # Convert to numpy array and reshape
        signals = np.array([np.array(sig).reshape(metadata['frequency_bins'], metadata['phi_bins']) 
                          for sig in df['signal'].values])
                
        P_values = df['P'].values
        SNR_values = df['SNR'].values if 'SNR' in df.columns else None
        
        print(f"Debug - Signal shape after reshape: {signals.shape}")
        print(f"Debug - Sample signal range: [{signals[0].min():.6f}, {signals[0].max():.6f}]")
        
        return signals, P_values, SNR_values, metadata
    else:
        raise ValueError(f"Unsupported polarization type: {metadata['polarization_type']}")

def plot_tensor_event(signal, P_value, SNR=None, event_idx=None, save_path=None):
    """
    Plot a single tensor event as a 2D heatmap in the style of phase_lineshape.py.
    
    Parameters:
    -----------
    signal : numpy.ndarray
        2D array of shape (frequency_bins, phi_bins) containing the signal data
    P_value : float
        Polarization value for this event
    SNR : float, optional
        Signal-to-noise ratio for this event
    event_idx : int, optional
        Index of the event being plotted
    save_path : str, optional
        Path to save the plot
    """
    # Set up the figure with black background
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111)
    
    # Print debug information
    print(f"Debug - Plot signal shape: {signal.shape}")
    print(f"Debug - Plot signal range: [{signal.min():.6f}, {signal.max():.6f}]")
    
    # Create the heatmap with proper extent and normalization
    im = plt.imshow(signal.T,  # Transpose to match phase_lineshape.py orientation
                   aspect='auto',
                   origin='lower',
                   extent=[30.88, 34.48, 0, 180],  # Swap axes to match phase_lineshape.py
                   cmap='magma')
                #    norm=LogNorm(vmin=max(signal.min(), 1e-6), vmax=signal.max()))
    
    # Add colorbar
    cbar = plt.colorbar(im, orientation='horizontal', pad=0.15)
    cbar.set_label('Signal Amplitude', fontsize=30, fontweight='bold')
    cbar.ax.tick_params(labelsize=22)
    
    # Labels and title
    plt.xlabel('Frequency (MHz)', fontsize=24)
    plt.ylabel('φ (degrees)', fontsize=24)
    
    title = f'Tensor Signal (P = {P_value:.4f})'
    if event_idx is not None:
        title = f'Event {event_idx}: ' + title
    if SNR is not None:
        title += f' (SNR = {SNR:.2f})'
    plt.title(title, fontsize=28, pad=20)
    
    # Format ticks
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # Set up major and minor ticks
    major_ticks_x = np.arange(30.5, 35, 0.5)
    minor_ticks_x = np.arange(30.5, 35, 0.1)
    major_ticks_y = np.arange(0, 181, 60)
    minor_ticks_y = np.arange(0, 181, 30)
    
    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)
    
    # Add grid
    ax.grid(True, which='major', linestyle='-', linewidth=1.5, alpha=0.7)
    ax.grid(True, which='minor', linestyle='--', linewidth=1, alpha=0.4)
    
    # Set background color
    ax.set_facecolor('black')
    
    # Ensure the save directory exists
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()
    else:
        plt.show()

def plot_baseline_phase_variation(signal, P_value, metadata, save_path=None):
    """
    Plot the signal data showing frequency vs amplitude with phi angle as color.
    
    Parameters:
    -----------
    signal : numpy.ndarray
        2D array of shape (signal_intensity, phase_angles) containing the signal data
    P_value : float
        Polarization value for this event
    metadata : dict
        Dictionary containing data structure information
    save_path : str, optional
        Path to save the plot
    """
    # Set up the figure with black background
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(1, 1, 1)
    
    # Create frequency array
    freq = np.linspace(30.88, 34.48, signal.shape[0])
    
    # Create a ScalarMappable for the colorbar
    norm = plt.Normalize(0, 180)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.magma, norm=norm)
    sm.set_array([])  # This is required for the colorbar to work
    
    # Plot each phase angle line with color mapping
    for i in range(signal.shape[1]):
        phi = i * (180 / signal.shape[1])  # Convert index to phi angle
        plt.plot(freq, signal[:, i], color=plt.cm.magma(norm(phi)), 
                linewidth=2, alpha=0.7)
    
    # Add colorbar for phi angle
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.15)
    cbar.set_label('φ (degrees)', fontsize=30, fontweight='bold')
    cbar.ax.tick_params(labelsize=22)
    
    # Labels and title
    plt.xlabel('Frequency (MHz)', fontsize=24)
    plt.ylabel('Signal Intensity', fontsize=24)
    plt.title(f'Signal Intensity vs Frequency for P = {P_value:.4f}', fontsize=28, pad=20)
    
    # Format ticks
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # Set up major and minor ticks
    major_ticks_x = np.arange(30.5, 35, 0.5)
    minor_ticks_x = np.arange(30.5, 35, 0.1)
    
    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    
    # Add grid
    ax.grid(True, which='major', linestyle='-', linewidth=1.5, alpha=0.7)
    ax.grid(True, which='minor', linestyle='--', linewidth=1, alpha=0.4)
    
    # Set background color
    ax.set_facecolor('black')
    
    # Add legend for phase angles
    handles, labels = ax.get_legend_handles_labels()
    # Only show every nth label to avoid overcrowding
    n = max(1, len(labels) // 8)  # Show approximately 8 labels
    ax.legend([handles[i] for i in range(0, len(handles), n)],
             [labels[i] for i in range(0, len(labels), n)],
             loc='upper right', fontsize=16)
    
    # Ensure the save directory exists
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Test and visualize tensor polarization data.')
    parser.add_argument('--file', required=True, help='Path to the Parquet file')
    parser.add_argument('--event', type=int, default=0, help='Index of the event to plot')
    parser.add_argument('--save', default='plots/', help='Path to save the plot (optional)')
    args = parser.parse_args()
    
    # Load the data
    signals, P_values, SNR_values, metadata = load_tensor_data(args.file)
    
    # Print some information about the data
    print(f"\nData Information:")
    print(f"Number of events: {len(signals)}")
    print(f"Signal shape: {signals[0].shape}")
    print(f"Polarization range: [{P_values.min():.4f}, {P_values.max():.4f}]")
    
    # Check if SNR_values exists and contains valid numbers
    if SNR_values is not None:
        # Convert to numpy array and check if any valid numbers exist
        snr_array = np.array(SNR_values, dtype=float)
        if not np.all(np.isnan(snr_array)):
            print(f"SNR range: [{np.nanmin(snr_array):.2f}, {np.nanmax(snr_array):.2f}]")
        else:
            print("No valid SNR values available (no noise added)")
    else:
        print("No SNR values available (no noise added)")
    print(f"\nMetadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    # Plot the selected event
    event_idx = args.event
    if event_idx >= len(signals):
        print(f"Error: Event index {event_idx} is out of range (max: {len(signals)-1})")
        return
    
    # Get SNR value for this event, handling None/NaN cases
    event_snr = None
    if SNR_values is not None:
        try:
            snr_value = float(SNR_values[event_idx])
            if not np.isnan(snr_value):
                event_snr = snr_value
        except (ValueError, TypeError):
            pass
    
    tensor_save_path = os.path.join(args.save, 'tensor_signal.jpeg')
    
    plot_tensor_event(
        signals[event_idx],
        P_values[event_idx],
        event_snr,
        event_idx,
        tensor_save_path
    )
    
    # Plot baseline variation
    baseline_save_path = os.path.join(args.save, 'baseline_phase_variation.jpeg')
    plot_baseline_phase_variation(signals[event_idx], P_values[event_idx], metadata, baseline_save_path)

if __name__ == "__main__":
    main() 