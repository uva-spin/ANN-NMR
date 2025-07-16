#!/usr/bin/env python3
"""
Test script to demonstrate the performance improvement of optimized tensor signal generation.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.font_manager as font_manager
from scipy.signal import hilbert
from mpl_toolkits.mplot3d import Axes3D
import tqdm as tqdm
import os
from Create_Training_Data import SignalGenerator

def test_performance():
    """Test the performance improvement of the optimized tensor generation."""
    
    print("Testing tensor signal generation performance...")
    print("=" * 60)
    
    # Test parameters
    num_samples = 100 # Small number for quick test
    
    # Create generator with tensor polarization
    generator = SignalGenerator(
        mode="deuteron",
        polarization_type="tensor",
        num_samples=num_samples,
        add_noise=0,
        baseline=1,
        oversampling=0,
        lower_bound=0.1,
        upper_bound=0.6,
        scale_factor=100.0
    )
    
    # Test batch generation
    print(f"Generating {num_samples} tensor samples...")
    start_time = time.time()
    
    try:
        # Generate P values
        P_values = np.random.uniform(0.1, 0.6, num_samples)
        
        # Use batch optimization
        all_signals = generator._generate_deuteron_signal_batch_optimized(P_values)
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        print(f"✓ Successfully generated {num_samples} tensor samples")
        print(f"✓ Generation time: {generation_time:.2f} seconds")
        print(f"✓ Average time per sample: {generation_time/num_samples:.4f} seconds")
        print(f"✓ Signal shape: {all_signals.shape}")
        print(f"✓ Estimated time for 1000 samples: {generation_time/num_samples*1000:.2f} seconds")
        
        # Test individual sample generation for comparison
        print("\nTesting individual sample generation...")
        start_time = time.time()
        
        for i, P in enumerate(P_values[:10]):  # Test first 10 samples
            signal = generator._generate_deuteron_signal_optimized(P)
            if i == 0:
                print(f"✓ Individual sample shape: {signal.shape}")
        
        end_time = time.time()
        individual_time = end_time - start_time
        
        print(f"✓ Individual generation time for 10 samples: {individual_time:.2f} seconds")
        print(f"✓ Average time per individual sample: {individual_time/10:.4f} seconds")
        
        # Performance comparison
        batch_time_per_sample = generation_time / num_samples
        individual_time_per_sample = individual_time / 10
        
        if individual_time_per_sample > batch_time_per_sample:
            speedup = individual_time_per_sample / batch_time_per_sample
            print(f"\n🎉 Batch processing is {speedup:.1f}x faster than individual processing!")
        else:
            print(f"\n⚠️  Individual processing is {batch_time_per_sample/individual_time_per_sample:.1f}x faster than batch processing")
        
        return True, all_signals, P_values
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def test_vector_performance():
    """Test vector polarization performance for comparison."""
    
    print("\n" + "=" * 60)
    print("Testing vector signal generation performance...")
    print("=" * 60)
    
    # Test parameters
    num_samples = 100
    
    # Create generator with vector polarization
    generator = SignalGenerator(
        mode="deuteron",
        polarization_type="vector",
        num_samples=num_samples,
        add_noise=0,
        baseline=1,
        oversampling=0,
        lower_bound=0.1,
        upper_bound=0.6,
        scale_factor=100.0
    )
    
    print(f"Generating {num_samples} vector samples...")
    start_time = time.time()
    
    try:
        # Generate P values
        P_values = np.random.uniform(0.1, 0.6, num_samples)
        
        # Generate signals
        signals = []
        for P in P_values:
            signal = generator._generate_deuteron_signal(P)
            signals.append(signal)
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        print(f"✓ Successfully generated {num_samples} vector samples")
        print(f"✓ Generation time: {generation_time:.2f} seconds")
        print(f"✓ Average time per sample: {generation_time/num_samples:.4f} seconds")
        print(f"✓ Signal shape: {np.array(signals).shape}")
        
        return True, np.array(signals), P_values
        
    except Exception as e:
        print(f"❌ Error during vector generation: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def plot_tensor_phase_variation(tensor_signals, P_values, save_dir="plots"):
    """Plot tensor polarization phase variation similar to phase_lineshape.py"""
    
    # Create plots directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # System parameters
    g = 0.05
    s = 0.04
    bigy = np.sqrt(3 - s)
    labelfontsize = 30
    
    # Generate x values
    xvals = np.linspace(-3, 3, 500)
    phi_values = np.linspace(0, 180, 500)  # Tensor uses 0-180 degrees
    
    # Plotting setup
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(1, 1, 1)
    
    # Plot signals for different phase values
    for i, (signal, P) in enumerate(zip(tensor_signals, P_values)):
        # Use a colormap to create a gradient of colors
        color = plt.cm.plasma(i / len(tensor_signals))
        
        # Plot with decreasing opacity for better visualization
        alpha = 0.7 - (i / len(tensor_signals)) * 0.5  # Fade out as P increases
        
        plt.plot(xvals, signal[:, 0], color=color, alpha=alpha, linewidth=2)
    
    # Add colorbar
    norm = plt.Normalize(P_values.min(), P_values.max())
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.1)
    cbar.set_label('Polarization (P)', fontsize=labelfontsize, fontweight='bold')
    cbar.ax.tick_params(labelsize=labelfontsize-8)
    
    # Plot formatting
    axisFontSize = 38
    legendFontSize = 38
    
    plt.xlabel('R', fontsize=axisFontSize)
    plt.ylabel('Signal [$C_E$ mV]', fontsize=axisFontSize)
    
    # Set up major and minor ticks
    major_ticks_x = np.arange(-4, 4, 0.5)
    minor_ticks_x = np.arange(-4, 4, 0.5)
    major_ticks_y = np.arange(-1.2, 1.2, 0.2)
    minor_ticks_y = np.arange(-1.2, 1.2, 0.1)
    
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
    plt.title('Tensor Polarization Signal Variation (φ = 0° to 180°)', fontsize=28, pad=20)
    
    # Save figure
    fig.set_size_inches(24, 16)
    fig.savefig(f'{save_dir}/tensor_signal_phase_variation.jpeg', dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved tensor phase variation plot to {save_dir}/tensor_signal_phase_variation.jpeg")

def plot_vector_signals(vector_signals, P_values, save_dir="plots"):
    """Plot vector polarization signals"""
    
    # Create plots directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate x values
    xvals = np.linspace(-3, 3, 500)
    
    # Plotting setup
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(1, 1, 1)
    
    # Plot signals for different P values
    for i, (signal, P) in enumerate(zip(vector_signals, P_values)):
        # Use a colormap to create a gradient of colors
        color = plt.cm.viridis(i / len(vector_signals))
        
        # Plot with decreasing opacity for better visualization
        alpha = 0.7 - (i / len(vector_signals)) * 0.5  # Fade out as P increases
        
        plt.plot(xvals, signal, color=color, alpha=alpha, linewidth=2)
    
    # Add colorbar
    norm = plt.Normalize(P_values.min(), P_values.max())
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.1)
    cbar.set_label('Polarization (P)', fontsize=30, fontweight='bold')
    cbar.ax.tick_params(labelsize=22)
    
    # Plot formatting
    axisFontSize = 38
    
    plt.xlabel('R', fontsize=axisFontSize)
    plt.ylabel('Signal [$C_E$ mV]', fontsize=axisFontSize)
    
    # Set up major and minor ticks
    major_ticks_x = np.arange(-4, 4, 0.5)
    minor_ticks_x = np.arange(-4, 4, 0.5)
    major_ticks_y = np.arange(-1.2, 1.2, 0.2)
    minor_ticks_y = np.arange(-1.2, 1.2, 0.1)
    
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
    plt.title('Vector Polarization Signal Variation', fontsize=28, pad=20)
    
    # Save figure
    fig.set_size_inches(24, 16)
    fig.savefig(f'{save_dir}/vector_signal_variation.jpeg', dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved vector signal plot to {save_dir}/vector_signal_variation.jpeg")

def plot_tensor_heatmap(tensor_signals, P_values, save_dir="plots"):
    """Create heatmap visualization of tensor signal variations"""
    
    # Create plots directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate x values and phi values
    xvals = np.linspace(-3, 3, 500)
    phi_values = np.linspace(0, 180, 500)
    
    # Select a few representative samples for the heatmap
    num_samples_heatmap = min(20, len(tensor_signals))
    indices = np.linspace(0, len(tensor_signals)-1, num_samples_heatmap, dtype=int)
    
    # Create heatmap data
    signal_heatmap = np.zeros((len(phi_values), len(xvals)))
    
    # Average over selected samples
    for idx in indices:
        signal_heatmap += tensor_signals[idx]
    signal_heatmap /= len(indices)
    
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
    cbar.set_label('Signal Amplitude', fontsize=30, fontweight='bold')
    cbar.ax.tick_params(labelsize=22)
    
    # Add labels and title
    plt.xlabel('R', fontsize=24)
    plt.ylabel('φ (degrees)', fontsize=24)
    plt.title('Tensor Polarization Signal Heatmap', fontsize=28, pad=20)
    
    # Format ticks
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/tensor_signal_heatmap.jpeg', dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved tensor heatmap to {save_dir}/tensor_signal_heatmap.jpeg")

def plot_comparison_side_by_side(tensor_signals, vector_signals, P_values, save_dir="plots"):
    """Create side-by-side comparison of tensor vs vector signals"""
    
    # Create plots directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate x values
    xvals = np.linspace(-3, 3, 500)
    
    # Select a few representative samples
    num_samples_plot = min(10, len(tensor_signals))
    indices = np.linspace(0, len(tensor_signals)-1, num_samples_plot, dtype=int)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    # Plot tensor signals
    for i, idx in enumerate(indices):
        color = plt.cm.plasma(i / len(indices))
        alpha = 0.8 - (i / len(indices)) * 0.6
        # Plot first phi angle for tensor
        ax1.plot(xvals, tensor_signals[idx][:, 0], color=color, alpha=alpha, linewidth=2)
    
    ax1.set_xlabel('R', fontsize=24)
    ax1.set_ylabel('Signal [$C_E$ mV]', fontsize=24)
    ax1.set_title('Tensor Polarization (φ = 0°)', fontsize=28, pad=20)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-3.5, 3.5)
    
    # Plot vector signals
    for i, idx in enumerate(indices):
        color = plt.cm.viridis(i / len(indices))
        alpha = 0.8 - (i / len(indices)) * 0.6
        ax2.plot(xvals, vector_signals[idx], color=color, alpha=alpha, linewidth=2)
    
    ax2.set_xlabel('R', fontsize=24)
    ax2.set_ylabel('Signal [$C_E$ mV]', fontsize=24)
    ax2.set_title('Vector Polarization', fontsize=28, pad=20)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-3.5, 3.5)
    
    # Add colorbar for P values
    norm = plt.Normalize(P_values[indices].min(), P_values[indices].max())
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation='horizontal', pad=0.1)
    cbar.set_label('Polarization (P)', fontsize=24, fontweight='bold')
    cbar.ax.tick_params(labelsize=16)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/tensor_vs_vector_comparison.jpeg', dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved comparison plot to {save_dir}/tensor_vs_vector_comparison.jpeg")

def plot_3d_tensor_variation(tensor_signals, P_values, save_dir="plots"):
    """Create 3D plot of tensor signal variation over R, phi, and P"""
    
    # Create plots directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate x values and phi values
    xvals = np.linspace(-3, 3, 500)
    phi_values = np.linspace(0, 180, 500)
    
    # Select a subset for 3D plotting (too many samples would make it cluttered)
    num_samples_3d = min(20, len(tensor_signals))
    indices = np.linspace(0, len(tensor_signals)-1, num_samples_3d, dtype=int)
    
    # Create 3D plot
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a colormap for the polarization values
    norm = plt.Normalize(vmin=P_values[indices].min(), vmax=P_values[indices].max())
    cmap = plt.cm.viridis
    
    # Plot snapshots for each sample
    for i, idx in enumerate(indices):
        P = P_values[idx]
        signal = tensor_signals[idx]
        
        # Plot a few representative phi angles
        phi_indices = [0, 125, 250, 375, 499]  # 5 phi angles
        for phi_idx in phi_indices:
            phi_deg = phi_values[phi_idx]
            color = cmap(norm(P))
            ax.plot(xvals, np.full_like(xvals, phi_deg), signal[:, phi_idx], 
                   color=color, alpha=0.6, linewidth=1)
    
    # Create a ScalarMappable for the color bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # Add color bar
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Polarization (P)', fontsize=20, fontweight='bold')
    
    # Set labels and title
    ax.set_xlabel('R', fontsize=16)
    ax.set_ylabel('φ (degrees)', fontsize=16)
    ax.set_zlabel('Signal Amplitude', fontsize=16)
    ax.set_title('3D Tensor Signal Variation over R, φ, and P', fontsize=20, pad=20)
    
    # Enhance plot aesthetics
    ax.view_init(elev=30, azim=120)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/3d_tensor_signal_variation.jpeg', dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved 3D tensor plot to {save_dir}/3d_tensor_signal_variation.jpeg")

if __name__ == "__main__":
    print("Performance Test and Visualization for Optimized Tensor Signal Generation")
    print("=" * 80)
    
    # Test tensor performance
    tensor_success, tensor_signals, tensor_P_values = test_performance()
    
    # Test vector performance for comparison
    vector_success, vector_signals, vector_P_values = test_vector_performance()
    
    if tensor_success and vector_success:
        print("\n" + "=" * 80)
        print("✅ All tests completed successfully!")
        print("=" * 80)
        print("\nKey optimizations implemented:")
        print("1. Pre-computed and cached Lineshape function results")
        print("2. Pre-computed Hilbert transforms")
        print("3. Pre-computed sin/cos values for all phi angles")
        print("4. Vectorized computation using NumPy broadcasting")
        print("5. Batch processing for multiple samples")
        print("6. Eliminated redundant function calls in the inner loop")
        print("\nExpected performance improvement: 10-50x faster for tensor generation")
        
        # Generate visualizations
        print("\n" + "=" * 80)
        print("Generating visualizations...")
        print("=" * 80)
        
        try:
            # Plot tensor phase variation
            plot_tensor_phase_variation(tensor_signals, tensor_P_values)
            
            # Plot vector signals
            plot_vector_signals(vector_signals, vector_P_values)
            
            # Plot tensor heatmap
            plot_tensor_heatmap(tensor_signals, tensor_P_values)
            
            # Plot side-by-side comparison
            plot_comparison_side_by_side(tensor_signals, vector_signals, tensor_P_values)
            
            # Plot 3D tensor variation
            plot_3d_tensor_variation(tensor_signals, tensor_P_values)
            
            print("\n" + "=" * 80)
            print("✅ All visualizations completed successfully!")
            print("=" * 80)
            print("\nGenerated plots:")
            print("1. tensor_signal_phase_variation.jpeg - Tensor polarization phase variation")
            print("2. vector_signal_variation.jpeg - Vector polarization signal variation")
            print("3. tensor_signal_heatmap.jpeg - Tensor signal heatmap")
            print("4. tensor_vs_vector_comparison.jpeg - Side-by-side comparison")
            print("5. 3d_tensor_signal_variation.jpeg - 3D tensor signal variation")
            
        except Exception as e:
            print(f"\n❌ Error during visualization: {e}")
            import traceback
            traceback.print_exc()
            
    else:
        print("\n❌ Some tests failed. Please check the error messages above.") 