#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from Custom_Scripts.Lineshape import *

class SignalGenerator:
    """
    A class for generating signal training data for deuteron or proton experiments.
    
    This class generates synthetic training data that combines baseline signals with either 
    deuteron or proton signals, adds optional noise, and saves the results to CSV files.
    It is designed to be called from a SLURM batch job for efficient generation of large datasets.
    """
    
    def __init__(self, 
                 mode="deuteron",
                 output_dir="Training_Data", 
                 num_samples=1000,
                 add_noise=False,
                 noise_level=0.000002,
                 oversampling=False,
                 oversampled_value=0.0005,
                 oversampling_upper_bound=0.0006,
                 oversampling_lower_bound=0.0004,
                 upper_bound=0.6,
                 lower_bound=0.1,
                 p_max=0.6,
                 alpha=2.0,
                 baseline=True):
        """
        Initialize the SignalGenerator with configuration parameters.
        
        Parameters:
        -----------
        mode : str
            Experiment type: "deuteron" or "proton"
        output_dir : str
            Directory to save output CSV files
        num_samples : int
            Number of signal samples to generate
        add_noise : bool
            Whether to add noise to the signals
        noise_level : float
            Standard deviation of the Gaussian noise
        oversampling : bool
            whether or not to oversample around a certain value (i.e., TE)
        oversampled_value : float
            The value around which to oversample
        oversampling_upper_bound : float
            upper bound of value around which you want to oversample
        oversampling_lower_bound : float
            lower bound of value around which you want to oversample
        upper_bound : float
            Upper bound of P value (not oversampled)
        lower_bound : float
            Lower bound of P value (not oversampled)
        p_max : float
            Maximum polarization value that can be sampled (outside oversampling range)
        alpha : float
            Decay rate for power law distribution that samples P's outside oversampling range
        baseline : bool
            Whether or not to add a baseline to the signal
        """
        self.mode = mode.lower()
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.add_noise = add_noise
        self.noise_level = noise_level
        self.oversampling = oversampling
        self.oversampled_value = oversampled_value
        self.oversampling_upper_bound = oversampling_upper_bound
        self.oversampling_lower_bound = oversampling_lower_bound
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.p_max = p_max
        self.alpha = alpha
        self.baseline = baseline
        
        # Common baseline parameters
        self.U = 2.4283
        self.eta = 1.04e-2
        self.phi = 6.1319
        self.Cstray = 10**(-20)
        self.shift = 0
        
        # Mode-specific default parameters
        if self.mode == "deuteron":
            self.Cknob = 0.1899
            self.cable = 6/2
            self.center_freq = 32.32
        elif self.mode == "proton":
            self.Cknob = 0.0647
            self.cable = 22/2
            self.center_freq = 213
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose 'deuteron' or 'proton'.")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("SignalGenerator")
    
    def _generate_proton_signal(self, x):
        """Generate a proton signal using Voigt profile."""
        sig = 0.1 + np.random.uniform(-0.009, 0.001)       
        gam = 0.1 + np.random.uniform(-0.009, 0.001)         
        amp = 0.005 + np.random.uniform(-0.005, 0.01)
        center = 213 + np.random.uniform(-0.1, 0.1)
        
        return Voigt(x, amp, sig, gam, center), None  # Return signal and None for P
    
    def _generate_deuteron_signal(self, P):
        """Generate a deuteron signal using the GenerateLineshape function."""
        X = np.linspace(-3, 3, 500)
        
        signal, _, _ = GenerateLineshape(P, X)      
        signal /= 1500.0 ## Scaling it down here
        return signal
    
    def _add_baseline_and_noise(self, signal, x):
        """Add baseline and optional noise to the signal."""
        if self.baseline:
            baseline = Baseline_Freq_Bin(x, self.U, self.Cknob, self.eta, self.cable, 
                               self.Cstray, self.phi, self.shift)
            combined_signal = signal + baseline
        else:
            combined_signal = signal
        
        if self.add_noise:
            noise = np.random.normal(0, self.noise_level, size=x.shape)
            return combined_signal + noise, combined_signal, noise
        else:
            return combined_signal, combined_signal, None
    
    def generate_samples(self, job_id=None):
        """
        Generate signal samples and save them to a CSV file.
        
        Parameters:
        -----------
        job_id : str or int, optional
            Identifier for the current job, used in the output filename
        
        Returns:
        --------
        str
            Path to the saved CSV file
        """
        signal_arr = []
        snr_arr = []
        
        # Get frequency range based on center frequency
        x, freq_lower_bound, freq_upper_bound = FrequencyBound(self.center_freq)
        
        self.logger.info(f"Generating {self.num_samples} samples in {self.mode} mode...")
        
        if self.oversampling:
            self.logger.info(f"Oversampling around {self.oversampled_value} between bounds: "
                            f"Lower bound: {self.oversampling_lower_bound}, Upper bound: {self.oversampling_upper_bound}")
            
            # Generate oversampled values
            oversample_P = np.random.uniform(self.oversampling_lower_bound, self.oversampling_upper_bound, self.num_samples)
            
            # Generate distribution of P's outside of oversampling range using power law
            P_min = self.oversampling_upper_bound
            u = np.random.uniform(0, 1, self.num_samples)
            P_power = P_min + (self.p_max - P_min) * (1 - u)**(1.0/self.alpha)
            
            # Combine both distributions
            P_values = np.concatenate([oversample_P, P_power])
        else:
            self.logger.info(f"Uniformly creating data between {self.lower_bound} and {self.upper_bound}")
            # Generate P's uniformly between [lower_bound, upper_bound]
            P_values = np.random.uniform(self.lower_bound, self.upper_bound, self.num_samples)
        
        for Ps in tqdm(P_values, desc="Generating sample data"):
            # Generate signal based on mode
            if self.mode == "deuteron":
                signal = self._generate_deuteron_signal(Ps)
            else:  # proton mode
                signal = self._generate_proton_signal(x)
                
            # Add baseline and noise
            noisy_signal, clean_signal, noise = self._add_baseline_and_noise(signal, x)
            
            # Calculate SNR if noise is added
            if self.add_noise and noise is not None and np.max(np.abs(noise)) > 0:
                snr = np.max(np.abs(clean_signal)) / np.max(np.abs(noise))
                snr_arr.append(snr)
            else:
                snr_arr.append(None)
                
            signal_arr.append(noisy_signal)
        
        # Create dataframe
        df = pd.DataFrame(signal_arr)
        
        # Add metadata columns if available
        if len(P_values) > 0:  # Check if P_values is not empty
            df['P'] = P_values
        if len(snr_arr) > 0:  # Check if snr_arr is not empty
            df['SNR'] = snr_arr
            
        # Determine filename
        filename = 'Sample'
        if job_id is not None:
            filename += f"_{job_id}"
        filename += ".csv"
        
        file_path = os.path.join(self.output_dir, filename)
        
        # Save dataframe
        try:
            df.to_csv(file_path, index=False)
            self.logger.info(f"CSV file saved successfully to {file_path}")
            return file_path
        except Exception as e:
            self.logger.error(f"Error saving CSV file: {e}")
            raise

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate signal data for training.')
    
    # Required arguments
    parser.add_argument('job_id', help='Job identifier for the output filename')
    parser.add_argument('mode', choices=['deuteron', 'proton'], help='Experiment type')
    parser.add_argument('num_samples', type=int, help='Number of samples to generate')
    parser.add_argument('add_noise', type=int, choices=[0, 1], help='Whether to add noise (0=False, 1=True)')
    
    # Optional arguments with defaults
    parser.add_argument('--oversampling', type=int, choices=[0, 1], default=0, 
                        help='Whether to oversample around a value (0=False, 1=True)')
    parser.add_argument('--oversampled_value', type=float, default=0.0005, 
                        help='Value to oversample around')
    parser.add_argument('--oversampling_upper_bound', type=float, default=0.0006, 
                        help='Upper bound for oversampling range')
    parser.add_argument('--oversampling_lower_bound', type=float, default=0.0004, 
                        help='Lower bound for oversampling range')
    parser.add_argument('--upper_bound', type=float, default=0.6, 
                        help='Upper bound of P value (not oversampled)')
    parser.add_argument('--lower_bound', type=float, default=0.1, 
                        help='Lower bound of P value (not oversampled)')
    parser.add_argument('--p_max', type=float, default=0.6, 
                        help='Maximum polarization value')
    parser.add_argument('--alpha', type=float, default=2.0, 
                        help='Decay rate for power law distribution')
    parser.add_argument('--baseline', type=int, choices=[0, 1], default=1, 
                        help='Whether to add a baseline (0=False, 1=True)')
    parser.add_argument('--noise_level', type=float, default=0.000002, 
                        help='Standard deviation of Gaussian noise')
    parser.add_argument('--output_dir', default='Training_Data', 
                        help='Directory to save output CSV files')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Create SignalGenerator with parsed arguments
    generator = SignalGenerator(
        mode=args.mode,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        add_noise=bool(args.add_noise),
        noise_level=args.noise_level,
        oversampling=bool(args.oversampling),
        oversampled_value=args.oversampled_value,
        oversampling_upper_bound=args.oversampling_upper_bound,
        oversampling_lower_bound=args.oversampling_lower_bound,
        upper_bound=args.upper_bound,
        lower_bound=args.lower_bound,
        p_max=args.p_max,
        alpha=args.alpha,
        baseline=bool(args.baseline)
    )
    
    # Generate samples
    generator.generate_samples(args.job_id)
    