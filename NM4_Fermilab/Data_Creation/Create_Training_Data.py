#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import pandas as pd
import logging
from scipy.signal import hilbert
from Lineshape import *

class SignalGenerator:
    """
    A class for generating signal training data for deuteron or proton experiments.
    
    This class generates synthetic training data that combines baseline signals with either 
    deuteron or proton signals, adds optional noise, and saves the results to Parquet files.
    It is designed to be called from a SLURM batch job for efficient generation of large datasets.
    """
    
    def __init__(self, 
                 mode="deuteron",
                 polarization_type="vector",
                 output_dir="Training_Data", 
                 num_samples=10,
                 add_noise=0,
                 noise_level=0.02,
                 oversampling=1,
                 oversampled_value=0.0005,
                 oversampling_upper_bound=0.0006,
                 oversampling_lower_bound=0.0004,
                 upper_bound=0.6,
                 lower_bound=0.1,
                 p_max=0.6,
                 alpha=2.0,
                 baseline=1,
                 shifting=0,
                 bound=0.08,
                 scale_factor=1.0):
        """
        Generate a lineshape (ND3 or NH3) with configuration parameters.
        
        Parameters:
        -----------
        mode : str
            Experiment type: "deuteron" or "proton"
        polarization_type : str
            "vector" or "tensor"
        output_dir : str
            Directory to save output Parquet files
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
        shifting : bool
            Whether or not to shift the signal
        bound : float
            Bound of the shift
        scale_factor : float
            Factor by which to scale the signal
        """
        self.mode = mode.lower()
        self.polarization_type = polarization_type.lower()
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
        self.scale_factor = scale_factor
        
        # Common baseline parameters
        self.U = 2.4283
        self.eta = 1.04e-2
        self.Cstray = 10**(-20)
        self.shift = 0
        self.shifting = shifting
        self.bound = bound
        
        # Define phi values for tensor polarization (500 phase angles from 0 to 360 degrees)
        if self.polarization_type == "tensor":
            self.phi_values = np.linspace(0, 180, 500)
        else:
            self.phi = 6.1319
        
        # Mode-specific default parameters
        if self.mode == "deuteron":
            self.Cknob = 0.2299
            self.cable = 6/2
            self.center_freq = 32.32
        elif self.mode == "proton":
            self.Cknob = 0.0647
            self.cable = 22/2
            self.center_freq = 213
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose 'deuteron' or 'proton'.")
        
        # System parameters for tensor polarization (from your second script)
        self.g = 0.05
        self.s = 0.04
        self.bigy = np.sqrt(3 - self.s)
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("SignalGenerator")
    
    def _tensor_signal_functions(self, x, eps):
        """Helper functions for tensor signal generation (from your second script)"""
        def cosal(x, eps):
            return (1 - eps * x - self.s) / self._bigxsquare(x, eps)

        def _bigxsquare(x, eps):
            return np.sqrt(self.g**2 + (1 - eps * x - self.s)**2)

        def mult_term(x, eps):
            return 1 / (2 * np.pi * np.sqrt(_bigxsquare(x, eps)))

        def cosaltwo(x, eps):
            return np.sqrt((1 + cosal(x, eps)) / 2)

        def sinaltwo(x, eps):
            return np.sqrt((1 - cosal(x, eps)) / 2)

        def termone(x, eps):
            return np.pi / 2 + np.arctan((self.bigy**2 - _bigxsquare(x, eps)) / 
                                       (2 * self.bigy * np.sqrt(_bigxsquare(x, eps)) * sinaltwo(x, eps)))

        def termtwo(x, eps):
            return np.log((self.bigy**2 + _bigxsquare(x, eps) + 2 * self.bigy * np.sqrt(_bigxsquare(x, eps)) * cosaltwo(x, eps)) /
                         (self.bigy**2 + _bigxsquare(x, eps) - 2 * self.bigy * np.sqrt(_bigxsquare(x, eps)) * cosaltwo(x, eps)))

        def icurve(x, eps):
            return mult_term(x, eps) * (2 * cosaltwo(x, eps) * termone(x, eps) + sinaltwo(x, eps) * termtwo(x, eps))
        
        return icurve(x, eps)
    
    def _bigxsquare(self, x, eps):
        """Helper function for tensor calculations"""
        return np.sqrt(self.g**2 + (1 - eps * x - self.s)**2)

    def _generate_proton_signal(self, x):
        """Generate a proton signal using Voigt profile.
        This is still being worked on..."""
        sig = 0.1 + np.random.uniform(-0.009, 0.001)       
        gam = 0.1 + np.random.uniform(-0.009, 0.001)         
        amp = 0.005 + np.random.uniform(-0.005, 0.01)
        center = 213 + np.random.uniform(-0.1, 0.1)
        
        return Voigt(x, amp, sig, gam, center), None

    def _generate_deuteron_signal(self, P):
        """Generate deuteron signal for both vector and tensor polarizations"""
        X = np.linspace(-3, 3, 500)  
        
        if self.polarization_type == "tensor":

            signals = np.zeros((500, 500))
            
            # Generate signal for each phase angle
            for i, phi_deg in enumerate(self.phi_values):
                phi_rad = np.deg2rad(phi_deg)

                if self.shifting:
                    signal= SamplingTensorLineshape(P, X, self.bound)
                else:
                    signal, _, _ = GenerateTensorLineshape(X, P, phi_deg)
                
                if self.baseline:
                    baseline = Baseline(X, self.U, self.Cknob, self.eta, self.cable, 
                                      self.Cstray, phi_deg, self.shift)
                    total_signal = (signal) / self.scale_factor + baseline
                else:
                    total_signal = (signal) / self.scale_factor
                
                signals[:, i] = total_signal
            
            return signals
            
        else:  # vector polarization
            if self.shifting:
                signal = SamplingVectorLineshape(P, X, self.bound) / self.scale_factor
            else:
                result = GenerateVectorLineshape(P, X)
                signal, _, _ = result
                signal = signal / self.scale_factor
            
            # Add baseline if required for vector polarization
            if self.baseline:
                baseline = Baseline(X, self.U, self.Cknob, self.eta, self.cable, 
                                  self.Cstray, self.phi, self.shift)
                signal = (signal) / self.scale_factor + baseline
            
            return signal
    
    def _add_noise_to_signal(self, signal):
        """Add noise to the signal if required"""
        if self.add_noise == 1:
            noise = np.random.normal(0, self.noise_level, size=signal.shape)
            return signal + noise, noise
        else:
            return signal, None
    
    def generate_samples(self, job_id=None):
        """
        Generate signal samples and save them to a Parquet file.
        
        Parameters:
        -----------
        job_id : str or int, optional
            Identifier for the current job, used in the output filename
        
        Returns:
        --------
        str
            Path to the saved Parquet file
        """
        signal_arr = []
        snr_arr = []
        
        self.logger.info(f"Generating {self.num_samples} samples in {self.mode} mode with {self.polarization_type} polarization...")
        
        def sample_exponential_with_cutoff(scale, p_min, p_max, size):
            samples = []
            while len(samples) < size:
                new_samples = p_min + np.random.exponential(scale=scale, size=size)
                filtered = new_samples[new_samples <= p_max]
                samples.extend(filtered.tolist())
            return np.array(samples[:size])
        
        if self.oversampling:
            self.logger.info(f"Oversampling around {self.oversampled_value} between bounds: "
                             f"Lower bound: {self.oversampling_lower_bound}, Upper bound: {self.oversampling_upper_bound}")
            
            oversample_P = np.random.uniform(self.oversampling_lower_bound, self.oversampling_upper_bound, self.num_samples)
            P_min = self.oversampling_upper_bound
            P_exp = sample_exponential_with_cutoff(scale=self.alpha, p_min=P_min, p_max=self.p_max, size=self.num_samples)
            P_values = np.concatenate([oversample_P, P_exp])
        else:
            self.logger.info(f"Uniformly creating data between {self.lower_bound} and {self.upper_bound}")
            P_values = np.random.uniform(self.lower_bound, self.upper_bound, self.num_samples)
        
        for i, P in enumerate(P_values):
            if self.mode == "deuteron":
                signal = self._generate_deuteron_signal(P)
            else:  # proton mode
                x = np.linspace(-3, 3, 500)  # Use same range for consistency
                signal, _ = self._generate_proton_signal(x)
            
            # Add noise if required
            noisy_signal, noise = self._add_noise_to_signal(signal)
            
            if self.polarization_type == "tensor":
                # For tensor: signal is (500, 500) -> reshape to (500, 500, 1)
                final_signal = noisy_signal.reshape(500, 500, 1)
                signal_arr.append(final_signal)
                
                # Calculate SNR for tensor signals
                if noise is not None:
                    signal_power = np.mean(signal**2)
                    noise_power = np.mean(noise**2)
                    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
                    snr_arr.append(snr)
                else:
                    snr_arr.append(None)
            else:
                # For vector: signal is (500,) -> keep as is
                signal_arr.append(noisy_signal)
                
                # Calculate SNR for vector signals
                if noise is not None:
                    signal_power = np.mean(signal**2)
                    noise_power = np.mean(noise**2)
                    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
                    snr_arr.append(snr)
                else:
                    snr_arr.append(None)
            
            if (i + 1) % 100 == 0:
                self.logger.info(f"Generated {i + 1}/{len(P_values)} samples")
        
        # Create DataFrame
        if self.polarization_type == "tensor":
            # For tensor, each signal is (500, 500, 1) - flatten for storage
            df = pd.DataFrame({
                'signal': [sig.flatten() for sig in signal_arr],  # Flatten (500,500,1) -> (250000,)
                'P': P_values,
                'SNR': snr_arr
            })
            
            self.logger.info(f"Generated {len(signal_arr)} tensor signals with shape (500, 500, 1)")
            self.logger.info(f"Flattened signals have shape: {len(df['signal'].iloc[0])}")
            
        else:
            # For vector, create DataFrame as before
            df = pd.DataFrame(signal_arr)
            if len(P_values) > 0:  
                df['P'] = P_values
            if len(snr_arr) > 0:  
                df['SNR'] = snr_arr
        
        # Determine filename
        filename = f'Sample_{self.polarization_type}'
        if job_id is not None:
            filename += f"_{job_id}"
        filename += ".parquet"
        
        file_path = os.path.join(self.output_dir, filename)
        
        # Save DataFrame with metadata
        try:
            metadata = {
                'polarization_type': self.polarization_type,
                'mode': self.mode,
                'frequency_bins': 500,
                'phi_bins': 500 if self.polarization_type == "tensor" else 1,
                'signal_shape': (500, 500, 1) if self.polarization_type == "tensor" else (500,),
                'is_flattened': True if self.polarization_type == "tensor" else False,
                'frequency_range': (-3, 3),  # Updated to match your second script
                'phi_range': (0, 180) if self.polarization_type == "tensor" else None,
                'num_samples': len(P_values)
            }
            
            df.attrs['metadata'] = str(metadata)
            df.to_parquet(file_path, engine='pyarrow', compression='snappy')
            
            self.logger.info(f"Parquet file saved successfully to {file_path}")
            self.logger.info(f"Data shape: {df.shape}")
            
            if self.polarization_type == "tensor":
                self.logger.info(f"Each signal represents (500 frequencies x 500 phase angles)")
                self.logger.info(f"To reconstruct: signal.reshape(500, 500)")
            
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error saving Parquet file: {e}")
            raise

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate signal data for training.')
    
    # Required arguments
    parser.add_argument('--job_id', help='Job identifier for the output filename')
    parser.add_argument('--mode', choices=['deuteron', 'proton'], help='Specimen type')
    parser.add_argument('--polarization_type', type=str, choices=['vector', 'tensor'], help='Polarization type (vector or tensor)')
    parser.add_argument('--num_samples', type=int, help='Number of samples to generate')
    
    parser.add_argument('--add_noise', type=int, choices=[0, 1], default=0,
                        help='Set to 1 to add noise to signals, 0 to disable')
    parser.add_argument('--oversampling', type=int, choices=[0, 1], default=0,
                        help='Set to 1 to enable oversampling, 0 to disable')
    parser.add_argument('--shifting', type=int, choices=[0, 1], default=0,
                        help='Set to 1 to enable shifting, 0 to disable')

    parser.add_argument('--oversampled_value', type=float, default=0.0005, 
                        help='Value to oversample around')
    parser.add_argument('--oversampling_upper_bound', type=float, default=0.0006, 
                        help='Upper bound for oversampling range')
    parser.add_argument('--oversampling_lower_bound', type=float, default=0.0004, 
                        help='Lower bound for oversampling range')
    parser.add_argument('--upper_bound', type=float, default=0.6, 
                        help='Upper bound of P value (not oversampled)')
    parser.add_argument('--lower_bound', type=float, default=0.0005, 
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
                        help='Directory to save output Parquet files')
    parser.add_argument('--bound', type=float, default=0.08, 
                        help='Bound of the shift')
    parser.add_argument('--scale_factor', type=float, default=1.0, 
                        help='Factor by which to scale the signal')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()    
    
    generator = SignalGenerator(
        mode=args.mode,
        polarization_type=args.polarization_type,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        add_noise=args.add_noise,
        noise_level=args.noise_level,
        oversampling=args.oversampling,
        oversampled_value=args.oversampled_value,
        oversampling_upper_bound=args.oversampling_upper_bound,
        oversampling_lower_bound=args.oversampling_lower_bound,
        upper_bound=args.upper_bound,
        lower_bound=args.lower_bound,
        p_max=args.p_max,
        alpha=args.alpha,
        baseline=bool(args.baseline),
        shifting=args.shifting,
        bound=args.bound,
        scale_factor=args.scale_factor
    )
        
    try:
        print("Generating signal...")
        
        import time
        start_time = time.time()
        
        generator.generate_samples(args.job_id)
        
        execution_time = time.time() - start_time
        print(f"Signal generation complete in {execution_time:.2f} seconds")
        
    except Exception as e:
        print("\nERROR DURING SIGNAL GENERATION:")
        print("-" * 60)
        
        import traceback
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nTraceback:")
        traceback.print_exc()
        
        print("\nDIAGNOSTIC INFORMATION:")
        try:
            print(f"Input parameters:")
            print(f"  Mode: {args.mode}")
            print(f"  Polarization type: {args.polarization_type}")
            print(f"  Num samples: {args.num_samples}")
            print(f"  Add noise: {args.add_noise}")
            print(f"  Oversampling: {args.oversampling}")
            
            print(f"\nOutput directory:")
            print(f"  Path: {args.output_dir}")
            print(f"  Exists: {os.path.exists(args.output_dir)}")
            print(f"  Writable: {os.access(args.output_dir, os.W_OK) if os.path.exists(args.output_dir) else 'N/A'}")
            
        except Exception as diag_error:
            print(f"Error during diagnostics: {diag_error}")
        
        print("-" * 60)
        sys.exit(1)