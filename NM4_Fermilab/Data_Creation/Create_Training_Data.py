#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import pandas as pd
import logging
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
                 num_samples=1000,
                 add_noise=0,
                 noise_level=0.000002,
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
                 bound=0.08):
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
        
        # Common baseline parameters
        self.U = 2.4283
        self.eta = 1.04e-2
        # self.phi = 6.1319
        self.Cstray = 10**(-20)
        self.shift = 0
        self.shifting = shifting
        self.bound = bound
        
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
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("SignalGenerator")
    
    def _generate_proton_signal(self, x):
        """Generate a proton signal using Voigt profile.
        This is still being worked on..."""

        # Work on tensor polarization in future


        sig = 0.1 + np.random.uniform(-0.009, 0.001)       
        gam = 0.1 + np.random.uniform(-0.009, 0.001)         
        amp = 0.005 + np.random.uniform(-0.005, 0.01)
        center = 213 + np.random.uniform(-0.1, 0.1)
        
        return Voigt(x, amp, sig, gam, center), None
    

    def _generate_deuteron_signal(self, P):
        X = np.linspace(30.88, 34.48, 500)
        
        if self.shifting:
            """Generate a deuteron signal using the Sampling_Lineshape function. Here, we are shifting the signal by a random amount within the bound to 
            capture more information about the lineshape"""
            if self.polarization_type == "vector":
                # Call the function directly
                signal = SamplingVectorLineshape(P, X, self.bound)
            elif self.polarization_type == "tensor":
                signal = SamplingTensorLineshape(P, X, self.bound)
            else:
                raise ValueError(f"Invalid polarization type: {self.polarization_type}. Choose 'vector' or 'tensor'.")
            
            signal = signal / 1500.0  # Scale here needs to be worked on
            return signal
        else:
            """Generate a deuteron signal using the GenerateLineshape function."""
            if self.polarization_type == "vector":
                # Call the function and unpack the returned tuple
                result = GenerateVectorLineshape(P, X)
            elif self.polarization_type == "tensor":
                result = GenerateTensorLineshape(P, X, self.phi)
            else:
                raise ValueError(f"Invalid polarization type: {self.polarization_type}. Choose 'vector' or 'tensor'.")
            
            # Now unpack the result tuple
            signal, _, _ = result
            signal /= 1500.0  # Scaling it down here
            return signal
    
    def _add_baseline_and_noise(self, signal, x):
        """Add baseline and optional noise to the signal."""
        if self.baseline:
            baseline = Baseline(x, self.U, self.Cknob, self.eta, self.cable, 
                               self.Cstray, self.phi, self.shift)
            combined_signal = signal + baseline
        else:
            combined_signal = signal
        
        if self.add_noise == 1:
            noise = np.random.normal(0, self.noise_level, size=x.shape)
            return combined_signal + noise, combined_signal, noise
        else:
            return combined_signal, combined_signal, None
    
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
        phi_arr = []
        
        # Get frequency range based on center frequency
   
        x = np.linspace(30.88,34.48,500)
        
        self.logger.info(f"Generating {self.num_samples} samples in {self.mode} mode...")
        
        def sample_exponential_with_cutoff(scale, p_min, p_max, size):
            samples = []
            while len(samples) < size:
                # Sample from exponential and shift by p_min
                new_samples = p_min + np.random.exponential(scale=scale, size=size)
                # Keep only values <= p_max
                filtered = new_samples[new_samples <= p_max]
                samples.extend(filtered.tolist())
            return np.array(samples[:size])
        
        if self.oversampling:
            self.logger.info(f"Oversampling around {self.oversampled_value} between bounds: "
                             f"Lower bound: {self.oversampling_lower_bound}, Upper bound: {self.oversampling_upper_bound}")
            
            oversample_P = np.random.uniform(self.oversampling_lower_bound, self.oversampling_upper_bound, self.num_samples)
            
            P_min = self.oversampling_upper_bound

            # Sample from exponential and truncate at p_max
            P_exp = sample_exponential_with_cutoff(scale=self.alpha, p_min=P_min, p_max=self.p_max, size=self.num_samples)
            
            P_values = np.concatenate([oversample_P, P_exp])
        else:
            self.logger.info(f"Uniformly creating data between {self.lower_bound} and {self.upper_bound}")
            # Generate P's uniformly between [lower_bound, upper_bound]
            P_values = np.random.uniform(self.lower_bound, self.upper_bound, self.num_samples)
        
        for Ps in P_values:
            self.phi = np.random.uniform(0, 360) ### Randomly sample a phase angle for the tensor polarization
            phi_arr.append(self.phi)
            if self.mode == "deuteron":
                signal = self._generate_deuteron_signal(Ps)
            else:  # proton mode
                signal, _ = self._generate_proton_signal(x)

            
                
            # Add baseline and noise
            noisy_signal, clean_signal, noise = self._add_baseline_and_noise(signal, x)
            
            # Calculate SNR if noise is added
            if self.add_noise == 1 and noise is not None and np.max(np.abs(noise)) > 0:
                snr = np.max(np.abs(clean_signal)) / np.max(np.abs(noise))
                snr_arr.append(snr)
            else:
                snr_arr.append(None)
                
            signal_arr.append(noisy_signal)
        
        # Create dataframe
        df = pd.DataFrame(signal_arr)
        
        if len(P_values) > 0:  
            df['P'] = P_values
        if len(snr_arr) > 0:  
            df['SNR'] = snr_arr
            
        # Determine filename
        filename = 'Sample'
        if job_id is not None:
            filename += f"_{job_id}"
        filename += ".parquet"
        
        file_path = os.path.join(self.output_dir, filename)
        
        # Saving Dataframe as Parquet
        try:
            df.to_parquet(file_path, engine='pyarrow', compression='snappy')
            self.logger.info(f"Parquet file saved successfully to {file_path}")
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
    parser.add_argument('--lower_bound', type=float, default=0.1, 
                        help='Lower bound of P value (not oversampled)')
    parser.add_argument('--p_max', type=float, default=0.6, 
                        help='Maximum polarization value')
    parser.add_argument('--alpha', type=float, default=2.0, 
                        help='Decay rate for power law distribution')
    parser.add_argument('--baseline', type=bool, default=True, 
                        help='Whether to add a baseline')
    parser.add_argument('--noise_level', type=float, default=0.000002, 
                        help='Standard deviation of Gaussian noise')
    parser.add_argument('--output_dir', default='Training_Data', 
                        help='Directory to save output Parquet files')
    parser.add_argument('--bound', type=float, default=0.08, 
                        help='Bound of the shift')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()    
    
    generator = SignalGenerator(
    mode=args.mode,
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
    baseline=args.baseline,
    shifting=args.shifting,
    bound=args.bound
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
            print(f"TensorFlow imported: {'tf' in globals()}")
            if 'tf' in globals():
                print(f"TensorFlow version: {tf.__version__}")
            
            # Check input parameters
            print(f"\nInput parameters:")
            print(f"  Mode: {args.mode}")
            print(f"  Num samples: {args.num_samples}")
            print(f"  Add noise: {args.add_noise}")
            print(f"  Oversampling: {args.oversampling}")
            
            # Check if frequency range is properly defined
            print(f"\nFrequency configuration:")
            if hasattr(generator, 'center_freq'):
                print(f"  Center frequency: {generator.center_freq} MHz")
            
            # Check for file system issues
            print(f"\nOutput directory:")
            print(f"  Path: {args.output_dir}")
            print(f"  Exists: {os.path.exists(args.output_dir)}")
            print(f"  Writable: {os.access(args.output_dir, os.W_OK) if os.path.exists(args.output_dir) else 'N/A'}")
            
            # import psutil   
            # process = psutil.Process(os.getpid())
            # print(f"\nMemory usage: {process.memory_info().rss / (1024 * 1024):.2f} MB")
            
        except Exception as diag_error:
            print(f"Error during diagnostics: {diag_error}")
        
        print("-" * 60)
        sys.exit(1)