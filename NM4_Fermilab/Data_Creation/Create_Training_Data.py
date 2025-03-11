import numpy as np
import pandas as pd
from scipy.integrate import quad
from tqdm import tqdm
import os
import sys
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
                 num_samples=10,
                 add_noise=False,
                 noise_level=0.000002):
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
        """
        self.mode = mode.lower()
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.add_noise = add_noise
        self.noise_level = noise_level
        
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
    
    def _generate_deuteron_signal(self, x):
        """Generate a deuteron signal using the GenerateLineshape function."""
        X = np.linspace(-3, 3, 500)
        
        ### Define the range of polarization you want to use right here ###
        P = np.random.uniform(0.01, 0.7)
        
        signal = GenerateLineshape(P, X) / 1500.0
        return signal, P
    
    def _add_baseline_and_noise(self, signal, x):
        """Add baseline and optional noise to the signal."""
        baseline = Baseline(x, self.U, self.Cknob, self.eta, self.cable, 
                           self.Cstray, self.phi, self.shift)
        
        combined_signal = signal + baseline
        
        if self.add_noise:
            noise = np.random.normal(0, self.noise_level, size=x.shape)
            return combined_signal + noise, combined_signal, noise
        else:
            return combined_signal, combined_signal, 0
    
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
        p_arr = []
        snr_arr = []
        
        # Get frequency range based on center frequency
        x, lower_bound, upper_bound = FrequencyBound(self.center_freq)
        
        self.logger.info(f"Generating {self.num_samples} samples in {self.mode} mode")
        
        for i in tqdm(range(self.num_samples), desc="Generating signals"):
            # Generate signal based on mode
            if self.mode == "deuteron":
                signal, P = self._generate_deuteron_signal(x)
            else:  # proton mode
                signal, P = self._generate_proton_signal(x)
                
            # Add baseline and noise
            noisy_signal, clean_signal, noise = self._add_baseline_and_noise(signal, x)
            
            # Calculate SNR if noise is added
            if self.add_noise and np.max(np.abs(noise)) > 0:
                snr = np.max(np.abs(clean_signal)) / np.max(np.abs(noise))
                snr_arr.append(snr)
                
            signal_arr.append(noisy_signal)
            if P is not None:
                p_arr.append(P)
        
        # Create dataframe
        df = pd.DataFrame(signal_arr)
        
        # Add metadata columns if available
        if p_arr:
            df['P'] = p_arr
        if snr_arr:
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
    
    @classmethod
    def from_command_line(cls):
        """
        Create a SignalGenerator instance from command line arguments.
        
        Command line arguments:
        -----------------------
        1: job_id (required) - Job identifier for the output filename
        2: mode (optional) - "deuteron" or "proton", defaults to "deuteron"
        3: num_samples (optional) - Number of samples to generate, defaults to 10
        4: add_noise (optional) - Whether to add noise (1=True, 0=False), defaults to 0
        
        Returns:
        --------
        tuple
            A tuple containing (SignalGenerator instance, job_id)
        """
        if len(sys.argv) < 2:
            print("Error: Job ID is required as the first argument")
            sys.exit(1)
        
        job_id = sys.argv[1]
        
        # Parse optional arguments
        mode = sys.argv[2] if len(sys.argv) > 2 else "deuteron"
        num_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        add_noise = bool(int(sys.argv[4])) if len(sys.argv) > 4 else False
        
        generator = cls(
            mode=mode,
            num_samples=num_samples,
            add_noise=add_noise
        )
        
        return generator, job_id


if __name__ == "__main__":
    # When run directly, generates data and indexes based on job_id
    generator, job_id = SignalGenerator.from_command_line()
    generator.generate_samples(job_id)
    
    
##### Running this script #####

'''
./jobscript <num_jobs> <proton/deuteron specimen (optional)>
'''