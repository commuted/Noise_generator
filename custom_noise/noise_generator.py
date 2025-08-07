# custom_noise/noise_generator.py
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import welch

def generate_custom_noise(sample_rate=44100, duration=3, alpha=1, output_file=None, plot_psd=False, random_seed=42):
    """
    Generates a noise signal with a custom power spectral density (PSD) of 
    1/f^alpha.

    Parameters:
    - sample_rate (int): Sampling frequency in Hz (default: 44100)
    - duration (float): Duration of the signal in seconds (default: 3)
    - alpha (float): Exponent for the power spectral density (default: 1 for pink noise)
    - output_file (str): Name of the output WAV file (default: 'custom_noise.wav')
    - plot_psd (bool): If True, plots the power spectral density (default: False)
    - random (int): Seed for reproducibility of random noise generation

    Returns:
    - numpy.ndarray: The generated noise signal as a 16-bit integer array.

    The function generates Gaussian white noise, applies a frequency-dependent 
    scaling to achieve the desired PSD, converts the result to a 16-bit 
    integer format  If wave_file argument is given saves it as a WAV file. 
    If `plot_psd` is True, it also plots and displays the power spectral density 
    of the generated signal.

    Example usage:
    ```python
    noise_signal = generate_custom_noise(alpha=0.5, plot_psd=True)
    ```
    This will generate pink noise with an exponent of 0.5 and display its PSD.
    """

    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive")
    if duration <= 0:
        raise ValueError("Duration must be positive")

    # Generate Gaussian white noise
    np.random.seed(random_seed)  # For reproducibility
    num_samples = int(sample_rate * duration)
    noise = np.random.normal(size=num_samples)

    # Compute Fourier Transform
    X = np.fft.fft(noise)

    # Compute frequencies
    freqs = np.fft.fftfreq(len(X), 1 / sample_rate)

    # Compute scaling factors for the frequency domain
    scaling_factors = np.ones_like(freqs)
    non_zero_mask = freqs != 0
    scaling_factors[non_zero_mask] = 1.0 / (np.abs(freqs[non_zero_mask]) ** (alpha / 2))

    # Apply scaling
    X_scaled = X * scaling_factors

    # Compute inverse Fourier Transform
    modified_signal = np.fft.ifft(X_scaled).real

    # Normalize the signal to prevent clipping
    max_amplitude = np.max(np.abs(modified_signal))
    if max_amplitude != 0:
        modified_signal /= max_amplitude

    # Convert to 16-bit integers
    modified_signal_int16 = (modified_signal * 32767).astype(np.int16)

    # Save as WAV file
    if output_file is not None:
        wavfile.write(output_file, sample_rate, modified_signal_int16)
        print(f"Generated {output_file} with alpha={alpha}.")

    # Optionally plot the power spectral density
    if plot_psd:
        frequencies, psd = welch(modified_signal_int16, fs=sample_rate, nperseg=1024)
        plt.figure(figsize=(12, 6))
        plt.loglog(frequencies, psd)
        plt.title(f"Power Spectral Density of 1/f^{alpha}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD (dB/Hz)")
        plt.grid(True)
        plt.show()

    return modified_signal_int16
