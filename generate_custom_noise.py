mport numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import welch


def generate_custom_noise(sample_rate=44100, duration=3, alpha=1,
output_file='custom_noise.wav'):
    """
    Generates a noise signal with a custom power spectral density (PSD) of 1/f^alpha.

    Parameters:
    - sample_rate (int): Sampling frequency in Hz (default: 44100)
    - duration (float): Duration of the signal in seconds (default: 3)
    - alpha (float): Exponent for the power spectral density (default: 1 for pink noise)
    - output_file (str): Name of the output WAV file (default: 'custom_noise.wav')
    """

    # Generate Gaussian white noise
    np.random.seed(42)  # For reproducibility
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
    wavfile.write(output_file, sample_rate, modified_signal_int16)

    print(f"Generated {output_file} with alpha={alpha}.")


    # Compute and plot the power spectral density (PSD)
    # log-log plot; power density falls off at 10 dB/decade of frequency
    # (−3.01 dB/octave)

    #title_str = f"Power Spectral Density of 1/f**{alpha}"
    #frequencies, psd = welch(modified_signal_int16, fs=sample_rate, nperseg=1024)
    #plt.figure(figsize=(12, 6))
    #plt.loglog(frequencies, psd)
    #plt.title(f"Power Spectral Density of 1/f**{alpha}")

    #plt.xlabel("Frequency (Hz)")
    #plt.ylabel("PSD (dB/Hz)")
    #plt.grid(True)
    #plt.show()


#generate_custom_noise(duration=10,alpha=0,output_file='white_noise.wav') # white
#generate_custom_noise(duration=10,alpha=1,output_file='pink_noise.wav') # pink
#generate_custom_noise(duration=10,alpha=2,output_file='brown_noise.wav') # brown
#generate_custom_noise(duration=10,alpha=-1,output_file='blue_noise.wav') # blue
#generate_custom_noise(duration=10,alpha=-2,output_file='purple_noise.wav') # purple

