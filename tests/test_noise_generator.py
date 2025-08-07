# tests/test_noise_generator.py
import pytest
import numpy as np
from scipy.signal import welch
from custom_noise import generate_custom_noise
import os
import matplotlib.pyplot as plt

# Fixture to clean up files after tests
@pytest.fixture(autouse=True)
def cleanup():
    yield
    for file in ['test_noise.wav', 'custom_noise.wav']:
        if os.path.exists(file):
            os.remove(file)

def test_generate_custom_noise():
    output_file = 'test_noise.wav'
    signal = generate_custom_noise(
        sample_rate=44100,
        duration=1,
        alpha=1,
        output_file=output_file,
        plot_psd=False
    )
    assert os.path.exists(output_file), "Output file was not created."
    assert len(signal) == 44100, "Signal length does not match expected duration."
    assert np.max(np.abs(signal)) <= 32767, "Signal exceeds 16-bit range."

def test_generate_custom_noise_default_output():
    signal = generate_custom_noise(
        sample_rate=44100,
        duration=1,
        alpha=1,
        plot_psd=False
    )
    output_file = 'custom_noise.wav'  # Default file
    assert not os.path.exists(output_file), "Default output file was not created."
    assert len(signal) == 44100, "Signal length does not match expected duration."
    assert np.max(np.abs(signal)) <= 32767, "Signal exceeds 16-bit range."

@pytest.mark.parametrize("alpha,expected_slope", [
    (0, 0),    # White noise
    (1, -1),   # Pink noise
    (2, -2),   # Brown noise
    (-1, 1),   # Blue noise
    (-2, 2),   # Purple noise
])
def test_generate_custom_noise_slope(alpha, expected_slope):
    sample_rate = 44100
    duration = 3
    signal = generate_custom_noise(
        sample_rate=sample_rate,
        duration=duration,
        alpha=alpha,
        plot_psd=False
    )
    expected_length = sample_rate * duration
    assert len(signal) == int(expected_length), f"Signal length {len(signal)} does not match expected {expected_length}"

    # Compute PSD
    freqs, psd = welch(signal, fs=sample_rate, nperseg=1024)
    mask = (freqs > 10) & (freqs < sample_rate / 4)  # Focus on mid-range frequencies
    log_freq = np.log(freqs[mask])
    log_psd = np.log(psd[mask])

    # Fit line to log-log PSD
    A = np.column_stack((log_freq, np.ones_like(log_freq)))
    slope, _ = np.linalg.lstsq(A, log_psd, rcond=None)[0]
    assert np.isclose(slope, expected_slope, atol=0.3), f"Slope {slope} not close to {expected_slope} for alpha={alpha}"

def test_generate_custom_noise_invalid_inputs():
    # Test negative sample_rate
    with pytest.raises(ValueError, match="Sample rate must be positive"):
        generate_custom_noise(sample_rate=-44100, duration=1, alpha=1)

    # Test zero duration
    with pytest.raises(ValueError, match="Duration must be positive"):
        generate_custom_noise(sample_rate=44100, duration=0, alpha=1)

    # Test invalid output file path
    with pytest.raises(OSError):
        generate_custom_noise(sample_rate=44100, duration=1, alpha=1, output_file='/invalid_path/noise.wav')

def test_generate_custom_noise_plot_psd():
    # Test that plotting runs without errors
    plt.figure()  # Create a figure to avoid interfering with other tests
    signal = generate_custom_noise(
        sample_rate=44100,
        duration=1,
        alpha=1,
        plot_psd=True
    )
    assert len(signal) == 44100, "Signal length does not match expected duration."
    plt.close()  # Close the figure

def test_generate_custom_noise_reproducibility():
    signal1 = generate_custom_noise(sample_rate=44100, duration=1, alpha=1, plot_psd=False)
    signal2 = generate_custom_noise(sample_rate=44100, duration=1, alpha=1, plot_psd=False)
    assert np.array_equal(signal1, signal2), "Signals are not reproducible with same seed."
