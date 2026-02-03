# tests/test_noise_generator.py
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.signal import welch

from custom_noise import (
    NOISE_PRESETS,
    blue_noise,
    brown_noise,
    generate_custom_noise,
    pink_noise,
    violet_noise,
    white_noise,
)


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
    signal = generate_custom_noise(
        sample_rate=44100,
        duration=1,
        alpha=1,
        plot_psd=True
    )
    assert len(signal) == 44100, "Signal length does not match expected duration."
    plt.close("all")  # Close all figures

def test_generate_custom_noise_reproducibility():
    signal1 = generate_custom_noise(sample_rate=44100, duration=1, alpha=1, plot_psd=False)
    signal2 = generate_custom_noise(sample_rate=44100, duration=1, alpha=1, plot_psd=False)
    assert np.array_equal(signal1, signal2), "Signals are not reproducible with same seed."


# Tests for new features


@pytest.mark.parametrize("noise_type,expected_alpha", [
    ("white", 0),
    ("pink", 1),
    ("brown", 2),
    ("brownian", 2),
    ("red", 2),
    ("blue", -1),
    ("violet", -2),
    ("purple", -2),
    ("azure", -2),
])
def test_noise_type_presets(noise_type, expected_alpha):
    """Test that noise_type presets map to correct alpha values."""
    assert NOISE_PRESETS[noise_type] == expected_alpha
    # Generate noise and verify it works
    signal = generate_custom_noise(noise_type=noise_type, duration=0.5)
    assert len(signal) == int(44100 * 0.5)


def test_noise_type_case_insensitive():
    """Test that noise_type is case-insensitive."""
    signal1 = generate_custom_noise(noise_type="PINK", duration=0.5, random_seed=123)
    signal2 = generate_custom_noise(noise_type="pink", duration=0.5, random_seed=123)
    assert np.array_equal(signal1, signal2)


def test_noise_type_invalid():
    """Test that invalid noise_type raises ValueError."""
    with pytest.raises(ValueError, match="Unknown noise_type"):
        generate_custom_noise(noise_type="invalid")


def test_alpha_and_noise_type_mutually_exclusive():
    """Test that specifying both alpha and noise_type raises ValueError."""
    with pytest.raises(ValueError, match="Cannot specify both"):
        generate_custom_noise(alpha=1, noise_type="pink")


def test_dtype_int16():
    """Test int16 output (default)."""
    signal = generate_custom_noise(duration=0.5, dtype="int16")
    assert signal.dtype == np.int16
    assert np.max(np.abs(signal)) <= 32767


def test_dtype_float32():
    """Test float32 output."""
    signal = generate_custom_noise(duration=0.5, dtype="float32")
    assert signal.dtype == np.float32
    assert np.max(np.abs(signal)) <= 1.0


def test_dtype_float64():
    """Test float64 output."""
    signal = generate_custom_noise(duration=0.5, dtype="float64")
    assert signal.dtype == np.float64
    assert np.max(np.abs(signal)) <= 1.0


def test_dtype_invalid():
    """Test that invalid dtype raises ValueError."""
    with pytest.raises(ValueError, match="Unknown dtype"):
        generate_custom_noise(dtype="int32")


def test_random_seed_none():
    """Test that random_seed=None produces different results each time."""
    signal1 = generate_custom_noise(duration=0.5, random_seed=None)
    signal2 = generate_custom_noise(duration=0.5, random_seed=None)
    # Very unlikely to be equal with random seeds
    assert not np.array_equal(signal1, signal2)


def test_default_is_pink_noise():
    """Test that default (no alpha or noise_type) produces pink noise."""
    signal_default = generate_custom_noise(duration=0.5, random_seed=42)
    signal_pink = generate_custom_noise(noise_type="pink", duration=0.5, random_seed=42)
    assert np.array_equal(signal_default, signal_pink)


# Tests for convenience functions


def test_white_noise_function():
    """Test white_noise() convenience function."""
    signal = white_noise(duration=0.5, random_seed=42)
    expected = generate_custom_noise(noise_type="white", duration=0.5, random_seed=42)
    assert np.array_equal(signal, expected)


def test_pink_noise_function():
    """Test pink_noise() convenience function."""
    signal = pink_noise(duration=0.5, random_seed=42)
    expected = generate_custom_noise(noise_type="pink", duration=0.5, random_seed=42)
    assert np.array_equal(signal, expected)


def test_brown_noise_function():
    """Test brown_noise() convenience function."""
    signal = brown_noise(duration=0.5, random_seed=42)
    expected = generate_custom_noise(noise_type="brown", duration=0.5, random_seed=42)
    assert np.array_equal(signal, expected)


def test_blue_noise_function():
    """Test blue_noise() convenience function."""
    signal = blue_noise(duration=0.5, random_seed=42)
    expected = generate_custom_noise(noise_type="blue", duration=0.5, random_seed=42)
    assert np.array_equal(signal, expected)


def test_violet_noise_function():
    """Test violet_noise() convenience function."""
    signal = violet_noise(duration=0.5, random_seed=42)
    expected = generate_custom_noise(noise_type="violet", duration=0.5, random_seed=42)
    assert np.array_equal(signal, expected)


def test_convenience_functions_with_dtype():
    """Test that convenience functions support dtype parameter."""
    signal = pink_noise(duration=0.5, dtype="float32")
    assert signal.dtype == np.float32
    assert np.max(np.abs(signal)) <= 1.0


def test_convenience_functions_with_output_file(tmp_path):
    """Test that convenience functions can save to file."""
    output_file = tmp_path / "test.wav"
    pink_noise(duration=0.5, output_file=str(output_file))
    assert output_file.exists()
