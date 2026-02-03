# custom_noise/noise_generator.py
from __future__ import annotations

from typing import Any, Literal, overload

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.io import wavfile
from scipy.signal import welch

# Named presets mapping to alpha values
NOISE_PRESETS: dict[str, float] = {
    "white": 0,
    "pink": 1,
    "brown": 2,
    "brownian": 2,
    "red": 2,
    "blue": -1,
    "azure": -2,
    "violet": -2,
    "purple": -2,
}


@overload
def generate_custom_noise(
    sample_rate: int = ...,
    duration: float = ...,
    alpha: float | None = ...,
    noise_type: str | None = ...,
    output_file: str | None = ...,
    plot_psd: bool = ...,
    random_seed: int | None = ...,
    dtype: Literal["int16"] = ...,
) -> npt.NDArray[np.int16]: ...


@overload
def generate_custom_noise(
    sample_rate: int = ...,
    duration: float = ...,
    alpha: float | None = ...,
    noise_type: str | None = ...,
    output_file: str | None = ...,
    plot_psd: bool = ...,
    random_seed: int | None = ...,
    dtype: Literal["float32"] = ...,
) -> npt.NDArray[np.float32]: ...


@overload
def generate_custom_noise(
    sample_rate: int = ...,
    duration: float = ...,
    alpha: float | None = ...,
    noise_type: str | None = ...,
    output_file: str | None = ...,
    plot_psd: bool = ...,
    random_seed: int | None = ...,
    dtype: Literal["float64"] = ...,
) -> npt.NDArray[np.float64]: ...


def generate_custom_noise(
    sample_rate: int = 44100,
    duration: float = 3,
    alpha: float | None = None,
    noise_type: str | None = None,
    output_file: str | None = None,
    plot_psd: bool = False,
    random_seed: int | None = 42,
    dtype: Literal["int16", "float32", "float64"] = "int16",
) -> npt.NDArray[np.int16] | npt.NDArray[np.float32] | npt.NDArray[np.float64]:
    """
    Generates a noise signal with a custom power spectral density (PSD) of
    1/f^alpha.

    Parameters:
    - sample_rate (int): Sampling frequency in Hz (default: 44100)
    - duration (float): Duration of the signal in seconds (default: 3)
    - alpha (float): Exponent for the power spectral density. Mutually exclusive
      with noise_type. If neither is specified, defaults to 1 (pink noise).
    - noise_type (str): Named preset for noise type. Options: "white" (alpha=0),
      "pink" (alpha=1), "brown"/"brownian"/"red" (alpha=2), "blue" (alpha=-1),
      "violet"/"purple"/"azure" (alpha=-2). Mutually exclusive with alpha.
    - output_file (str): Path to save as WAV file (optional)
    - plot_psd (bool): If True, plots the power spectral density (default: False)
    - random_seed (int | None): Seed for reproducibility. Use None for random
      results each time (default: 42)
    - dtype (str): Output data type - "int16" for 16-bit integers (default),
      "float32" or "float64" for normalized floating point [-1.0, 1.0]

    Returns:
    - numpy.ndarray: The generated noise signal in the specified dtype.

    The function generates Gaussian white noise, applies a frequency-dependent
    scaling to achieve the desired PSD, and normalizes the result.
    If output_file is given, saves it as a WAV file.
    If `plot_psd` is True, it also plots and displays the power spectral density
    of the generated signal.

    Example usage:
    ```python
    # Using alpha directly
    noise_signal = generate_custom_noise(alpha=1, plot_psd=True)

    # Using named preset
    pink_noise = generate_custom_noise(noise_type="pink", duration=5)

    # Get normalized float output
    float_noise = generate_custom_noise(noise_type="brown", dtype="float32")
    ```
    """
    # Resolve alpha from noise_type or default
    if alpha is not None and noise_type is not None:
        raise ValueError("Cannot specify both 'alpha' and 'noise_type'")

    if noise_type is not None:
        noise_type_lower = noise_type.lower()
        if noise_type_lower not in NOISE_PRESETS:
            valid_types = ", ".join(sorted(NOISE_PRESETS.keys()))
            raise ValueError(f"Unknown noise_type '{noise_type}'. Valid options: {valid_types}")
        resolved_alpha = NOISE_PRESETS[noise_type_lower]
    elif alpha is not None:
        resolved_alpha = alpha
    else:
        resolved_alpha = 1  # Default to pink noise

    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive")
    if duration <= 0:
        raise ValueError("Duration must be positive")

    # Generate Gaussian white noise using modern RNG
    rng = np.random.default_rng(random_seed)
    num_samples = int(sample_rate * duration)
    noise = rng.standard_normal(num_samples)

    # Compute Fourier Transform
    X = np.fft.fft(noise)

    # Compute frequencies
    freqs = np.fft.fftfreq(len(X), 1 / sample_rate)

    # Compute scaling factors for the frequency domain
    scaling_factors = np.ones_like(freqs)
    non_zero_mask = freqs != 0
    scaling_factors[non_zero_mask] = 1.0 / (
        np.abs(freqs[non_zero_mask]) ** (resolved_alpha / 2)
    )

    # Apply scaling
    X_scaled = X * scaling_factors

    # Compute inverse Fourier Transform
    modified_signal = np.fft.ifft(X_scaled).real

    # Normalize the signal to [-1, 1]
    max_amplitude: np.floating[Any] = np.max(np.abs(modified_signal))
    if max_amplitude != 0:
        modified_signal /= max_amplitude

    # Convert to requested dtype
    if dtype == "int16":
        output_signal: npt.NDArray[Any] = (modified_signal * 32767).astype(np.int16)
    elif dtype == "float32":
        output_signal = modified_signal.astype(np.float32)
    elif dtype == "float64":
        output_signal = modified_signal.astype(np.float64)
    else:
        raise ValueError(f"Unknown dtype '{dtype}'. Valid options: int16, float32, float64")

    # Save as WAV file
    if output_file is not None:
        # WAV files need int16
        wav_signal = (
            output_signal
            if dtype == "int16"
            else (modified_signal * 32767).astype(np.int16)
        )
        wavfile.write(output_file, sample_rate, wav_signal)
        print(f"Generated {output_file} with alpha={resolved_alpha}.")

    # Optionally plot the power spectral density
    if plot_psd:
        plot_signal = (
            output_signal
            if dtype == "int16"
            else (modified_signal * 32767).astype(np.int16)
        )
        frequencies, psd = welch(plot_signal, fs=sample_rate, nperseg=1024)
        plt.figure(figsize=(12, 6))
        plt.loglog(frequencies, psd)
        plt.title(f"Power Spectral Density of 1/f^{resolved_alpha}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD (dB/Hz)")
        plt.grid(True)
        plt.show()

    return output_signal


# Convenience functions for common noise types


def white_noise(
    duration: float = 3,
    sample_rate: int = 44100,
    output_file: str | None = None,
    random_seed: int | None = 42,
    dtype: Literal["int16", "float32", "float64"] = "int16",
) -> npt.NDArray[np.int16] | npt.NDArray[np.float32] | npt.NDArray[np.float64]:
    """Generate white noise (equal power at all frequencies, alpha=0).

    Args:
        duration: Duration in seconds (default: 3)
        sample_rate: Sample rate in Hz (default: 44100)
        output_file: Path to save as WAV file (optional)
        random_seed: Seed for reproducibility, None for random (default: 42)
        dtype: Output format - "int16", "float32", or "float64" (default: "int16")

    Returns:
        Noise signal as numpy array
    """
    return generate_custom_noise(
        sample_rate=sample_rate,
        duration=duration,
        noise_type="white",
        output_file=output_file,
        random_seed=random_seed,
        dtype=dtype,
    )


def pink_noise(
    duration: float = 3,
    sample_rate: int = 44100,
    output_file: str | None = None,
    random_seed: int | None = 42,
    dtype: Literal["int16", "float32", "float64"] = "int16",
) -> npt.NDArray[np.int16] | npt.NDArray[np.float32] | npt.NDArray[np.float64]:
    """Generate pink noise (1/f spectrum, alpha=1).

    Pink noise has equal power per octave, making it perceptually balanced.
    Commonly used for audio testing and sound masking.

    Args:
        duration: Duration in seconds (default: 3)
        sample_rate: Sample rate in Hz (default: 44100)
        output_file: Path to save as WAV file (optional)
        random_seed: Seed for reproducibility, None for random (default: 42)
        dtype: Output format - "int16", "float32", or "float64" (default: "int16")

    Returns:
        Noise signal as numpy array
    """
    return generate_custom_noise(
        sample_rate=sample_rate,
        duration=duration,
        noise_type="pink",
        output_file=output_file,
        random_seed=random_seed,
        dtype=dtype,
    )


def brown_noise(
    duration: float = 3,
    sample_rate: int = 44100,
    output_file: str | None = None,
    random_seed: int | None = 42,
    dtype: Literal["int16", "float32", "float64"] = "int16",
) -> npt.NDArray[np.int16] | npt.NDArray[np.float32] | npt.NDArray[np.float64]:
    """Generate brown/Brownian noise (1/f^2 spectrum, alpha=2).

    Also known as red noise. Has a deeper, rumbling quality.
    Power decreases 6dB per octave.

    Note: True Brownian noise has infinite power at DC (0 Hz), making it
    non-stationary with unbounded drift. This implementation leaves the DC
    component unscaled to produce a bounded, stationary signal suitable for
    audio applications.

    Args:
        duration: Duration in seconds (default: 3)
        sample_rate: Sample rate in Hz (default: 44100)
        output_file: Path to save as WAV file (optional)
        random_seed: Seed for reproducibility, None for random (default: 42)
        dtype: Output format - "int16", "float32", or "float64" (default: "int16")

    Returns:
        Noise signal as numpy array
    """
    return generate_custom_noise(
        sample_rate=sample_rate,
        duration=duration,
        noise_type="brown",
        output_file=output_file,
        random_seed=random_seed,
        dtype=dtype,
    )


def blue_noise(
    duration: float = 3,
    sample_rate: int = 44100,
    output_file: str | None = None,
    random_seed: int | None = 42,
    dtype: Literal["int16", "float32", "float64"] = "int16",
) -> npt.NDArray[np.int16] | npt.NDArray[np.float32] | npt.NDArray[np.float64]:
    """Generate blue noise (f spectrum, alpha=-1).

    Power increases 3dB per octave. Has a bright, hissing quality.

    Args:
        duration: Duration in seconds (default: 3)
        sample_rate: Sample rate in Hz (default: 44100)
        output_file: Path to save as WAV file (optional)
        random_seed: Seed for reproducibility, None for random (default: 42)
        dtype: Output format - "int16", "float32", or "float64" (default: "int16")

    Returns:
        Noise signal as numpy array
    """
    return generate_custom_noise(
        sample_rate=sample_rate,
        duration=duration,
        noise_type="blue",
        output_file=output_file,
        random_seed=random_seed,
        dtype=dtype,
    )


def violet_noise(
    duration: float = 3,
    sample_rate: int = 44100,
    output_file: str | None = None,
    random_seed: int | None = 42,
    dtype: Literal["int16", "float32", "float64"] = "int16",
) -> npt.NDArray[np.int16] | npt.NDArray[np.float32] | npt.NDArray[np.float64]:
    """Generate violet/purple noise (f^2 spectrum, alpha=-2).

    Power increases 6dB per octave. Has a sharp, high-frequency emphasis.

    Args:
        duration: Duration in seconds (default: 3)
        sample_rate: Sample rate in Hz (default: 44100)
        output_file: Path to save as WAV file (optional)
        random_seed: Seed for reproducibility, None for random (default: 42)
        dtype: Output format - "int16", "float32", or "float64" (default: "int16")

    Returns:
        Noise signal as numpy array
    """
    return generate_custom_noise(
        sample_rate=sample_rate,
        duration=duration,
        noise_type="violet",
        output_file=output_file,
        random_seed=random_seed,
        dtype=dtype,
    )
