# Custom Noise Generator

A Python package for generating noise signals with custom power spectral density (PSD) of 1/f^alpha.

## Installation

```bash
pip install custom_noise
```

Or install from source:

```bash
git clone https://github.com/yourusername/custom_noise.git
cd custom_noise
pip install -e .
```

## Quick Start

```python
from custom_noise import pink_noise, brown_noise

# Generate 5 seconds of pink noise
signal = pink_noise(duration=5)

# Save brown noise to a WAV file
brown_noise(duration=10, output_file="brown_noise.wav")
```

## Noise Types

| Function | Alpha | Description |
|----------|-------|-------------|
| `white_noise()` | 0 | Equal power at all frequencies |
| `pink_noise()` | 1 | 1/f spectrum, equal power per octave |
| `brown_noise()` | 2 | 1/f² spectrum, deep rumbling (also called red/Brownian noise)* |
| `blue_noise()` | -1 | f spectrum, bright/hissing quality |
| `violet_noise()` | -2 | f² spectrum, high-frequency emphasis |

*Note: True Brownian noise has infinite power at DC (0 Hz), causing unbounded drift. This implementation leaves the DC component unscaled to produce a bounded, stationary signal suitable for audio use.

## Usage

### Basic Usage

```python
from custom_noise import pink_noise, white_noise, brown_noise

# Generate noise (returns numpy array)
signal = pink_noise(duration=3)

# Save directly to file
white_noise(duration=5, output_file="white.wav")

# Custom sample rate
signal = brown_noise(duration=2, sample_rate=48000)
```

### Output Formats

```python
from custom_noise import pink_noise

# 16-bit integers (default) - ready for WAV files
signal_int = pink_noise(duration=1, dtype="int16")

# Normalized float [-1.0, 1.0] - for DSP processing
signal_float = pink_noise(duration=1, dtype="float32")
```

### Custom Alpha Values

For noise colors not covered by the convenience functions:

```python
from custom_noise import generate_custom_noise

# Custom alpha value
signal = generate_custom_noise(alpha=0.5, duration=3)

# Gray noise (alpha ≈ 1, psychoacoustically balanced)
signal = generate_custom_noise(alpha=1, duration=5)
```

### Reproducibility

```python
from custom_noise import pink_noise

# Same seed = same output
signal1 = pink_noise(duration=1, random_seed=42)
signal2 = pink_noise(duration=1, random_seed=42)
assert (signal1 == signal2).all()

# Random each time
signal = pink_noise(duration=1, random_seed=None)
```

### Visualization

```python
from custom_noise import generate_custom_noise

# Plot the power spectral density
generate_custom_noise(alpha=1, duration=3, plot_psd=True)
```

## Command Line Interface

```bash
# Generate pink noise
python -m custom_noise --type pink --duration 5 -o pink.wav

# Generate with custom alpha
python -m custom_noise --alpha 1.5 --duration 10 -o custom.wav

# Show PSD plot
python -m custom_noise --type brown --duration 3 --plot

# See all options
python -m custom_noise --help
```

### CLI Options

| Option | Description |
|--------|-------------|
| `-t, --type TYPE` | Noise type: white, pink, brown, blue, violet |
| `-a, --alpha ALPHA` | Custom alpha exponent |
| `-d, --duration SECONDS` | Duration in seconds (default: 3) |
| `-r, --sample-rate HZ` | Sample rate in Hz (default: 44100) |
| `-o, --output FILE` | Output WAV file path |
| `--plot` | Display PSD plot |
| `-s, --seed SEED` | Random seed for reproducibility |

## API Reference

### Convenience Functions

All convenience functions share these parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `duration` | float | 3 | Duration in seconds |
| `sample_rate` | int | 44100 | Sample rate in Hz |
| `output_file` | str | None | Path to save WAV file |
| `random_seed` | int | 42 | Seed for reproducibility (None for random) |
| `dtype` | str | "int16" | Output type: "int16", "float32", "float64" |

### generate_custom_noise()

Full control over noise generation:

```python
generate_custom_noise(
    sample_rate=44100,    # Sample rate in Hz
    duration=3,           # Duration in seconds
    alpha=None,           # PSD exponent (mutually exclusive with noise_type)
    noise_type=None,      # Preset: "white", "pink", "brown", "blue", "violet"
    output_file=None,     # Path to save WAV file
    plot_psd=False,       # Show PSD plot
    random_seed=42,       # Seed for reproducibility
    dtype="int16",        # Output format
)
```

## How It Works

1. Generate Gaussian white noise
2. Apply FFT to transform to frequency domain
3. Scale frequencies by 1/|f|^(α/2) to achieve desired PSD slope
4. Inverse FFT back to time domain
5. Normalize and convert to requested output format

## Requirements

- Python >= 3.9
- NumPy
- SciPy
- Matplotlib (for plotting)

## License

MIT License
