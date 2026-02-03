# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python package (`custom_noise`) for generating noise signals with custom power spectral density (PSD) of 1/f^alpha. Supports white (alpha=0), pink (alpha=1), brown (alpha=2), blue (alpha=-1), and purple (alpha=-2) noise.

## Commands

```bash
# Install package in development mode (with dev dependencies)
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test
pytest tests/test_noise_generator.py::test_generate_custom_noise

# Run tests with verbose output
pytest -v

# Run linter
ruff check .

# Run formatter
ruff format .

# Run type checker
mypy custom_noise
```

## CLI Usage

```bash
# Generate pink noise (default)
python -m custom_noise --type pink --duration 5 -o pink_noise.wav

# Generate with custom alpha
python -m custom_noise --alpha 1.5 --duration 10 -o custom.wav

# Generate brown noise at 48kHz
python -m custom_noise --type brown --sample-rate 48000 -o brown_48k.wav

# Show PSD plot
python -m custom_noise --type white --plot

# See all options
python -m custom_noise --help
```

## Library Usage

```python
from custom_noise import pink_noise, brown_noise, white_noise

# Simple usage
signal = pink_noise(duration=5)
brown_noise(duration=10, output_file="brown.wav")

# With options
signal = pink_noise(duration=5, sample_rate=48000, dtype="float32")

# For custom alpha values
from custom_noise import generate_custom_noise
signal = generate_custom_noise(alpha=1.5, duration=3)
```

## Architecture

The package exposes convenience functions (`pink_noise`, `brown_noise`, etc.) and a general `generate_custom_noise()` function via `custom_noise/__init__.py`.

**Core algorithm** (`custom_noise/noise_generator.py`):
1. Generate Gaussian white noise using numpy's modern RNG
2. Apply FFT to transform to frequency domain
3. Scale frequencies by 1/|f|^(alpha/2) to achieve desired PSD slope
4. Inverse FFT back to time domain
5. Normalize and convert to requested output format

**Key parameters**:
- `alpha`: Controls the PSD slope (0=white, 1=pink, 2=brown, negative=blue/purple)
- `noise_type`: Named preset ("white", "pink", "brown", "blue", "violet", etc.)
- `output_file`: When provided, saves as WAV file; otherwise returns array only
- `dtype`: Output format - "int16" (default), "float32", or "float64"
- `random_seed`: Ensures reproducibility (default: 42, use None for random)
