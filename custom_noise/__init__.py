# custom_noise/__init__.py
"""Generate noise signals with custom power spectral density (1/f^alpha).

Quick start:
    >>> from custom_noise import pink_noise, brown_noise
    >>> signal = pink_noise(duration=5)
    >>> brown_noise(duration=10, output_file="brown.wav")

Available noise types:
    - white_noise():  Equal power at all frequencies (alpha=0)
    - pink_noise():   1/f spectrum, equal power per octave (alpha=1)
    - brown_noise():  1/f^2 spectrum, deep rumbling (alpha=2)
    - blue_noise():   f spectrum, bright/hissing (alpha=-1)
    - violet_noise(): f^2 spectrum, high-frequency emphasis (alpha=-2)

For custom alpha values, use generate_custom_noise(alpha=X).
"""
from .noise_generator import (
    NOISE_PRESETS,
    blue_noise,
    brown_noise,
    generate_custom_noise,
    pink_noise,
    violet_noise,
    white_noise,
)

__all__ = [
    "generate_custom_noise",
    "white_noise",
    "pink_noise",
    "brown_noise",
    "blue_noise",
    "violet_noise",
    "NOISE_PRESETS",
]
__version__ = "0.1.0"
