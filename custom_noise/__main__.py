# custom_noise/__main__.py
"""CLI interface for custom_noise package.

Usage:
    python -m custom_noise [options]

Examples:
    python -m custom_noise --type pink --duration 5 -o pink_noise.wav
    python -m custom_noise --alpha 1.5 --duration 10 -o custom.wav
    python -m custom_noise --type brown --plot
"""
from __future__ import annotations

import argparse
import sys

from .noise_generator import NOISE_PRESETS, generate_custom_noise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="custom_noise",
        description="Generate noise signals with custom power spectral density (1/f^alpha).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Noise type presets:
  white     alpha=0   Equal power at all frequencies
  pink      alpha=1   Power decreases 3dB per octave
  brown     alpha=2   Power decreases 6dB per octave (also: brownian, red)
  blue      alpha=-1  Power increases 3dB per octave
  violet    alpha=-2  Power increases 6dB per octave (also: purple, azure)

Examples:
  %(prog)s --type pink --duration 5 -o pink_noise.wav
  %(prog)s --alpha 1.5 --duration 10 -o custom.wav
  %(prog)s --type brown --sample-rate 48000 -o brown_48k.wav
""",
    )

    # Noise specification (mutually exclusive)
    noise_group = parser.add_mutually_exclusive_group()
    noise_group.add_argument(
        "-t",
        "--type",
        choices=sorted(set(NOISE_PRESETS.keys())),
        help="Noise type preset (default: pink)",
        metavar="TYPE",
    )
    noise_group.add_argument(
        "-a",
        "--alpha",
        type=float,
        help="Custom alpha exponent for 1/f^alpha PSD",
        metavar="ALPHA",
    )

    # Audio parameters
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=3.0,
        help="Duration in seconds (default: 3.0)",
        metavar="SECONDS",
    )
    parser.add_argument(
        "-r",
        "--sample-rate",
        type=int,
        default=44100,
        help="Sample rate in Hz (default: 44100)",
        metavar="HZ",
    )

    # Output
    parser.add_argument(
        "-o",
        "--output",
        help="Output WAV file path (required unless --plot is used)",
        metavar="FILE",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display power spectral density plot",
    )

    # Reproducibility
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: random)",
        metavar="SEED",
    )

    args = parser.parse_args(argv)

    # Validate: need either output file or plot
    if not args.output and not args.plot:
        parser.error("Must specify --output FILE and/or --plot")

    # Resolve noise type
    noise_type = args.type
    alpha = args.alpha
    if noise_type is None and alpha is None:
        noise_type = "pink"  # Default

    try:
        generate_custom_noise(
            sample_rate=args.sample_rate,
            duration=args.duration,
            alpha=alpha,
            noise_type=noise_type,
            output_file=args.output,
            plot_psd=args.plot,
            random_seed=args.seed,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
