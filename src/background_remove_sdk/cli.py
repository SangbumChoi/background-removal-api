"""Command-line interface: ``bg-remove INPUT [-o OUTPUT]``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from background_remove_sdk import core


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bg-remove",
        description="Remove the background from an image and save a transparent PNG.",
    )
    parser.add_argument("input", help="path to the input image")
    parser.add_argument(
        "-o",
        "--output",
        help="path for the output PNG (default: <input>_no_bg.png next to the input)",
    )
    parser.add_argument(
        "--mask",
        action="store_true",
        help="output the grayscale foreground mask instead of the cutout",
    )
    parser.add_argument(
        "--point",
        nargs=2,
        type=int,
        metavar=("X", "Y"),
        help="extract only the object containing pixel (X, Y), cropped to its bounding box",
    )
    parser.add_argument(
        "--mode",
        default="base",
        choices=["base", "fast", "base-nightly"],
        help="model variant: base (quality), fast (speed) (default: base)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='torch device, e.g. "cuda:0" or "cpu" (default: auto-detect)',
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.mask and args.point:
        print("error: --mask and --point cannot be combined", file=sys.stderr)
        return 2

    kind = "mask" if args.mask else "object" if args.point else "rgba"
    output = Path(args.output) if args.output else core.default_output_path(args.input, kind)

    try:
        if args.mask:
            core.generate_mask(args.input, output_path=output, mode=args.mode, device=args.device)
        elif args.point:
            x, y = args.point
            core.extract_object_at_point(
                args.input, x, y, output_path=output, mode=args.mode, device=args.device
            )
        else:
            core.remove_background(
                args.input, output_path=output, mode=args.mode, device=args.device
            )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
