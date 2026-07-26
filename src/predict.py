"""Run sponsor-logo inference on an image, directory, or video."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.weights))
    model.predict(
        source=str(args.source),
        conf=args.confidence,
        save=True,
        show_labels=True,
        show_conf=True,
    )


if __name__ == "__main__":
    main()
