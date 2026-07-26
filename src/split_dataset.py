"""Create deterministic YOLO train/validation folders."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def copy_pair(
    image: Path,
    label: Path,
    output: Path,
    split: str,
) -> None:
    image_dir = output / "images" / split
    label_dir = output / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image, image_dir / image.name)
    shutil.copy2(label, label_dir / label.name)


def main() -> None:
    args = parse_args()
    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1")

    images = sorted(
        path
        for path in args.images.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    pairs = []
    missing_labels = []
    for image in images:
        label = args.labels / f"{image.stem}.txt"
        if label.exists():
            pairs.append((image, label))
        else:
            missing_labels.append(image.name)

    if len(pairs) < 2:
        raise ValueError("At least two matched image/label pairs are required")

    random.Random(args.seed).shuffle(pairs)
    val_count = max(1, round(len(pairs) * args.val_ratio))
    validation = set(image for image, _ in pairs[:val_count])

    for image, label in pairs:
        split = "val" if image in validation else "train"
        copy_pair(image, label, args.output, split)

    print(f"Matched pairs: {len(pairs)}")
    print(f"Train: {len(pairs) - val_count}")
    print(f"Validation: {val_count}")
    print(f"Images without labels: {len(missing_labels)}")


if __name__ == "__main__":
    main()
