# Sponsor Logo Detection with YOLOv8

This computer-vision project detects sponsor and brand logos in sports media
using a custom YOLOv8 object detector.

## Project overview

- Prepared and annotated a dataset with 10 logo classes.
- Trained a YOLOv8 detector for 25 epochs at 640-pixel image size.
- Evaluated precision, recall, mAP, and confusion matrices.
- Built reusable commands for training and image/video inference.

The 10 classes are Adidas, Budweiser, Coca-Cola, Heineken, Mastercard,
McDonald's, PlayStation, Qatar Airways, Nike, and Visa.

## Repository structure

```text
.
├── dataset.example.yaml
├── docs/
│   ├── baseline_results.csv
│   ├── confusion_matrix.png
│   └── training_curves.png
├── requirements.txt
└── src/
    ├── predict.py
    ├── split_dataset.py
    └── train.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place YOLO-format images and labels under `data/source/images` and
`data/source/labels`. Then create a deterministic 80/20 split:

```bash
python src/split_dataset.py \
  --images data/source/images \
  --labels data/source/labels \
  --output data/split
```

Train and validate:

```bash
python src/train.py --data dataset.example.yaml --epochs 25
```

Run inference on an image, folder, or video:

```bash
python src/predict.py --weights runs/detect/sponsor_logos/weights/best.pt \
  --source path/to/media.mp4
```

## Results note

The `docs` folder preserves the original 25-epoch baseline artifacts. The
original experiment used the same image directory for training and validation,
so those metrics demonstrate pipeline behavior but should not be interpreted
as holdout performance. The included split utility fixes that issue for future
experiments.

The image dataset, model weights, virtual environment, and inference media are
excluded to keep the repository small and to respect source-media rights.

## Author

Cheikh Atamao  
[LinkedIn](https://www.linkedin.com/in/cheikh-atamao-056393263) ·
[GitHub](https://github.com/catamao17-star)
