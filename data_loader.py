import warnings

import numpy as np
from PIL import Image

from features import extract_features


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def scan_anchors(base_dir, dates):
    sample_ids, class_names = set(), set()
    for date in dates:
        date_dir = base_dir / date
        if not date_dir.is_dir():
            continue
        for class_dir in date_dir.iterdir():
            if not class_dir.is_dir():
                continue
            class_names.add(class_dir.name)
            sample_ids.update(
                f"{class_dir.name}_{path.name}"
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )

    if not sample_ids:
        raise FileNotFoundError(f"No images found in {base_dir.resolve()}")
    return sorted(sample_ids), sorted(class_names)


def make_global_fold_map(sample_ids, n_folds, seed=7):
    if n_folds < 2 or len(sample_ids) < n_folds:
        raise ValueError("Invalid number of folds")
    shuffled = np.random.RandomState(seed).permutation(sample_ids)
    return {sample_id: i % n_folds for i, sample_id in enumerate(shuffled)}


def load_date(date_dir, class_names, image_size):
    if not date_dir.is_dir():
        raise FileNotFoundError(f"Date directory not found: {date_dir.resolve()}")

    features, labels, sample_ids = [], [], []
    for label, class_name in enumerate(class_names):
        class_dir = date_dir / class_name
        if not class_dir.is_dir():
            warnings.warn(f"Missing class folder: {class_dir}")
            continue
        for path in sorted(class_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                with Image.open(path) as image:
                    vector = extract_features(image.resize(image_size))
            except Exception as exc:
                warnings.warn(f"Skipping {path}: {exc}")
                continue
            features.append(vector)
            labels.append(label)
            sample_ids.append(f"{class_name}_{path.name}")

    if not features:
        raise ValueError(f"No readable images in {date_dir.resolve()}")
    return np.asarray(features), np.asarray(labels), np.asarray(sample_ids)
