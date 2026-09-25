#!/usr/bin/env python3
"""Coin anomaly detection with a frozen, ImageNet-pretrained ViT-B/16.

Install (Python 3.10+):
    python -m pip install torch torchvision numpy pillow

Expected dataset:
    dataset_public_circulation_quality/train/good/*.{tif,png,jpg,...}
    dataset_public_circulation_quality/test/good/*
    dataset_public_circulation_quality/test/bad/*

Examples:
    python coin_defect_detection.py fit --data-root /path/to/dataset
    python coin_defect_detection.py evaluate --test-dir /path/to/dataset/test
    python coin_defect_detection.py predict --input /path/to/coin.tif

fit saves coin_detector.npz; evaluate/predict load it using --model.
The first fit or inference downloads torchvision's pretrained weights (~330 MB)
if they are not already cached. No network fine-tuning is performed.

Whole images are resized with aspect ratio preserved and black padding, without
cropping the rim. One 768-dimensional CLS embedding represents each image.
This is an image-level baseline, not a defect-localization/heatmap model.
Small defects, orientation, and lighting can affect performance.

By default, both a regularized full-covariance Mahalanobis model and a cosine
kNN model are fitted. Their scores and decisions remain separate. A seeded
holdout of good TRAINING images calibrates each threshold; test labels are
used only for evaluation. The holdout is not subsequently added to the fit set.
The threshold quantile is an empirical operating point, not a probability of
defect or a guaranteed false-positive rate. Split repeated captures of the
same physical coin before using this script to avoid train/test leakage.

References:
    https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html
    https://docs.pytorch.org/vision/stable/_modules/torchvision/models/vision_transformer.html
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageOps


EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}
BACKBONE = "torchvision.vit_b_16.IMAGENET1K_V1"
IMAGE_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_ROOT = Path.home() / "Downloads" / "dataset_public_circulation_quality"


def image_paths(source: Path) -> list[Path]:
    """List supported images deterministically; fail rather than skip bad input."""
    source = source.expanduser().resolve()
    if source.is_file():
        paths = [source] if source.suffix.lower() in EXTENSIONS else []
    elif source.is_dir():
        paths = sorted(p for p in source.rglob("*")
                       if p.is_file() and p.suffix.lower() in EXTENSIONS
                       and not any(part.startswith(".")
                                   for part in p.relative_to(source).parts))
    else:
        raise ValueError(f"Input does not exist: {source}")
    if not paths:
        raise ValueError(f"No supported images found in {source}")
    return paths


def preprocess_image(path: Path) -> np.ndarray:
    """Return normalized RGB CHW float32, retaining the entire image."""
    try:
        with Image.open(path) as source:
            if getattr(source, "n_frames", 1) != 1:
                raise ValueError("Multi-page images are not supported")
            image = ImageOps.exif_transpose(source).convert("RGB")
            image = ImageOps.pad(image, (IMAGE_SIZE, IMAGE_SIZE),
                                 method=Image.Resampling.BILINEAR, color=(0, 0, 0))
            pixels = np.asarray(image, dtype=np.float32) / 255.0
    except Exception as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc
    return np.ascontiguousarray(((pixels - MEAN) / STD).transpose(2, 0, 1))


def load_backbone(device_name: str):
    """Lazy imports let --help and numerical utilities run without PyTorch."""
    try:
        import torch
        from torchvision.models import ViT_B_16_Weights, vit_b_16
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("Install compatible PyTorch/torchvision packages with "
                           "'python -m pip install torch torchvision numpy pillow'. "
                           f"Original error: {exc}") from exc
    if device_name == "auto":
        device_name = ("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device_name)
    print(f"Loading frozen {BACKBONE} on {device} ...", flush=True)
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads = torch.nn.Identity()  # forward now returns the final CLS embedding
    model.requires_grad_(False)
    model.eval().to(device)
    return model, device


def extract_features(paths: list[Path], model, device, batch_size: int) -> np.ndarray:
    import torch

    batches = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start:start + batch_size]
            batch = torch.from_numpy(np.stack([preprocess_image(p) for p in batch_paths]))
            features = model(batch.to(device)).cpu().numpy()
            if features.ndim != 2 or features.shape[1] != 768:
                raise ValueError(f"Expected (batch, 768) embeddings; got {features.shape}")
            if not np.isfinite(features).all():
                raise ValueError("ViT produced non-finite embeddings")
            batches.append(features)
            print(f"  Embedded {min(start + batch_size, len(paths))}/{len(paths)} images",
                  flush=True)
    return np.concatenate(batches, axis=0)


def fit_gaussian(features: np.ndarray, shrinkage: float, ridge: float) -> dict:
    """Store a whitening transform, avoiding an explicit covariance inverse.

    Sigma = (1-shrinkage)*sample_cov + shrinkage*mean_variance*I + ridge*I.
    The distance is ||(x-mu) W||_2, where W W^T = Sigma^{-1}.
    """
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Gaussian fitting needs at least two finite feature vectors")
    if not 0 <= shrinkage <= 1 or not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("shrinkage must be in [0, 1] and ridge must be positive")
    mu = x.mean(axis=0)
    centered = x - mu
    cov = centered.T @ centered / (len(x) - 1)
    target_variance = np.trace(cov) / cov.shape[0]
    cov *= 1.0 - shrinkage
    cov.flat[::cov.shape[0] + 1] += shrinkage * target_variance + ridge
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Positive ridge guarantees positivity mathematically; floor roundoff errors.
    whitening = eigenvectors / np.sqrt(np.maximum(eigenvalues, ridge))
    return {"mu": mu, "whitening": whitening}


def unit_rows(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(x).all() or np.any(norm <= 1e-12):
        raise ValueError("Cosine scoring requires finite, nonzero feature vectors")
    return x / norm


def score_features(features: np.ndarray, state: dict, method: str, k: int) -> np.ndarray:
    if method == "mahalanobis":
        whitened = (features - state["mu"]) @ state["whitening"]
        return np.linalg.norm(whitened, axis=1)
    if method != "knn":
        raise ValueError(f"Unknown scoring method: {method}")
    bank = state["bank"]  # Already normalized, containing only fitting-set good images.
    neighbors = min(k, len(bank))
    if neighbors < 1:
        raise ValueError("k and reference-bank size must be positive")
    scores = []
    # Bound the query-by-reference similarity matrix's memory usage.
    for start in range(0, len(features), 128):
        similarity = unit_rows(features[start:start + 128]) @ bank.T
        top = np.partition(similarity, similarity.shape[1] - neighbors, axis=1)[:, -neighbors:]
        scores.extend(1.0 - np.clip(top, -1.0, 1.0).mean(axis=1))
    return np.asarray(scores)


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Rank-based ROC-AUC with average ranks for ties; positive class is defect."""
    labels = np.asarray(labels)
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("AUC requires binary labels and finite scores")
    n_pos, n_neg = int(labels.sum()), int((labels == 0).sum())
    if not n_pos or not n_neg:
        return None
    order = np.argsort(scores)
    _, starts, counts = np.unique(scores[order], return_index=True, return_counts=True)
    sorted_ranks = np.repeat(starts + (counts + 1) / 2.0, counts)
    positive_rank_sum = sorted_ranks[labels[order] == 1].sum()
    return float((positive_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def classification_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = scores > threshold
    tp = int(np.sum(predicted & (labels == 1)))
    fp = int(np.sum(predicted & (labels == 0)))
    tn = int(np.sum(~predicted & (labels == 0)))
    fn = int(np.sum(~predicted & (labels == 1)))
    return {"roc_auc": roc_auc(labels, scores), "threshold": threshold,
            "accuracy": (tp + tn) / len(labels),
            "defect_recall": tp / (tp + fn) if tp + fn else None,
            "defect_precision": tp / (tp + fp) if tp + fp else None,
            "good_false_positive_rate": fp / (fp + tn) if fp + tn else None,
            "true_positive": tp, "false_positive": fp,
            "true_negative": tn, "false_negative": fn}


def save_detector(path: Path, metadata: dict, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Opening explicitly prevents numpy silently appending an extra .npz suffix.
    with path.open("wb") as stream:
        np.savez_compressed(stream, metadata=np.array(json.dumps(metadata)), **state)


def load_detector(path: Path) -> tuple[dict, dict]:
    with np.load(path.expanduser(), allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"].item()))
        state = {key: saved[key].copy() for key in saved.files if key != "metadata"}
    if metadata.get("format_version") != 1 or metadata.get("backbone") != BACKBONE:
        raise ValueError("Detector format/backbone is not supported by this script")
    return metadata, state


def write_scores(path: Path, paths: list[Path], scores: dict, thresholds: dict,
                 labels: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["image", "true_label"]
    for method in scores:
        columns.extend([f"{method}_score", f"{method}_prediction"])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for i, image in enumerate(paths):
            row = {"image": str(image), "true_label": "" if labels is None else int(labels[i])}
            for method, values in scores.items():
                row[f"{method}_score"] = float(values[i])
                row[f"{method}_prediction"] = "defect" if values[i] > thresholds[method] else "good"
            writer.writerow(row)


def fit(args) -> None:
    paths = image_paths(args.data_root / "train" / "good")
    n_calibration = max(2, int(math.ceil(len(paths) * args.calibration_fraction)))
    if len(paths) - n_calibration < 2:
        raise ValueError("Need at least two fitting and two held-out calibration images")
    permutation = np.random.default_rng(args.seed).permutation(len(paths))
    calibration_indices, fit_indices = permutation[:n_calibration], permutation[n_calibration:]
    model, device = load_backbone(args.device)
    features = extract_features(paths, model, device, args.batch_size)
    reference, calibration = features[fit_indices], features[calibration_indices]
    methods = ["mahalanobis", "knn"] if args.method == "both" else [args.method]
    state = {}
    if "mahalanobis" in methods:
        state.update(fit_gaussian(reference, args.shrinkage, args.ridge))
    if "knn" in methods:
        state["bank"] = unit_rows(reference)
    scores = {method: score_features(calibration, state, method, args.k) for method in methods}
    thresholds = {method: float(np.quantile(values, args.threshold_quantile, method="higher"))
                  for method, values in scores.items()}
    metadata = {"format_version": 1, "backbone": BACKBONE, "image_size": IMAGE_SIZE,
                "preprocessing": "RGB, aspect-preserving resize/pad, ImageNet normalization",
                "methods": methods, "k": args.k, "shrinkage": args.shrinkage,
                "ridge": args.ridge, "thresholds": thresholds,
                "threshold_quantile": args.threshold_quantile, "seed": args.seed,
                "fit_count": len(reference), "calibration_count": len(calibration),
                "fit_paths": [str(paths[i]) for i in fit_indices],
                "calibration_paths": [str(paths[i]) for i in calibration_indices]}
    save_detector(args.model.expanduser(), metadata, state)
    print(f"Saved detector: {args.model.expanduser().resolve()}")
    print(f"Fitted on {len(reference)} good images; calibrated on {len(calibration)} held-out good images.")
    print("Thresholds: " + json.dumps(thresholds, indent=2))


def infer(args) -> None:
    metadata, state = load_detector(args.model)
    labels = None
    if args.command == "evaluate":
        root = args.test_dir.expanduser().resolve()
        paths = image_paths(root)
        exclusions = {p.expanduser().resolve() for p in args.exclude_image}
        missing = exclusions.difference(paths)
        if missing:
            raise ValueError(f"Excluded images were not found in the test set: {sorted(map(str, missing))}")
        paths = [p for p in paths if p not in exclusions]
        if not paths:
            raise ValueError("No evaluation images remain after exclusions")
        # The first subdirectory is the class: good=0, every other class=1.
        if any(len(p.relative_to(root).parts) < 2 for p in paths):
            raise ValueError("Evaluation images must be inside test/good or test/<defect-type>")
        labels = np.array([0 if p.relative_to(root).parts[0] == "good" else 1 for p in paths])
        used = set(metadata["fit_paths"] + metadata["calibration_paths"])
        if any(str(p) in used for p in paths):
            raise ValueError("Evaluation includes images used for fitting/calibration")
    else:
        paths = image_paths(args.input)
    model, device = load_backbone(args.device)
    features = extract_features(paths, model, device, args.batch_size)
    scores = {method: score_features(features, state, method, metadata["k"])
              for method in metadata["methods"]}
    write_scores(args.output.expanduser(), paths, scores, metadata["thresholds"], labels)
    print(f"Saved image scores: {args.output.expanduser().resolve()}")
    if labels is not None:
        report = {"image_count": len(paths), "good_count": int(np.sum(labels == 0)),
                  "defect_count": int(np.sum(labels == 1)),
                  "excluded_images": sorted(map(str, exclusions)),
                  "methods": {method: classification_metrics(labels, values, metadata["thresholds"][method])
                              for method, values in scores.items()}}
        report_path = args.output.expanduser().with_suffix(".metrics.json")
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        print(f"Saved metrics: {report_path.resolve()}")
    elif len(paths) == 1:
        for method, values in scores.items():
            threshold = metadata["thresholds"][method]
            label = "defect" if values[0] > threshold else "good"
            print(f"{method}: {label}; score={values[0]:.6f}; threshold={threshold:.6f}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", type=Path, default=Path("coin_detector.npz"), help="Detector file to save/load")
    common.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    common.add_argument("--batch-size", type=int, default=16)
    fitting = commands.add_parser("fit", parents=[common], help="Fit and calibrate on train/good only")
    fitting.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    fitting.add_argument("--method", choices=["mahalanobis", "knn", "both"], default="both")
    fitting.add_argument("--k", type=int, default=5)
    fitting.add_argument("--shrinkage", type=float, default=0.1, help="Covariance shrinkage toward scaled identity")
    fitting.add_argument("--ridge", type=float, default=1e-3, help="Additional covariance diagonal regularization")
    fitting.add_argument("--calibration-fraction", type=float, default=0.2)
    fitting.add_argument("--threshold-quantile", type=float, default=0.99,
                         help="Held-out good-score quantile; lower flags more images")
    fitting.add_argument("--seed", type=int, default=42)
    evaluation = commands.add_parser("evaluate", parents=[common], help="Evaluate labeled test subdirectories")
    evaluation.add_argument("--test-dir", type=Path, default=DEFAULT_ROOT / "test")
    evaluation.add_argument("--output", type=Path, default=Path("coin_test_scores.csv"))
    evaluation.add_argument("--exclude-image", type=Path, action="append", default=[],
                            help="Exclude one known duplicate/invalid test image; repeat as needed")
    prediction = commands.add_parser("predict", parents=[common], help="Score one image or an unlabeled folder")
    prediction.add_argument("--input", type=Path, required=True)
    prediction.add_argument("--output", type=Path, default=Path("coin_predictions.csv"))
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.command == "fit":
        if not 0 < args.calibration_fraction < 1 or not 0 < args.threshold_quantile < 1:
            parser.error("Calibration fraction and threshold quantile must be between 0 and 1")
        if args.k < 1 or not 0 <= args.shrinkage <= 1 or not math.isfinite(args.ridge) or args.ridge <= 0:
            parser.error("Need k>=1, shrinkage in [0,1], and positive finite ridge")
        args.data_root = args.data_root.expanduser()
    try:
        fit(args) if args.command == "fit" else infer(args)
    except (ValueError, RuntimeError, OSError, KeyError, np.linalg.LinAlgError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
