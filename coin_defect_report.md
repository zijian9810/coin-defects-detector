# Coin defect detection evaluation

Evaluation date: September 25, 2026.

The frozen ViT-B/16 baseline in `coin_defect_detection.py` was fitted and tested
on the supplied circulation-quality coin-image dataset using an Apple GPU.
Mahalanobis scoring achieved ROC-AUC **0.9254** and cosine kNN achieved
**0.9084**. At thresholds calibrated exclusively on good training images,
both methods detected **307 of 495 defects (62.02%)**, missing **188 defects**.

## Dataset and separation

| Subset | Images | Purpose |
|---|---:|---|
| Good training images used for fitting | 2,564 | Fit covariance or build the reference bank |
| Held-out good training images | 641 | Calibrate decision thresholds |
| Good test images | 800 | Evaluate false alarms |
| Defective test images | 495 | Evaluate defect detection |

The 3,205 good training images were split with random seed 42 and a 20%
calibration holdout. Calibration images were not subsequently added to the
fitting set. Test labels were not used for fitting, threshold calibration, or
hyperparameter tuning.

One of the original 801 good test images was excluded after a SHA-256 check
confirmed it was an exact byte-for-byte duplicate of a training image:

- Excluded: `test/good/Circulation_quality__good__CAM1-20260313 163344171_resized-checkpoint.tif`
- Matching training image: `train/good/Circulation_quality__good__CAM1-20260313 163344171_resized.tif`

The final test set contained **1,295 images**. Source images were not modified.
This duplicate check identifies identical files; it does not rule out distinct
photographs of the same physical coin or recompressed copies.

## Model and settings

- Backbone: torchvision ViT-B/16 with `IMAGENET1K_V1` pretrained weights, frozen
  in evaluation mode with its classification head replaced by an identity layer.
- Input: RGB images resized with preserved aspect ratio and padded to 224 × 224,
  retaining the entire image rather than cropping the rim.
- Normalization: ImageNet channel means and standard deviations.
- Representation: one 768-dimensional final CLS embedding per image.
- Mahalanobis: full sample covariance with shrinkage 0.1 toward a scaled identity,
  plus diagonal ridge 0.001; distances computed through a whitening transform.
- kNN: mean cosine similarity to the five nearest good fitting-set embeddings;
  anomaly score is one minus this similarity.
- Thresholds: empirical 99th percentile of held-out good scores using the
  `higher` quantile method. An image is flagged when its score exceeds the threshold.
- Batch size: 16; device: Apple GPU (`mps`).

The ViT was not fine-tuned on coin images. The two methods produce independent
scores and predictions; no ensemble rule was fitted.

## Results

| Metric | Mahalanobis | Cosine kNN |
|---|---:|---:|
| ROC-AUC | 0.9254318 | 0.9083535 |
| Decision threshold | 28.1052906 | 0.0753429 |
| Defects detected | 307 / 495 | 307 / 495 |
| Defects missed | 188 / 495 | 188 / 495 |
| Defect recall | 62.02% | 62.02% |
| Good images incorrectly flagged | 6 / 800 | 8 / 800 |
| Good-image false-positive rate | 0.75% | 1.00% |
| Precision among flagged images | 98.08% | 97.46% |
| Accuracy | 85.02% | 84.86% |

The equal detection counts do not imply that the methods flagged the same images.
ROC-AUC measures ranking across thresholds; recall and false-positive rate above
describe the particular good-only calibrated operating points.

Both methods have limited recall at these thresholds. A single global embedding
can overlook small local defects, and lighting or orientation can affect image
features. These results apply to this dataset split and do not establish
production performance. Changes to thresholds or models should be selected on
separate validation data rather than tuned against these test labels.

## Reproduce

Use Python 3.10 or newer. The tested environment was Python 3.12.3 on macOS
26.3 ARM64, with torch 2.14.0, torchvision 0.29.0, NumPy 2.5.3, and Pillow 12.3.0.
The first run downloads pretrained weights if they are not already cached.

Run from the directory containing `coin_defect_detection.py`:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.14.0 torchvision==0.29.0 numpy==2.5.3 pillow==12.3.0

# Set this to your local dataset directory.
COIN_DATASET=/path/to/dataset_public_circulation_quality

.venv/bin/python coin_defect_detection.py fit \
  --data-root "$COIN_DATASET" \
  --model coin_test_run/coin_detector.npz \
  --method both --batch-size 16 --device auto

.venv/bin/python coin_defect_detection.py evaluate \
  --test-dir "$COIN_DATASET/test" \
  --model coin_test_run/coin_detector.npz \
  --output coin_test_run/coin_test_scores.csv \
  --batch-size 16 --device auto \
  --exclude-image "$COIN_DATASET/test/good/Circulation_quality__good__CAM1-20260313 163344171_resized-checkpoint.tif"

.venv/bin/python coin_defect_detection.py predict \
  --model coin_test_run/coin_detector.npz \
  --input /path/to/new_coin.tif \
  --output coin_test_run/coin_predictions.csv
```

The exclusion above is specific to the evaluated dataset; omit it if that image
is absent from your copy. `--device auto` selects CUDA, Apple GPU, or CPU as
available. Hardware and library differences can cause small numerical changes.
On the tested Mac, setting `SSL_CERT_FILE=/etc/ssl/cert.pem` was needed to use
the system certificate bundle for the initial weight download.

## Outputs and verification

Running the commands generates a saved detector (`coin_detector.npz`), a CSV of
per-image scores and predictions, and a metrics JSON file. The dataset, cached
pretrained weights, fitted detector, and per-image outputs are local run artifacts,
not part of this two-file source-and-report upload.

The completed local run verified detector reload, exported labels and predictions,
confusion counts, separation of fitting/calibration paths from evaluated paths,
and removal of the identified duplicate. ROC-AUC was independently recomputed
by comparing every defective-image score against every good-image score,
counting tied scores as half a win.

Evaluated script SHA-256:
`b2762bd539ccbdf410d6e2972b8376512cd346f11c72dc9f7df696f3d3b79250`.
