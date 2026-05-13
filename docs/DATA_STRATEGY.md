# Data Strategy

The project has roughly 10-15 landmark classes with about 400 images per class. That is enough to fine-tune, but too small to trust a single random validation split.

## Split Policy

Use this policy for every candidate repo so that MobileCLIP2 and MobileNetV4 are compared fairly.

1. Build a canonical `labels_master.json` from `Dataset/<landmark_id>/labels.json`.
2. Include only `label_status == confirmed` in supervised train/validation/test.
3. Put `uncertain`, `reject`, and non-confirmed records into `holdout_non_confirmed`.
4. Lock a final test split per class using `seed=20260513`.
5. Use 5-fold stratified validation on the remaining trainval set.
6. Never tune thresholds or hyperparameters on the locked final test split.
7. Record `dataset_fingerprint` in the split manifest and W&B config so growing datasets are never compared as if they were the same snapshot.

## Why K-Fold

With limited data, a single validation split can overrate a model because the easier images landed in validation. K-fold validation gives:

- mean score
- standard deviation
- fold-level failure patterns
- more trustworthy candidate ranking

## Leakage Control

The splitter keeps exact duplicate file hashes in the same split when possible. It does not guarantee perceptual duplicate removal, so before final reporting, run a perceptual hash audit if the dataset contains crawled near-duplicates.

## Out-of-Scope Handling

Out-of-scope performance cannot be proven from positive landmark classes alone. Use these as separate calibration/evaluation inputs:

- `label_status == reject`
- `label_status == uncertain`
- non-target landmarks in Jongno/Seoul
- indoor/object/person/travel photos unrelated to target landmarks

Report `out_of_scope_auroc` only when a real negative set exists. Until then, call it a threshold smoke check, not a validated rejection metric.

## Model Selection Rule

Pick the model with the best mean validation Top-3 accuracy. If models are tied within one fold standard deviation, prefer:

1. better Top-1
2. better out-of-scope AUROC
3. successful ONNX export
4. lower latency and smaller model

Only after selecting finalists should the locked final test be used.
