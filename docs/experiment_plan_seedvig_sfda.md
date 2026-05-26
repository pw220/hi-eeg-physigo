# SEED-VIG EEG Fatigue Detection Experiment Plan

## Research Direction

Project topic: EEG-based fatigue/vigilance detection under cross-subject distribution shift.

Main method direction: offline source-free domain adaptation (SFDA). The source model is trained with labeled source subjects, then adapted to an unlabeled target subject without access to source data.

## Dataset

Raw dataset path:

```text
data/raw/SEED-VIG
```

Useful folders:

```text
data/raw/SEED-VIG/EEG_Feature_5Bands
data/raw/SEED-VIG/EEG_Feature_2Hz
data/raw/SEED-VIG/Forehead_EEG
data/raw/SEED-VIG/EOG_Feature
data/raw/SEED-VIG/perclos_labels
data/raw/SEED-VIG/Raw_Data
```

Recommended first-stage input:

```text
EEG_Feature_5Bands + perclos_labels
```

Reason: this is fast, stable, and close to the official SEED-VIG feature protocol. After the experimental framework works, raw EEG preprocessing can be added as a stronger but slower track.

## Label Definition

SEED-VIG provides continuous PERCLOS labels. Use two settings:

1. Regression: predict continuous vigilance/fatigue score.
2. Classification: convert PERCLOS into classes.

Suggested binary classification thresholds:

```text
alert: PERCLOS <= 0.35
fatigue: PERCLOS >= 0.70
discard middle region during binary training/evaluation
```

Alternative three-class setup:

```text
alert: PERCLOS <= 0.35
transition: 0.35 < PERCLOS < 0.70
fatigue: PERCLOS >= 0.70
```

The exact threshold should be reported clearly in the paper.

## Evaluation Protocols

Primary protocol for the paper:

```text
Leave-One-Subject-Out / Leave-One-Session-Out
```

For each target session:

```text
source = all other sessions with labels
target adaptation = unlabeled target windows
target evaluation = target labels used only for metrics
```

Secondary protocol:

```text
official temporal 5-fold split
```

Use this mainly as a sanity check, not as the main cross-subject claim.

## Baselines

Start with these baselines:

1. Source-only model, no adaptation.
2. Target normalization only.
3. Pseudo-label fine-tuning.
4. TENT-style entropy minimization.
5. SHOT-style source-free adaptation.
6. Label-shift-aware SFDA.
7. Temporal-consistency SFDA.

## Proposed Method

Working method name:

```text
FAT-SFDA: Fatigue-Aware Source-Free Domain Adaptation
```

Core components:

1. Train a source model on labeled source subjects.
2. Freeze or partially freeze the source classifier.
3. Adapt target feature extractor using unlabeled target windows.
4. Use confidence-aware pseudo-labeling.
5. Estimate target class prior to reduce label-shift collapse.
6. Add temporal consistency because fatigue changes gradually.

Candidate loss:

```text
L = L_entropy
  + lambda_pl * L_pseudo_label
  + lambda_temporal * L_temporal_smoothness
  + lambda_prior * L_class_prior
```

## Metrics

For classification:

```text
balanced accuracy
macro F1
AUROC
confusion matrix
```

For regression:

```text
RMSE
MAE
Pearson correlation
Spearman correlation
```

## First Implementation Milestones

1. Inspect `.mat` keys and array shapes.
2. Convert SEED-VIG features and labels into a processed table/array format.
3. Implement LOSO splits.
4. Train source-only baseline.
5. Add simple offline target adaptation.
6. Save per-target metrics as CSV.
7. Generate paper-ready tables and figures.

