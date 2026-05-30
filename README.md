# DrowEEG

DrowEEG is a lightweight research package for EEG-based drowsiness recognition. The current stable method is an EEGNet source-only LOSO baseline for SEED-VIG raw EEG and the processed balanced SADT mini dataset.

No TRACE, SFDA, Riemannian reference, pseudo-labeling, entropy minimization, or adaptation method is implemented in the current baseline.

## Package API

The recommended new interface is:

```python
import droweeg

print(droweeg.list_datasets())
print(droweeg.list_models())
print(droweeg.list_methods())

model = droweeg.model("eegnet", channels=17, samples=1600, num_classes=2)
dataset = droweeg.dataset("sadt-balanced", path="data/sad-data.mat")

results = droweeg.run(
    dataset="sadt-balanced",
    model="eegnet",
    method="source_only",
    protocol="loso",
    sadt_balanced_path="data/sad-data.mat",
    run_all_loso=True,
    epochs=50,
    device="cuda",
)
```

Current registries:

- datasets: `seedvig`, `sadt-balanced`, `standard-npz`
- models: `eegnet`
- methods: `source_only`

Advanced users can register custom components with `droweeg.register_model(...)`, `droweeg.register_dataset(...)`, and `droweeg.register_method(...)`. See `docs/custom_model.md`.

## Raw Data vs DrowEEG Standard Format

DrowEEG does not aim to parse every raw EEG format. Different labs store the same EEG dataset in different raw layouts, so general raw-format support would make the package unstable.

The recommended custom-data route is:

1. Preprocess your EEG into windowed arrays.
2. Store samples as `X` with shape `(N, C, T)`.
3. Provide labels `y` and subject IDs `subjects`.
4. Use `Dataset.from_arrays(...)` or save a reusable `standard-npz` file.

Example:

```python
dataset = droweeg.Dataset.from_arrays(
    X=X,
    y=y,
    subjects=subjects,
    sfreq=128,
    label_names={0: "alert", 1: "fatigue"},
)

droweeg.save_standard_dataset(
    "my_dataset.npz",
    X=X,
    y=y,
    subjects=subjects,
    sfreq=128,
    label_names={0: "alert", 1: "fatigue"},
)

dataset = droweeg.dataset("standard-npz", path="my_dataset.npz")
```

Official adapters such as `seedvig` and `sadt-balanced` exist for selected public datasets and convert their known formats into the same internal standard representation. See `docs/standard_dataset_format.md`.

## Current Working Pipeline

- Loads SEED-VIG raw `.mat` EEG sessions.
- Verifies 200 Hz sampling rate and 17 EEG channels.
- Segments each session into non-overlapping 8-second windows with shape `(17, 1600)`.
- Runs subject-wise LOSO training with source-subject validation only.
- Trains EEGNet with binary output logits.
- Saves per-sample predictions, summary metrics, integrity reports, checkpoints, and a checkpoint manifest.

Active files:

- `python -m droweeg.train`
- `train_eegnet_source.py`
- `data/seedvig_dataset.py`
- `data/seedvig_integrity.py`
- `models/eegnet.py`
- `utils/metrics.py`
- `utils/seed.py`

## Dataset Placement

Do not commit SEED-VIG data to GitHub. See `DATA.md`.

Default local layout:

```text
data/raw/SEED-VIG/
├── Raw_Data/
└── perclos_labels/
```

Colab / Google Drive example:

```text
/content/drive/MyDrive/SEED-VIG/Raw_Data
/content/drive/MyDrive/SEED-VIG/perclos_labels
```

SADT-balanced is a processed balanced `.mat` mini dataset and should also stay out of Git. This is not the raw/continuous SADT `.set` dataset. The default local path is:

```text
data/sad-data.mat
```

## Label Modes

- `threshold35` default: PERCLOS `<= 0.35` is alert class `0`; PERCLOS `> 0.35` is fatigue class `1`; no intermediate samples are discarded.
- `strict035070`: PERCLOS `< 0.35` is alert class `0`; PERCLOS `> 0.70` is fatigue class `1`; `0.35 <= PERCLOS <= 0.70` is discarded.

## Local CPU Smoke Tests

Dry run:

```bash
python train_eegnet_source.py --run-all-loso --max-folds 2 --dry-run --label-mode threshold35
```

One-fold CPU smoke test:

```bash
python train_eegnet_source.py --target-subject 1 --epochs 1 --batch-size 64 --device cpu --label-mode threshold35 --class-balance weighted_loss
```

## Running Different Datasets And Models

For now, `eegnet` is the only supported model and `source_only` is the only supported method. New DrowEEG commands use `python -m droweeg.train`.

SEED-VIG example:

```bash
python -m droweeg.train \
  --dataset seedvig \
  --model eegnet \
  --method source_only \
  --protocol loso \
  --target-subject 1 \
  --epochs 2 \
  --batch-size 64 \
  --device cpu \
  --label-mode threshold35 \
  --class-balance weighted_loss
```

SADT-balanced example:

```bash
python -m droweeg.train \
  --dataset sadt-balanced \
  --model eegnet \
  --method source_only \
  --protocol loso \
  --sadt-balanced-path data/sad-data.mat \
  --target-subject 1 \
  --epochs 2 \
  --batch-size 64 \
  --device cpu \
  --validation-mode sample_stratified \
  --class-balance weighted_loss
```

SADT-balanced full LOSO GPU example:

```bash
python -m droweeg.train \
  --dataset sadt-balanced \
  --model eegnet \
  --method source_only \
  --protocol loso \
  --sadt-balanced-path data/sad-data.mat \
  --run-all-loso \
  --epochs 50 \
  --batch-size 64 \
  --device cuda \
  --validation-mode none \
  --checkpoint-policy last \
  --class-balance weighted_loss
```

Dry run:

```bash
python -m droweeg.train \
  --dataset sadt-balanced \
  --model eegnet \
  --method source_only \
  --protocol loso \
  --run-all-loso \
  --dry-run
```

The old `train_eegnet_source.py` commands remain available for backward compatibility.

Configurable one-fold training example. The backbone remains the faithful ARL EEGNet-8,2 port; pooling is fixed at `(1, 4)` then `(1, 8)` to match the original architecture.

```bash
python train_eegnet_source.py \
  --target-subject 1 \
  --epochs 50 \
  --batch-size 64 \
  --lr 1e-3 \
  --optimizer adamw \
  --weight-decay 1e-4 \
  --early-stop-patience 15 \
  --monitor-metric macro_f1 \
  --lr-scheduler plateau \
  --eegnet-f1 8 \
  --eegnet-d 2 \
  --eegnet-f2 0 \
  --eegnet-temporal-kernel 64 \
  --eegnet-separable-kernel 16 \
  --eegnet-pool1 4 \
  --eegnet-pool2 8 \
  --eegnet-dropout 0.5 \
  --eegnet-norm-rate 0.25 \
  --device cuda \
  --label-mode threshold35 \
  --class-balance weighted_loss
```

Do not run full LOSO or long training locally unless you have suitable hardware.

## Full GPU LOSO Command

```bash
python train_eegnet_source.py --run-all-loso --epochs 100 --batch-size 64 --device cuda --label-mode threshold35 --class-balance weighted_loss --optimizer adamw --weight-decay 0.0001 --early-stop-patience 15 --monitor-metric macro_f1 --lr-scheduler plateau
```

## Google Colab Usage

One-fold GPU test:

```bash
python train_eegnet_source.py \
  --target-subject 1 \
  --epochs 5 \
  --batch-size 64 \
  --optimizer adamw \
  --weight-decay 1e-4 \
  --early-stop-patience 15 \
  --monitor-metric macro_f1 \
  --device cuda \
  --label-mode threshold35 \
  --class-balance weighted_loss \
  --raw-data-dir /content/drive/MyDrive/SEED-VIG/Raw_Data \
  --label-dir /content/drive/MyDrive/SEED-VIG/perclos_labels \
  --output-dir /content/drive/MyDrive/EEG_outputs/seedvig_eegnet_source_only
```

Full LOSO GPU run:

```bash
python train_eegnet_source.py \
  --run-all-loso \
  --epochs 100 \
  --batch-size 64 \
  --optimizer adamw \
  --weight-decay 1e-4 \
  --early-stop-patience 15 \
  --monitor-metric macro_f1 \
  --lr-scheduler plateau \
  --device cuda \
  --label-mode threshold35 \
  --class-balance weighted_loss \
  --raw-data-dir /content/drive/MyDrive/SEED-VIG/Raw_Data \
  --label-dir /content/drive/MyDrive/SEED-VIG/perclos_labels \
  --output-dir /content/drive/MyDrive/EEG_outputs/seedvig_eegnet_source_only \
  --skip-existing
```

More Colab command snippets are in `scripts/colab_setup_commands.md`.

## Output Files

Outputs are written under `--output-dir`, default `outputs/`.

- `eegnet_source_only_{label_mode}_subject_{subject_id}.csv`: per-sample target predictions.
- `eegnet_source_only_{label_mode}_summary.csv`: one summary row per label mode, target subject, seed, and class-balance mode.
- `checkpoints/eegnet_source_only_{label_mode}_subject_{subject_id}_seed{seed}_{run_id}.pt`: unique checkpoint.
- `checkpoints_manifest.csv`: checkpoint and run manifest.
- `seedvig_integrity_{label_mode}.csv`: dataset integrity report.
- `loso_fold_integrity_{label_mode}_subject_{subject_id}.txt`: selected fold integrity report.

Use `--skip-existing` to skip completed folds in all-LOSO mode. Use `--overwrite` only when you intentionally want to replace an existing run ID or latest output.

Inspect a checkpoint:

```bash
python scripts/inspect_checkpoint.py --checkpoint path/to/model.pt
```

## Metrics

For binary fatigue detection, class `0` is alert and class `1` is fatigue/drowsy. Fatigue is always treated as the positive class.

Reported metrics include accuracy, balanced accuracy, macro precision, macro recall, macro F1, weighted F1, fatigue precision, fatigue recall, fatigue F1, alert precision, alert recall, alert F1, ROC-AUC, AUPRC, and the stable confusion matrix values `tn`, `fp`, `fn`, `tp`.

Terminology:

- `sensitivity` is fatigue recall: `TP / (TP + FN)`.
- `specificity` is alert recall: `TN / (TN + FP)`.
- `miss_rate` is `1 - sensitivity`.
- `majority_accuracy` is the accuracy of always predicting the majority class in that target subject.

Accuracy is reported for comparison with prior fatigue-detection studies. For SEED-VIG `threshold35`, class distributions can be strongly imbalanced, so balanced accuracy and macro F1 are the primary metrics for comparing source-only LOSO performance.

Overall LOSO metrics are aggregated as subject-wise mean and standard deviation across completed target-subject folds, not pooled sample-level accuracy.

## Leakage-Prevention Rules

- Target labels are never used during training, validation, normalization, clipping, class weighting, early stopping, threshold selection, or model selection.
- Target labels are used only for final evaluation and diagnostic prediction CSVs.
- Validation is split only from source subjects.
- Class weights are computed only from source-training labels.
- Normalization and robust clipping statistics are computed only from source-training EEG windows.

## GitHub and Data Policy

Do not commit raw datasets, label files, processed arrays, checkpoints, prediction CSVs, experiment outputs, or large binary files.

Manual GitHub commands:

```bash
git status
git add .
git commit -m "Prepare EEGNet source-only baseline repository"
```

If using GitHub CLI later:

```bash
gh auth login
gh repo create hi-eeg-physigo --private --source=. --remote=origin --push
```

If creating a GitHub repository manually:

1. Create an empty private repository on GitHub.
2. Run:

```bash
git remote add origin <repo-url>
git branch -M main
git push -u origin main
```
