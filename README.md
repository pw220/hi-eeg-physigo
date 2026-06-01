# DrowEEG

DrowEEG is a lightweight research package for EEG-based drowsiness recognition. The current stable method is an EEGNet source-only LOSO baseline for SEED-VIG raw EEG and the processed balanced SADT mini dataset.

No TRACE, SFDA, Riemannian reference, pseudo-labeling, entropy minimization, or adaptation method is implemented in the current baseline.

The method boundary already reserves an `adaptation_protocol` argument for
future SFDA work. Current source-only runs use `adaptation_protocol="none"`.
Future transductive SFDA will mean adapting on a target subject's unlabeled EEG
and revealing target labels only inside final evaluation metrics.

## Package API

The recommended interface is path-based: users preprocess data once into a
DrowEEG-standard `.npz`, then pass that file to DrowEEG.

```python
import droweeg

print(droweeg.list_models())
print(droweeg.list_methods())

dataset = droweeg.load_dataset("data/processed/sadt/sadt_unbalanced.npz")
dataset.summary()
model = droweeg.model("eegnet", dataset=dataset)

results = droweeg.run(
    data="data/processed/sadt/sadt_unbalanced.npz",
    model="eegnet",
    method="source_only",
    protocol="loso",
    epochs=50,
    device="cuda",
    validation_mode="none",
    epoch_log_interval=10,
)
results.summary()
```

LOSO target selection uses stable fold subject indices. If raw subject IDs are
not continuous, DrowEEG maps them to `1..N` and keeps the original IDs in logs,
summary CSVs, and prediction CSVs.

```python
dataset = droweeg.load_dataset("my_dataset.npz")
print(dataset.get_subject_mapping())  # e.g. {1: "a", 2: "b", 3: "d", 4: "f"}
```

By default, `droweeg.run(...)` evaluates the full LOSO protocol. Run one fold
with `target_subject=1`, selected folds with `target_subjects=[1, 3]`, or
explicitly request every fold with `run_all_loso=True`. Selected folds and full
LOSO are mutually exclusive.

```python
results = droweeg.run(
    data="data/processed/sadt/sadt_unbalanced.npz",
    model="eegnet",
    method="source_only",
    protocol="loso",
    target_subjects=[1, 3, 5],
    target_id_space="canonical",
    epochs=50,
    device="cuda",
    validation_mode="none",
)
```

You can also select by raw subject IDs:

```python
dataset = droweeg.load_dataset("data/processed/sadt/sadt_unbalanced.npz")
dataset.summary()

results = droweeg.run(
    dataset=dataset,
    model="eegnet",
    method="source_only",
    protocol="loso",
    target_subjects=[1, 22, 35],
    target_id_space="raw",
    epochs=50,
    device="cuda",
    log_level="normal",
    epoch_log_interval=10,
    output_dir="./outputs",
)

df = results.to_dataframe()
print(df)
```

Console logging is controlled with `log_level` / `--log-level`:

- `quiet`: final aggregate summary and output directory only.
- `normal`: compact run overview, fold progress, epoch table at epoch 1, every `epoch_log_interval` epochs, and final epoch, fold result, final summary.
- `verbose`: adds split audit and protocol details.
- `debug`: adds full reproducibility, preprocessing checks, raw commands, and saved paths.

Detailed run information is saved under the run directory, including
`run_config.json`, `reproducibility.json`, `model_selection_policy.json`,
`split_audit/`, `metrics/`, `predictions/`, `checkpoints/`, and
`artifacts.json`.

Create a small synthetic example dataset:

```python
toy = droweeg.make_toy_dataset(n_subjects=4, samples_per_subject=20)
toy.save("toy_droweeg.npz")
print(droweeg.load_dataset("toy_droweeg.npz").get_metadata())
```

Current public model/method names:

- models: `eegnet`
- methods: `source_only`

Official adapters such as `seedvig` and `sadt-balanced` are provided for
conversion and backward compatibility, but the recommended training input is a
standard `.npz` file. Advanced users can register custom components with
`droweeg.register_model(...)`, `droweeg.register_dataset(...)`, and
`droweeg.register_method(...)`. See `docs/custom_model.md`.

## Raw Data vs DrowEEG Standard Format

DrowEEG does not aim to parse every raw EEG format. Different labs store the same EEG dataset in different raw layouts, so general raw-format support would make the package unstable.

The recommended route for every dataset, including a user's private dataset, is:

1. Preprocess your EEG into windowed arrays.
2. Store samples as `X` with shape `(N, C, T)`.
3. Provide labels `y` and subject IDs `subjects`.
4. Use `Dataset.from_arrays(...)` or save a reusable standard `.npz` file.

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

dataset = droweeg.load_dataset("my_dataset.npz")
model = droweeg.model("eegnet", dataset=dataset)
```

Train from the file:

```bash
python -m droweeg.train \
  --data my_dataset.npz \
  --model eegnet \
  --method source_only \
  --protocol loso \
  --run-all-loso \
  --epochs 50 \
  --device cuda
```

See `docs/standard_dataset_format.md`.

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

SADT-balanced can be converted to a standard `.npz` file. The source `.mat`
should also stay out of Git. This is not the raw/continuous SADT `.set` dataset.

```bash
python scripts/convert_sadt_balanced_standard.py \
  --input-path data/processed/sadt/sad-balance.mat \
  --output-path data/processed/sadt/sadt_balanced.npz \
  --overwrite
```

## Building SEED-VIG Standard Cached Datasets

DrowEEG does not bundle SEED-VIG data. Provide your local `Raw_Data` and `perclos_labels` folders, then build reusable DrowEEG-standard `.npz` caches. The cache contains fixed EEG windows, labels, subjects, sessions, sample IDs, PERCLOS values, and metadata only.

The cache builder does not apply global normalization, robust clipping, class balancing, class weights, or train/test splitting. Fold-specific preprocessing remains inside training so normalization and optional clipping are computed from source-training samples only.

Main threshold35 cache. This keeps intermediate PERCLOS samples by using
`PERCLOS > 0.35` as the fatigue/reduced-vigilance class, then excludes subjects
with fewer than 50 samples in either class after aggregating all sessions:

```bash
python scripts/build_seedvig_standard.py \
  --raw-data-dir /content/drive/MyDrive/EEG-Data/SEED_VIG/Raw_Data \
  --label-dir /content/drive/MyDrive/EEG-Data/SEED_VIG/perclos_labels \
  --label-mode threshold35 \
  --filter-level subject \
  --min-samples-per-class 50 \
  --session-policy all_valid \
  --balance-mode none \
  --output-path /content/drive/MyDrive/EEG-Data/processed/seedvig/seedvig_8s_threshold35_min50_all_sessions.npz \
  --overwrite
```

Strict cache. This discards intermediate PERCLOS samples, applies the same
subject-level minimum class count, and keeps only the most class-balanced valid
session when a subject has multiple sessions:

```bash
python scripts/build_seedvig_standard.py \
  --raw-data-dir /content/drive/MyDrive/EEG-Data/SEED_VIG/Raw_Data \
  --label-dir /content/drive/MyDrive/EEG-Data/SEED_VIG/perclos_labels \
  --label-mode strict035070 \
  --alert-threshold 0.35 \
  --fatigue-threshold 0.70 \
  --filter-level subject \
  --min-samples-per-class 50 \
  --session-policy one_most_balanced \
  --balance-mode none \
  --output-path /content/drive/MyDrive/EEG-Data/processed/seedvig/seedvig_8s_strict035070_min50_one_session.npz \
  --overwrite
```

Train from a cached file:

```bash
python -m droweeg.train \
  --data /content/drive/MyDrive/EEG-Data/processed/seedvig/seedvig_8s_threshold35_min50_all_sessions.npz \
  --model eegnet \
  --method source_only \
  --protocol loso \
  --run-all-loso \
  --epochs 50 \
  --device cuda \
  --validation-mode none \
  --checkpoint-policy last \
  --disable-early-stop
```

## Building SADT RT-Labelled Unbalanced Dataset

DrowEEG does not bundle SADT data. Provide the official preprocessed continuous EEGLAB `.set/.fdt` sessions. The builder extracts 3-second pre-deviation EEG epochs, downsamples each epoch to 128 Hz, labels trials from local/global reaction time, discards transition trials, filters sessions with too few samples in either class, and selects the most balanced valid session per subject by default.

The final dataset is not forced to be class-balanced. Global normalization, robust clipping, and class weighting are not applied during cache construction; they remain fold-specific inside training.

Natural RT-labelled cache:

```bash
python scripts/build_sadt_rt_standard.py \
  --input-dir /content/drive/MyDrive/EEG-Data/SADT_preprocessed \
  --output-path /content/drive/MyDrive/EEG-Data/processed/sadt/sadt_rt_unbalanced.npz \
  --sfreq-out 128 \
  --epoch-seconds 3 \
  --rt-cleaning range \
  --rt-min-sec 0.30 \
  --rt-max-sec 10.0 \
  --global-rt-window-sec 90 \
  --min-samples-per-class 50 \
  --session-policy one_most_balanced \
  --balance-mode none \
  --overwrite
```

ICNN-compatible RT-labelled cache:

This setting is intended for comparison with the ICNN paper. It keeps the
paper's session-level minimum class count and one-session-per-subject protocol,
and uses explicit RT cleaning plus the global-RT/session-selection settings that
recover the reported 11-subject scale on the local SADT files. It is
ICNN-compatible, not an exact reproduction of the authors' released processed
table.

```bash
python scripts/build_sadt_rt_standard.py \
  --input-dir data/sadt-raw \
  --output-path data/processed/sadt/sadt_unbalanced.npz \
  --sfreq-out 128 \
  --epoch-seconds 3 \
  --rt-cleaning range \
  --rt-min-sec 0.30 \
  --rt-max-sec 12.0 \
  --global-rt-mode include_current_window \
  --global-rt-window-sec 90 \
  --min-samples-per-class 50 \
  --session-policy one_most_balanced \
  --subject-session-selection largest_total \
  --balance-mode none \
  --overwrite
```

Training from the cache:

```bash
python -m droweeg.train \
  --data data/processed/sadt/sadt_unbalanced.npz \
  --model eegnet \
  --method source_only \
  --protocol loso \
  --run-all-loso \
  --epochs 50 \
  --device cuda \
  --validation-mode none \
  --checkpoint-policy last \
  --disable-early-stop
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

## Running Standard Datasets And Models

For now, `eegnet` is the only supported model and `source_only` is the only supported method. New DrowEEG commands use `python -m droweeg.train`.

SEED-VIG cached-file example:

```bash
python -m droweeg.train \
  --data data/processed/seedvig/seedvig_8s_threshold35_min50_all_sessions.npz \
  --model eegnet \
  --method source_only \
  --protocol loso \
  --target-subject 1 \
  --epochs 2 \
  --batch-size 64 \
  --device cpu \
  --class-balance weighted_loss
```

SADT-balanced example:

```bash
python -m droweeg.train \
  --data data/processed/sadt/sadt_balanced.npz \
  --model eegnet \
  --method source_only \
  --protocol loso \
  --target-subject 1 \
  --epochs 2 \
  --batch-size 64 \
  --device cpu \
  --validation-mode sample_stratified \
  --class-balance weighted_loss
```

SADT unbalanced full LOSO GPU example:

```bash
python -m droweeg.train \
  --data data/processed/sadt/sadt_unbalanced.npz \
  --model eegnet \
  --method source_only \
  --protocol loso \
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
  --data data/processed/sadt/sadt_unbalanced.npz \
  --model eegnet \
  --method source_only \
  --protocol loso \
  --run-all-loso \
  --dry-run
```

The old `train_eegnet_source.py` commands remain available for backward compatibility.

### No-Validation Runs And Target Diagnostics

`--validation-mode none` trains on all non-target source subjects and selects a checkpoint only from the declared `--checkpoint-policy`, usually `last` or `fixed_epoch`. If `--test-every-epochs N` is enabled, DrowEEG periodically evaluates the held-out target subject for diagnostics, but those target metrics are audit-only. They must not be used for early stopping, checkpoint selection, hyperparameter tuning, or model selection.

For package users, prefer `fixed_epoch` when the epoch is pre-declared, or use `subject_split` / `sample_stratified` validation when you need source-only model selection. The reported "best target diagnostic" epoch is useful for analysis plots, not for choosing the final model.

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
