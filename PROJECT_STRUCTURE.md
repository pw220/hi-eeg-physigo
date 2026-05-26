# Project Structure

Active baseline files:

- `train_eegnet_source.py`: EEGNet source-only LOSO training, dry-run planning, evaluation, outputs, checkpoints, and manifests.
- `data/seedvig_dataset.py`: SEED-VIG raw EEG loading, segmentation, label processing, preprocessing helpers.
- `data/seedvig_integrity.py`: dataset and fold integrity reports.
- `models/eegnet.py`: EEGNet binary classifier.
- `utils/metrics.py`: classification metrics and probability diagnostics.
- `utils/seed.py`: reproducibility helpers.

Folders:

- `data/`: dataset loading and integrity report code. Source files under `data/` are tracked; raw and processed data are ignored.
- `models/`: neural network backbones.
- `utils/`: metrics, seed control, and shared helpers.
- `scripts/`: small utility scripts.
- `outputs/`: generated experiment outputs, ignored by git.
- `results/`: final paper-ready result tables or figures; currently ignored and can be selectively tracked later if needed.
- `docs/`: notes, plans, and experiment logs.
- `notebooks/`: optional Colab or analysis notebooks.
- `configs/`: optional configuration files.

Potentially unused or legacy:

- `eeg_physio_project/`: appears to be an old or placeholder folder.
- `configs/seedvig.yaml`: not used by the current CLI baseline.
- `scripts/inspect_seedvig.py`: utility script, not part of the main training path.
- `docs/experiment_plan_seedvig_sfda.md`: planning notes for later work; not part of the current source-only baseline.

Do not delete these automatically. Review them before cleanup.
