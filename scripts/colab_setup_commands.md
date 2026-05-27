# Colab Setup Commands

Check GPU:

```bash
!nvidia-smi
```

Mount Google Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Clone the repository:

```bash
!git clone <your-repo-url>
%cd hi-eeg-physigo
```

Install dependencies:

```bash
!pip install -r requirements.txt
```

Dry run:

```bash
!python train_eegnet_source.py \
  --run-all-loso \
  --max-folds 2 \
  --dry-run \
  --label-mode threshold35 \
  --raw-data-dir /content/drive/MyDrive/SEED-VIG/Raw_Data \
  --label-dir /content/drive/MyDrive/SEED-VIG/perclos_labels \
  --output-dir /content/drive/MyDrive/EEG_outputs/seedvig_eegnet_source_only
```

One-fold GPU test:

```bash
!python train_eegnet_source.py \
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
!python train_eegnet_source.py \
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
