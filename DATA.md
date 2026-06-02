# Data Policy and Layout

SEED-VIG data must not be committed to GitHub. Raw EEG files, PERCLOS label files, extracted features, processed arrays, checkpoints, and experiment outputs should stay local or in private storage such as Google Drive.

Download SEED-VIG from the official source, then place the files manually.

Expected layout:

```text
SEED-VIG/
├── Raw_Data/
└── perclos_labels/
```

Local default layout used by this repository:

```text
data/raw/SEED-VIG/
├── Raw_Data/
└── perclos_labels/
```

Example Google Colab / Drive paths:

```text
/content/drive/MyDrive/SEED-VIG/Raw_Data
/content/drive/MyDrive/SEED-VIG/perclos_labels
```

Use these paths with:

```bash
python train_eegnet_source.py \
  --raw-data-dir /content/drive/MyDrive/SEED-VIG/Raw_Data \
  --label-dir /content/drive/MyDrive/SEED-VIG/perclos_labels
```

Processed local datasets are also ignored by Git. The current recommended
EEGDA-standard files are:

```text
data/processed/
├── seedvig/
│   ├── seedvig_8s_threshold35_min50_all_sessions.npz
│   └── seedvig_8s_strict035070_min50_one_session.npz
└── sadt/
    ├── sadt_balanced.npz
    └── sadt_unbalanced.npz
```

Do not commit these files. In Colab, place equivalent files under a private
Drive folder such as `/content/drive/MyDrive/EEG-Data/processed/`.

Source conversion files such as `sad-balance.mat` may exist locally, but the
package-facing training input should be the converted `.npz` file.
