# DrowEEG Standard Dataset Format

DrowEEG does not try to parse every raw EEG format. Users should preprocess their data into windowed EEG arrays once, then train through DrowEEG.

## Required Data

- `X`: EEG samples as `float32`, shape `(N, C, T)`.
- `subjects`: subject IDs, shape `(N,)`.
- `y`: integer labels, shape `(N,)`, required for supervised source-only training. Unlabeled arrays are reserved for future target-adaptation workflows.

Shape convention:

- `N`: samples/windows.
- `C`: EEG channels.
- `T`: time samples.
- EEGNet receives `(batch, 1, C, T)`.

For binary drowsiness recognition:

- `0`: alert.
- `1`: fatigue/drowsy.

## Optional Data

- `sessions`: session IDs, shape `(N,)`.
- `sample_ids`: sample IDs, shape `(N,)`.
- `sfreq`: sampling rate.
- `channel_names`: list of channel names.
- `label_names`: mapping such as `{0: "alert", 1: "fatigue"}`.
- `metadata`: dictionary with dataset notes.

## Python Arrays

```python
import numpy as np
import droweeg

X = np.random.randn(20, 30, 384).astype("float32")
y = np.array([0, 1] * 10, dtype="int64")
subjects = np.repeat(np.arange(1, 6), 4)

dataset = droweeg.Dataset.from_arrays(
    X=X,
    y=y,
    subjects=subjects,
    sfreq=128,
    label_names={0: "alert", 1: "fatigue"},
)

model = droweeg.model("eegnet", dataset=dataset)
```

## Save Once, Reuse Later

```python
droweeg.save_standard_dataset(
    "my_dataset.npz",
    X=X,
    y=y,
    subjects=subjects,
    sfreq=128,
    label_names={0: "alert", 1: "fatigue"},
)
```

Then load it:

```python
dataset = droweeg.dataset("standard-npz", path="my_dataset.npz")
print(dataset.get_metadata())
```

## Train With Standard NPZ

```bash
python -m droweeg.train \
  --dataset standard-npz \
  --standard-npz-path my_dataset.npz \
  --model eegnet \
  --method source_only \
  --protocol loso \
  --run-all-loso \
  --epochs 50 \
  --device cuda
```

Current source-only metrics assume binary labels. Future source-free adaptation workflows may use unlabeled target arrays.

## Official Adapters

Official adapters such as `seedvig` and `sadt-balanced` convert known public dataset layouts into this same standard representation:

```python
sadt = droweeg.dataset("sadt-balanced", path="data/sad-data.mat")
standard = sadt.to_standard_dataset()
print(standard.get_metadata())
```

SEED-VIG can also be converted, but it loads and windows the raw continuous EEG and can be memory-heavy:

```python
seedvig = droweeg.dataset(
    "seedvig",
    raw_data_dir="data/raw/SEED-VIG/Raw_Data",
    label_dir="data/raw/SEED-VIG/perclos_labels",
    label_mode="threshold35",
)
standard = seedvig.to_standard_dataset()
```

For large SEED-VIG experiments, the compatibility trainer may still stream through the official adapter path rather than materializing a full `.npz` first. The conceptual interface remains the same: official adapters produce `(X, y, subjects, ...)` before model training.
