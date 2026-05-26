from pathlib import Path

import scipy.io as sio


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "raw" / "SEED-VIG"


def describe_mat(path: Path) -> None:
    data = sio.loadmat(path)
    print(f"\n{path}")
    for key, value in data.items():
        if key.startswith("__"):
            continue
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        print(f"  {key}: shape={shape}, dtype={dtype}")


def main() -> None:
    feature_dir = DATA_ROOT / "EEG_Feature_5Bands"
    label_dir = DATA_ROOT / "perclos_labels"

    feature_files = sorted(feature_dir.glob("*.mat"))
    if not feature_files:
        raise FileNotFoundError(f"No .mat files found in {feature_dir}")

    first_feature = feature_files[0]
    first_label = label_dir / first_feature.name

    print(f"Dataset root: {DATA_ROOT}")
    print(f"Feature files: {len(feature_files)}")
    print(f"Label files: {len(list(label_dir.glob('*.mat')))}")

    describe_mat(first_feature)
    if first_label.exists():
        describe_mat(first_label)
    else:
        print(f"\nMissing matching label file: {first_label}")


if __name__ == "__main__":
    main()

