from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from droweeg.datasets.sadt_balanced import SADTBalancedDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the processed balanced SADT .mat file to DrowEEG-standard .npz."
    )
    parser.add_argument("--input-path", default="data/processed/sadt/sad-balance.mat")
    parser.add_argument("--output-path", default="data/processed/sadt/sadt_balanced.npz")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"SADT balanced input file not found: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite to replace it: {output_path}")

    dataset = SADTBalancedDataset(input_path).load().to_standard_dataset()
    metadata = dataset.get_metadata()
    print("sadt_balanced_conversion")
    print(f"  input={input_path}")
    print(f"  output={output_path}")
    print(f"  X_shape={dataset.X.shape}")
    print(f"  y_shape={dataset.y.shape}")
    print(f"  subjects={metadata['subjects']}")
    print(f"  label_distribution={metadata['label_distribution']}")
    dataset.save(output_path)
    print(f"Saved DrowEEG-standard dataset: {output_path}")


if __name__ == "__main__":
    main()
