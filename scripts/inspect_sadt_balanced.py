from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from droweeg.datasets.sadt_balanced import inspect


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the processed balanced SADT mini dataset")
    parser.add_argument("--path", default="data/sad-data.mat")
    parser.add_argument("--output", default=None, help="Optional CSV path for the top-level inspection summary")
    args = parser.parse_args()

    result = inspect(args.path)
    print("sadt_balanced_inspection")
    for key, value in result.items():
        if key == "per_subject_label_distribution":
            print("  per_subject_label_distribution")
            for subject, counts in value.items():
                print(f"    subject_{subject}={counts}")
        else:
            print(f"  {key}={value}")

    if args.output is not None:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        flat = {key: value for key, value in result.items() if key != "per_subject_label_distribution"}
        pd.DataFrame([flat]).to_csv(output, index=False)
        print(f"saved={output}")


if __name__ == "__main__":
    main()
