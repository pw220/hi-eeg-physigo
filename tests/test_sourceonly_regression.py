from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import eegda

GOLDEN_PATH = ROOT / "tests" / "golden_sourceonly.json"
TOL = 1e-5


def test_sourceonly_matches_golden(tmp_path: Path) -> None:
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    toy = eegda.make_toy_dataset(**golden["config"]["toy"])
    data_path = tmp_path / "toy_sourceonly.npz"
    output_dir = tmp_path / "outputs"
    toy.save(data_path)

    run = golden["config"]["run"]
    results = eegda.run(
        data=str(data_path),
        model="eegnet",
        method="source_only",
        protocol="loso",
        run_all_loso=True,
        epochs=run["epochs"],
        batch_size=run["batch_size"],
        device="cpu",
        validation_mode=run["validation_mode"],
        checkpoint_policy=run["checkpoint_policy"],
        class_balance=run["class_balance"],
        loss_type=run["loss_type"],
        optimizer=run["optimizer"],
        lr=run["lr"],
        weight_decay=run["weight_decay"],
        seed=run["seed"],
        eegnet_temporal_kernel=run["eegnet_temporal_kernel"],
        eegnet_separable_kernel=run["eegnet_separable_kernel"],
        output_dir=str(output_dir),
        log_level="quiet",
    )

    df = results.to_dataframe().sort_values("target_subject")
    rows = df[golden["metric_keys"]].to_dict(orient="records")
    expected = golden["fold_metrics"]
    assert len(rows) == len(expected)
    for actual_row, expected_row in zip(rows, expected, strict=True):
        for key in golden["metric_keys"]:
            actual = actual_row[key]
            expected_value = expected_row[key]
            if isinstance(expected_value, float):
                assert abs(float(actual) - expected_value) <= TOL, (key, actual, expected_value)
            else:
                assert actual == expected_value, (key, actual, expected_value)
