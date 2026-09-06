"""Register the exact time-by-space folds before any Stage 5 prediction exists."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test/20260904_5"
STAGE4_SCRIPTS = ROOT / "5_Test/20260904_4/scripts"
sys.path.insert(0, str(STAGE4_SCRIPTS))

from temporal_common import FOLDS, observations  # noqa: E402


OBSERVATION_SOURCE = ROOT / "5_Test/20260824_18/outputs/tn_observations_audited.parquet"
POSITION_SOURCE = ROOT / "5_Test/20260824_12/outputs/tn_observations_primary_2016_2024.parquet"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def fold_frames(
    frame: pd.DataFrame,
    training_years: list[int],
    evaluation_year: int,
    holdout_type: str,
    holdout_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame.loc[frame.year.isin(training_years)].copy()
    test = frame.loc[frame.year.eq(evaluation_year)].copy()
    if holdout_type == "LORO":
        train = train.loc[~train.reach_id.astype(int).eq(holdout_id)].copy()
        test = test.loc[test.reach_id.astype(int).eq(holdout_id)].copy()
    elif holdout_type == "LOTO":
        train = train.loc[~train.terminal_tree_id.astype(int).eq(holdout_id)].copy()
        test = test.loc[test.terminal_tree_id.astype(int).eq(holdout_id)].copy()
    else:
        raise ValueError(holdout_type)
    return train, test


def main() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError("The registered sparrow environment is required")
    obs = observations()
    if obs.year.max() > 2024:
        raise RuntimeError("Stage 5 formal folds must not include 2025 TN")

    rows: list[dict[str, object]] = []
    leakage_checks: list[bool] = []
    for temporal_fold, (training_years_raw, evaluation_year) in FOLDS.items():
        training_years = list(map(int, training_years_raw))
        evaluation = obs.loc[obs.year.eq(int(evaluation_year))]
        for holdout_type, column in (("LORO", "reach_id"), ("LOTO", "terminal_tree_id")):
            for holdout_id_raw in sorted(evaluation[column].astype(int).unique()):
                holdout_id = int(holdout_id_raw)
                train, test = fold_frames(
                    obs, training_years, int(evaluation_year), holdout_type, holdout_id
                )
                if test.empty or train.empty:
                    raise RuntimeError(
                        f"Empty registered fold {temporal_fold} {holdout_type} {holdout_id}"
                    )
                if holdout_type == "LORO":
                    excluded = not train.reach_id.astype(int).eq(holdout_id).any()
                else:
                    excluded = not train.terminal_tree_id.astype(int).eq(holdout_id).any()
                leakage_checks.append(bool(excluded))
                if not excluded:
                    raise RuntimeError(
                        f"Held-out spatial block leaked into training: "
                        f"{temporal_fold} {holdout_type} {holdout_id}"
                    )
                fold_id = (
                    f"{temporal_fold}_{holdout_type}_"
                    f"{'R' if holdout_type == 'LORO' else 'T'}{holdout_id:03d}"
                )
                rows.append(
                    {
                        "fold_id": fold_id,
                        "temporal_fold": temporal_fold,
                        "holdout_type": holdout_type,
                        "holdout_id": str(holdout_id),
                        "training_years_json": json.dumps(training_years),
                        "evaluation_year": int(evaluation_year),
                        "train_rows": int(len(train)),
                        "train_stations": int(train.station_key.nunique()),
                        "train_reaches": int(train.reach_id.nunique()),
                        "train_trees": int(train.terminal_tree_id.nunique()),
                        "test_rows": int(len(test)),
                        "test_stations": int(test.station_key.nunique()),
                        "test_reaches": int(test.reach_id.nunique()),
                        "test_trees": int(test.terminal_tree_id.nunique()),
                    }
                )

    registry = pd.DataFrame(rows).sort_values(
        ["temporal_fold", "holdout_type", "holdout_id"]
    ).reset_index(drop=True)
    duplicate = registry.fold_id.duplicated().any()
    expected_counts: dict[str, dict[str, int]] = {}
    for fold, (_, evaluation_year) in FOLDS.items():
        evaluation = obs.loc[obs.year.eq(int(evaluation_year))]
        expected_counts[fold] = {
            "LORO": int(evaluation.reach_id.nunique()),
            "LOTO": int(evaluation.terminal_tree_id.nunique()),
        }
    actual_counts = {
        fold: {
            kind: int(
                registry.loc[
                    registry.temporal_fold.eq(fold)
                    & registry.holdout_type.eq(kind),
                    "fold_id",
                ].nunique()
            )
            for kind in ("LORO", "LOTO")
        }
        for fold in FOLDS
    }
    checks = {
        "fold_ids_unique": not bool(duplicate),
        "all_spatial_blocks_excluded_from_training": all(leakage_checks),
        "all_test_rows_nonempty": bool((registry.test_rows > 0).all()),
        "all_train_rows_nonempty": bool((registry.train_rows > 0).all()),
        "counts_match_evaluation_year_blocks": actual_counts == expected_counts,
        "formal_tn_ends_2024": int(obs.year.max()) == 2024,
        "no_2025_rows_in_registry": bool((registry.evaluation_year <= 2024).all()),
    }
    status = "PASS_NESTED_FOLD_REGISTRY" if all(checks.values()) else "FAIL_NESTED_FOLD_REGISTRY"
    output = RUN / "outputs/nested_fold_registry.parquet"
    atomic_parquet(registry, output)
    report = {
        "stage": "20260904_5",
        "status": status,
        "purpose": "exact time-by-space fold registration before predictions",
        "capacities_to_evaluate": ["H7", "H14", "H22"],
        "fold_count": int(len(registry)),
        "fold_counts": actual_counts,
        "checks": checks,
        "input_hashes": {
            str(OBSERVATION_SOURCE): sha256(OBSERVATION_SOURCE),
            str(POSITION_SOURCE): sha256(POSITION_SOURCE),
        },
        "output_sha256": sha256(output),
    }
    atomic_json(report, RUN / "reports/nested_fold_registry_audit.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if status.startswith("FAIL"):
        raise RuntimeError(status)


if __name__ == "__main__":
    main()
