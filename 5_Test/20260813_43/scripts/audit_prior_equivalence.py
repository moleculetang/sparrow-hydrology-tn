from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
N_STATIONS = 111
SIGMA_STATION = 1.0
SIGMA_SLOPE = 0.15
SIGMA_REGIME = 0.25
SIGMA_FIXED = 3.0
SIGMA_GROUP = 1.5
SIGMA_MULTISTORE = 0.30


def main() -> None:
    rows = []

    # Legacy: u_i iid N(0,sigma^2). Repaired W1: n-1 independent columns,
    # last site's effect is minus their sum.
    rows.append({
        "block": "station_intercept_or_random_slope",
        "contrast": "two non-reference stations",
        "legacy_variance_factor": 2.0,
        "W1_variance_factor": 2.0,
        "relative_difference": 0.0,
    })
    rows.append({
        "block": "station_intercept_or_random_slope",
        "contrast": "non-reference minus last station",
        "legacy_variance_factor": 2.0,
        "W1_variance_factor": float(N_STATIONS + 2),
        "relative_difference": float((N_STATIONS + 2) / 2 - 1),
    })

    # Legacy regime prediction in each gate is b+r_g. W1 makes dry-low b.
    legacy_regime_var = SIGMA_SLOPE**2 + SIGMA_REGIME**2
    w1_reference_var = SIGMA_SLOPE**2
    rows.append({
        "block": "regime_station_slope",
        "contrast": "dry_low marginal prediction coefficient",
        "legacy_variance_factor": legacy_regime_var,
        "W1_variance_factor": w1_reference_var,
        "relative_difference": w1_reference_var / legacy_regime_var - 1,
    })

    # Spatial base plus group deviation has the same reference-class issue.
    legacy_group_var = SIGMA_FIXED**2 + SIGMA_GROUP**2
    w1_mid_var = SIGMA_FIXED**2
    rows.append({
        "block": "spatial_group_slope",
        "contrast": "mid_area marginal prediction coefficient",
        "legacy_variance_factor": legacy_group_var,
        "W1_variance_factor": w1_mid_var,
        "relative_difference": w1_mid_var / legacy_group_var - 1,
    })

    # Exact duplicate process columns: z*b1 + z*b2 has variance sum of the
    # independent coefficient variances. Deleting one without variance folding
    # necessarily changes the prior predictive covariance.
    rows.append({
        "block": "exact_duplicate_multistore_feature",
        "contrast": "duplicate pair total coefficient",
        "legacy_variance_factor": 2 * SIGMA_MULTISTORE**2,
        "W1_variance_factor": SIGMA_MULTISTORE**2,
        "relative_difference": -0.5,
    })

    audit = pd.DataFrame(rows)
    audit["equivalent_at_1e_12"] = audit["relative_difference"].abs() <= 1e-12
    audit.to_csv(ROOT / "reports" / "prior_predictive_equivalence.csv", index=False, encoding="utf-8-sig")
    terminal = {
        "terminal": "REGRESSION_REPARAMETERIZATION_EQUIVALENCE_FAILURE",
        "model_run": False,
        "station_count": N_STATIONS,
        "all_checked_blocks_equivalent": bool(audit["equivalent_at_1e_12"].all()),
        "failed_blocks": audit.loc[~audit["equivalent_at_1e_12"], "block"].tolist(),
        "required_repair": "restore zero-preserving gates with the full original Gaussian-prior column space, then rerun all folds",
    }
    (ROOT / "terminal_gate.json").write_text(json.dumps(terminal, ensure_ascii=False, indent=2), encoding="utf-8")
    print(terminal["terminal"])


if __name__ == "__main__":
    main()
