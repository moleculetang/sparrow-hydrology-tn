from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pyarrow
import scipy

from runtime_guard import assert_sparrow_runtime


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md",
    "experiment_contract.md",
    "scenario_contract.json",
    "run_all.ps1",
    "inputs/parent_indata.parquet",
    "inputs/excluded_stations.csv",
    "inputs/topology/topology_edges.csv",
    "scripts/run_prior_semantics_repair.py",
    "scripts/run_2019_2022_validation.py",
    "scripts/gate_scenario.py",
    "scripts/verify_selected_determinism.py",
    "scripts/build_delivery.py",
    "scripts/components/q72_prior_semantics_component.py",
    "outputs/P1/q72_three_fold_oof_predictions.parquet",
    "outputs/fit_2006_2018_eval_2019_2022/validation_predictions_2019_2022.parquet",
    "reports/reproduction_gate.json",
    "reports/deterministic_reproduction_gate.json",
    "reports/terminal_gate.json",
]


def main() -> None:
    runtime = assert_sparrow_runtime()
    missing = [x for x in REQUIRED if not (ROOT / x).is_file()]
    links = []
    old_test_reference_hits = []
    for path in ROOT.rglob("*"):
        if path.is_symlink():
            links.append(str(path.relative_to(ROOT)))
        if path.name != "audit_standalone.py" and path.is_file() and path.suffix in {".py", ".ps1", ".md", ".json"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "20260813_52" in text or "20260813_53" in text:
                old_test_reference_hits.append(str(path.relative_to(ROOT)))
    panel = pd.read_parquet(ROOT / "inputs/parent_indata.parquet")
    oof = pd.read_parquet(ROOT / "outputs/P1/q72_three_fold_oof_predictions.parquet")
    external = pd.read_parquet(ROOT / "outputs/fit_2006_2018_eval_2019_2022/validation_predictions_2019_2022.parquet")
    result = {
        "terminal": "S111_STANDALONE_INDEPENDENCE_AUDIT_PASS" if not missing and not links and not old_test_reference_hits else "S111_STANDALONE_INDEPENDENCE_AUDIT_FAILURE",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "missing_required_assets": missing,
        "symbolic_links": links,
        "runtime_old_test_reference_hits": old_test_reference_hits,
        "input_panel": {"rows": int(len(panel)), "reaches": int(panel.comid.nunique()), "months_per_reach_min": int(panel.groupby("comid").size().min()), "months_per_reach_max": int(panel.groupby("comid").size().max())},
        "outputs": {"oof_rows": int(len(oof)), "oof_stations": int(oof.q_site.nunique()), "external_rows": int(len(external)), "external_stations": int(external.q_site.nunique())},
        "environment": {"runtime": runtime, "python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__, "pyarrow": pyarrow.__version__, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"), "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS")},
    }
    (ROOT / "reports/standalone_independence_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "reports/environment.json").write_text(json.dumps(result["environment"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["terminal"].endswith("FAILURE"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
