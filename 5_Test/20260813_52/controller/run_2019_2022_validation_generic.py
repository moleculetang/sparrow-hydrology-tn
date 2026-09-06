from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from runtime_guard import assert_sparrow_runtime


RUNTIME = assert_sparrow_runtime()
ROOT = Path(__file__).resolve().parents[1]
COMPONENT = ROOT / "scripts" / "components" / "q72_prior_semantics_component.py"
INPUT = ROOT / "inputs" / "parent_indata.parquet"
TOPOLOGY = ROOT / "inputs" / "topology" / "topology_edges.csv"
OUT = ROOT / "outputs" / "fit_2006_2018_eval_2019_2022"
REPORT = OUT / "model_reports"
FIGURE = OUT / "model_figures"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_component():
    spec = importlib.util.spec_from_file_location("q72_station_sensitivity", COMPONENT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def main() -> None:
    gate = json.loads((ROOT / "reports" / "reproduction_gate.json").read_text(encoding="utf-8"))
    if gate["terminal"] != "FILTERED_INPUT_AND_OOF_POPULATION_PASS":
        raise RuntimeError("OOF population gate must pass first")
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    FIGURE.mkdir(parents=True, exist_ok=True)
    module = load_component()
    module.INPUT_PATH, module.TOPOLOGY_PATH = INPUT, TOPOLOGY
    module.REPORT_DIR, module.FIG_DIR = REPORT, FIGURE
    module.CAL_END_YEAR, module.INNER_TRAIN_END_YEAR = 2018, 2015
    module.STATE_CALENDAR_MODE = "full_forcing"
    module.FORCING_SEMANTICS_MODE = "prescribed_aet_balance"
    module.MASS_ACCOUNTING_MODE = "explicit_upstream_volume"
    module.DYNAMIC_BETA_W = 0.0
    module.configure_engineering_repair(True)
    module.configure_full_gaussian_prior_space(True)
    module.ZERO_PRESERVING_FULL_PRIOR_MODE = True
    module.set_et_feature_block_mode("full")
    module.NETWORK_INPUT_SCALE = 1.0
    module.NETWORK_INPUT_SEMANTICS = "d8_reach193_periodic_spinup_zero_gate_full_gaussian_prior_space"
    module.DETERMINISTIC_SPINUP_MODE = True
    module.SCENARIO_ID = ROOT.name
    module.write_readme = lambda *_args, **_kwargs: None
    started = datetime.now()
    module.main()
    full = pd.read_parquet(REPORT / "monthly_bayes_seasonal_hysteresis_prediction_vs_observed_2006_2022.parquet")
    validation = full.loc[full.year.between(2019, 2022)].copy()
    excluded = set(pd.read_csv(ROOT / "inputs" / "excluded_stations.csv", encoding="utf-8-sig").q_site.astype(str))
    if validation.empty or validation.q_site.astype(str).isin(excluded).any() or not validation.actual.gt(0).all():
        raise RuntimeError("2019-2022 validation population gate failed")
    validation["evaluation_label"] = "POST_HOC_STATION_COMBINATION_SENSITIVITY"
    validation.to_parquet(OUT / "validation_predictions_2019_2022.parquet", index=False)
    manifest = {
        "terminal": "FIT_2006_2018_PREDICT_2019_2022_COMPLETE",
        "runtime": RUNTIME,
        "scenario_id": ROOT.name,
        "validation_rows": len(validation),
        "validation_stations": int(validation.q_site.nunique()),
        "elapsed_seconds": (datetime.now() - started).total_seconds(),
        "input_sha256": sha256(INPUT),
        "prediction_sha256": sha256(OUT / "validation_predictions_2019_2022.parquet"),
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
