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
    spec = importlib.util.spec_from_file_location("q72_final_temporal_validation", COMPONENT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def main() -> None:
    reproduction = json.loads((ROOT / "reports" / "reproduction_gate.json").read_text(encoding="utf-8"))
    if reproduction["terminal"] != "P1_INDEPENDENT_REPRODUCTION_PASS":
        raise RuntimeError("P1 reproduction gate must pass before temporal validation")

    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    FIGURE.mkdir(parents=True, exist_ok=True)
    module = load_component()
    module.INPUT_PATH = INPUT
    module.TOPOLOGY_PATH = TOPOLOGY
    module.REPORT_DIR = REPORT
    module.FIG_DIR = FIGURE
    module.CAL_END_YEAR = 2018
    module.INNER_TRAIN_END_YEAR = 2015
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
    module.SCENARIO_ID = "P1_fit_2006_2018_eval_2019_2022"
    module.write_readme = lambda *_args, **_kwargs: None

    started = datetime.now()
    module.main()
    ended = datetime.now()
    full_path = REPORT / "monthly_bayes_seasonal_hysteresis_prediction_vs_observed_2006_2022.parquet"
    full = pd.read_parquet(full_path)
    validation = full.loc[full["year"].between(2019, 2022)].copy()
    if validation.empty or not validation["actual"].gt(0).all():
        raise RuntimeError("Positive-flow 2019-2022 validation population gate failed")
    validation["evaluation_label"] = "HISTORICALLY_VIEWED_NON_PRISTINE_EVALUATION_PERIOD"
    validation["fit_years"] = "2006-2018"
    validation["inner_selection_years"] = "2006-2015_vs_2016-2018"
    validation.to_parquet(OUT / "validation_predictions_2019_2022.parquet", index=False)
    validation.to_csv(OUT / "validation_predictions_2019_2022.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "runtime": RUNTIME,
        "terminal": "FIT_2006_2018_PREDICT_2019_2022_COMPLETE",
        "fit_years": [2006, 2018],
        "inner_selection_train_years": [2006, 2015],
        "inner_selection_validation_years": [2016, 2018],
        "evaluation_years": [2019, 2022],
        "evaluation_label": "HISTORICALLY_VIEWED_NON_PRISTINE_EVALUATION_PERIOD",
        "validation_rows": int(len(validation)),
        "validation_stations": int(validation["q_site"].nunique()),
        "elapsed_seconds": (ended - started).total_seconds(),
        "input_sha256": sha256(INPUT),
        "component_sha256": sha256(COMPONENT),
        "validation_prediction_sha256": sha256(OUT / "validation_predictions_2019_2022.parquet"),
    }
    (OUT / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
