from __future__ import annotations

import contextlib
import io
import json
import traceback
from pathlib import Path

from runtime_guard import assert_sparrow_runtime


RUNTIME = assert_sparrow_runtime()
RUN = Path(__file__).resolve().parents[1]
OUT = RUN / "notebooks" / "four_scenario_diagnostics.ipynb"
VALIDATION = RUN / "notebooks" / "notebook_execution_validation.json"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


def execute_cells(cells: list[dict]) -> tuple[bool, list[dict]]:
    namespace: dict = {"__name__": "__notebook__"}
    execution_count = 0
    errors: list[dict] = []
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        execution_count += 1
        stream = io.StringIO()
        try:
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                exec(compile("".join(cell["source"]), f"notebook-cell-{index}", "exec"), namespace, namespace)
            cell["execution_count"] = execution_count
            output = stream.getvalue()
            if output:
                cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": output.splitlines(keepends=True)}]
        except Exception as exc:  # pragma: no cover - audit path
            cell["execution_count"] = execution_count
            trace = traceback.format_exc()
            cell["outputs"] = [{
                "output_type": "error",
                "ename": type(exc).__name__,
                "evalue": str(exc),
                "traceback": trace.splitlines(),
            }]
            errors.append({"cell_index": index, "error": repr(exc), "traceback": trace})
            break
    return not errors, errors


def main() -> None:
    cells = [
        markdown(
            "# Q72 conflict-overlay and state-calendar diagnostics\n\n"
            "Companion audit notebook for the four fixed scenarios in `20260810_4`.\n"
        ),
        markdown("## tl;dr\n"),
        code(
            "from pathlib import Path\n"
            "import json\n"
            "import pandas as pd\n"
            "import numpy as np\n\n"
            f"RUN = Path(r'{RUN}')\n"
            "REPORT = RUN / 'reports' / 'scenario_diagnostics'\n"
            "summary = json.loads((REPORT / 'diagnostics_summary.json').read_text(encoding='utf-8'))\n"
            "station = pd.DataFrame(summary['station_summary']).set_index('scenario')\n"
            "pooled = pd.DataFrame(summary['pooled_summary']).set_index('scenario')\n"
            "legacy = pd.DataFrame(summary['legacy_summary']).set_index('scenario')\n"
            "print(\n"
            "    'State continuity is an engineering correction, but not a performance breakthrough.\\n'\n"
            "    f\"S01 changes states on {summary['state_change_summary']['rows_with_any_state_change_s01_vs_s00']:,} OOF rows; \"\n"
            "    f\"median station log-NSE changes from {station.loc['S00_NATIVE','median_log_nse']:.4f} to \"\n"
            "    f\"{station.loc['S01','median_log_nse']:.4f}; \"\n"
            "    f\"{int(legacy.loc['S01','lowflow_log_rmse_improved'])}/{int(legacy.loc['S01','evaluable_targets'])} legacy targets improve in low-flow log-RMSE.\\n\"\n"
            "    'The discharge overlay remains provisional: 27/198 conflict keys are resolved and 171 remain unresolved.'\n"
            ")\n"
        ),
        markdown(
            "## Context & Methods\n\n"
            "The experiment holds Q72 equations, parameters, folds, station/Reach policy, exclusions, topology, Qma and forcing fixed. "
            "It contrasts original versus provisional discharge labels and observed-only versus full-forcing state advancement.\n\n"
            "### Key Assumptions\n\n"
            "- `S00_NATIVE` is the exact reproduction control.\n"
            "- Log metrics use `log1p(Q_m3s)`, matching the frozen Q72 audit.\n"
            "- Low/high flow months are the exact lowest/highest 25% within each Reach-fold.\n"
            "- Residual ACF uses exact calendar lags within the same Reach-fold.\n"
            "- Performance does not adjudicate discharge truth.\n"
        ),
        markdown("## Data\n"),
        code(
            "integrity = json.loads((REPORT / 'scenario_integrity.json').read_text(encoding='utf-8'))\n"
            "integrity_table = pd.DataFrame([{'scenario': key, **value['checks']} for key, value in integrity['scenarios'].items()])\n"
            "columns = ['scenario','rows_8738','reaches_110','folds_3','keys_unique','keys_equal_s00','actual_equal_s00','state_calendar_mode_expected','input_hash_expected']\n"
            "print(integrity_table[columns].to_string(index=False))\n"
            "assert integrity['passed']\n"
            "assert integrity_table.drop(columns=['scenario','maximum_actual_difference_s00']).to_numpy().all()\n"
        ),
        markdown("## Results\n"),
        code(
            "station_metrics = pd.read_csv(REPORT / 'station_metrics.csv', encoding='utf-8-sig')\n"
            "legacy_metrics = pd.read_csv(REPORT / 'legacy_lowflow_metrics.csv', encoding='utf-8-sig')\n"
            "acf = pd.read_csv(REPORT / 'residual_acf_metrics.csv', encoding='utf-8-sig')\n"
            "station_effects = pd.read_csv(REPORT / 'station_paired_effects.csv', encoding='utf-8-sig')\n"
            "summary_table = station_metrics.groupby('scenario').agg(\n"
            "    stations=('comid','size'), median_log_nse=('log_nse','median'), median_kge=('kge','median'),\n"
            "    median_abs_pbias=('pbias_pct', lambda x: np.median(np.abs(x))),\n"
            "    median_lowflow_log_rmse=('lowflow_log_rmse','median'),\n"
            "    median_highflow_log_rmse=('highflow_log_rmse','median'),\n"
            ").loc[['S00_NATIVE','S10','S01','S11']]\n"
            "print(summary_table.round(6).to_string())\n"
        ),
        markdown(
            "### Four-scenario station medians\n\n"
            "![Four-scenario station medians](../reports/scenario_diagnostics/charts/scenario_station_medians.png)\n"
        ),
        code(
            "legacy_comparison = legacy_metrics[legacy_metrics['scenario'] != 'S00_NATIVE'].groupby('scenario').agg(\n"
            "    targets=('comid','size'), lowflow_log_rmse_improved=('lowflow_log_rmse_improved_vs_s00','sum'),\n"
            "    absolute_median_log_bias_improved=('absolute_lowflow_log_bias_improved_vs_s00','sum'),\n"
            "    absolute_error_improved=('lowflow_absolute_error_improved_vs_s00','sum'),\n"
            ").loc[['S10','S01','S11']]\n"
            "print(legacy_comparison.to_string())\n"
        ),
        markdown(
            "### Legacy low-flow targets\n\n"
            "![Legacy low-flow improvements](../reports/scenario_diagnostics/charts/legacy_lowflow_improvement_counts.png)\n"
        ),
        code(
            "acf_pivot = acf.pivot(index='scenario', columns='lag_months', values='weighted_absolute_acf').loc[['S00_NATIVE','S10','S01','S11']]\n"
            "print('Weighted absolute residual ACF by exact within-fold calendar lag')\n"
            "print(acf_pivot.round(6).to_string())\n"
        ),
        code(
            "state_delta = station_effects[station_effects['metric'] == 'log_nse']['state_continuity_effect']\n"
            "print(pd.Series({\n"
            "    'stations_improved': int((state_delta > 0).sum()),\n"
            "    'stations_worsened': int((state_delta < 0).sum()),\n"
            "    'median_delta': float(state_delta.median()),\n"
            "    'mean_delta': float(state_delta.mean()),\n"
            "}).to_string())\n"
        ),
        markdown(
            "### Station-level state-continuity effects\n\n"
            "![Station-level state-continuity log-NSE changes](../reports/scenario_diagnostics/charts/state_continuity_station_log_nse_delta.png)\n"
        ),
        markdown(
            "## Takeaways\n\n"
            "- Full-forcing advancement changes the intended states, especially on gap-exposed OOF rows, without adding no-Q rows to fitting.\n"
            "- S01 has a very small median station log-NSE increase and mixed station effects; pooled log-NSE is slightly lower than S00.\n"
            "- Residual-memory changes are negligible, so state continuity does not explain away the historical low-flow discrepancy.\n"
            "- S10/S11 performance is descriptive only because 171 conflict keys remain unresolved.\n"
            "- The defensible terminal state is completed state repair with unresolved source-data conflicts, not a fully corrected new baseline.\n"
        ),
    ]
    passed, errors = execute_cells(cells)
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python (conda sparrow)", "language": "python", "name": "sparrow"},
            "language_info": {"name": "python", "version": "3.11"},
            "execution": {
                "method": "deterministic in-process cell execution under conda sparrow",
                "reason": "sparrow intentionally lacks Jupyter runtime packages; no package was installed or borrowed from another environment",
                "runtime": RUNTIME,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    validation = {
        "notebook": str(OUT),
        "execution_method": notebook["metadata"]["execution"]["method"],
        "runtime": RUNTIME,
        "code_cells": sum(cell["cell_type"] == "code" for cell in cells),
        "executed_code_cells": sum(cell["cell_type"] == "code" and cell["execution_count"] is not None for cell in cells),
        "errors": errors,
        "passed": passed and all(cell["execution_count"] is not None for cell in cells if cell["cell_type"] == "code"),
        "schema_checks": {
            "nbformat_4": notebook["nbformat"] == 4,
            "all_cells_have_type": all("cell_type" in cell for cell in cells),
            "all_code_cells_have_outputs": all("outputs" in cell for cell in cells if cell["cell_type"] == "code"),
            "required_sections_present": all(
                any(title in "".join(cell["source"]) for cell in cells if cell["cell_type"] == "markdown")
                for title in ["## tl;dr", "## Context & Methods", "## Data", "## Results", "## Takeaways"]
            ),
        },
    }
    validation["passed"] = validation["passed"] and all(validation["schema_checks"].values())
    VALIDATION.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    if not validation["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
