from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


TEST_ROOT = Path(r"E:\SPARROW\5_Test")
CONTROL = TEST_ROOT / "20260721_1"
CONTROL_DIR = CONTROL / "reports" / "dynamic_station_screening"
STATE_PATH = CONTROL_DIR / "chain_state.json"
DECISION_LEDGER = CONTROL_DIR / "decision_ledger.csv"
SET_LEDGER = CONTROL_DIR / "station_set_ledger.csv"
OUTPUT = CONTROL_DIR / "interaction_candidate_queue.csv"
REPORT = CONTROL_DIR / "interaction_candidate_queue.md"
FOLDS = (
    "fit_2006_2011_eval_2012_2013",
    "fit_2006_2013_eval_2014_2015",
    "fit_2006_2015_eval_2016_2018",
)


def group_id(stations: tuple[str, ...]) -> str:
    return hashlib.sha256("\0".join(stations).encode("utf-8")).hexdigest()[:12]


def main() -> None:
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    parent = TEST_ROOT / str(state["accepted_parent_run"])
    audit_run = TEST_ROOT / str(state["last_influence_audit_run"])
    audit_path = audit_run / "reports" / "station_influence_audit" / "full_ablation_candidates.csv"
    audit = pd.read_csv(audit_path, encoding="utf-8-sig")
    sites = sorted(audit["q_site"].astype(str).unique())
    decision = pd.read_csv(DECISION_LEDGER, encoding="utf-8-sig")
    exact = decision[decision["policy_sha256"].astype(str).eq(str(state["policy_sha256"]))].copy()
    exact = exact.drop_duplicates("candidate_station", keep="last").set_index("candidate_station")

    parts = []
    for fold in FOLDS:
        path = parent / "reports" / "station_screening" / "blocked_folds" / fold / "evaluation_predictions.csv"
        frame = pd.read_csv(path, encoding="utf-8-sig", usecols=["q_site", "year", "month", "actual", "predict"])
        frame = frame[frame["q_site"].astype(str).isin(sites)].copy()
        frame["fold"] = fold
        frame["log_residual"] = np.log(pd.to_numeric(frame["actual"], errors="coerce") + 1.0e-6) - np.log(
            pd.to_numeric(frame["predict"], errors="coerce") + 1.0e-6
        )
        parts.append(frame)
    residual = pd.concat(parts, ignore_index=True)
    wide = residual.pivot_table(index=["fold", "year", "month"], columns="q_site", values="log_residual")
    corr = wide.corr(min_periods=24)

    groups: dict[tuple[str, ...], dict[str, object]] = {}

    def add(stations: tuple[str, ...], reason: str, score: float, correlation: float | None = None) -> None:
        stations = tuple(sorted(stations))
        item = groups.setdefault(stations, {"reasons": set(), "score": -np.inf, "max_abs_residual_correlation": np.nan})
        item["reasons"].add(reason)
        item["score"] = max(float(item["score"]), float(score))
        if correlation is not None:
            previous = float(item["max_abs_residual_correlation"])
            item["max_abs_residual_correlation"] = max(abs(float(correlation)), previous if np.isfinite(previous) else 0.0)

    strong_edges: list[tuple[str, str]] = []
    for a, b in itertools.combinations(sites, 2):
        value = float(corr.loc[a, b])
        if np.isfinite(value) and abs(value) >= 0.25:
            threshold_reason = "strong_residual_correlation" if abs(value) >= 0.35 else "compensating_residual_correlation"
            add((a, b), threshold_reason, 100.0 * abs(value), value)
        if np.isfinite(value) and value >= 0.35:
            strong_edges.append((a, b))

    # Use maximal cliques, not connected components: every pair inside a group must be strongly related.
    strong_edge_set = {tuple(sorted(edge)) for edge in strong_edges}
    cliques = []
    for size in range(3, min(5, len(sites) + 1)):
        for subset in itertools.combinations(sites, size):
            if all(tuple(sorted(pair)) in strong_edge_set for pair in itertools.combinations(subset, 2)):
                cliques.append(subset)
    maximal_cliques = [clique for clique in cliques if not any(set(clique) < set(other) for other in cliques)]
    for clique in maximal_cliques:
        values = [abs(float(corr.loc[a, b])) for a, b in itertools.combinations(clique, 2)]
        add(tuple(clique), "strong_residual_maximal_clique", 110.0 + len(clique), max(values))

    # Exact single-ablation good gains can combine even when residual correlation is moderate.
    gain_sites = []
    for site in sites:
        if site in exact.index and pd.to_numeric(exact.loc[site, "cumulative_good_gain"], errors="coerce") > 0:
            gain_sites.append(site)
    for a, b in itertools.combinations(gain_sites, 2):
        gain = float(exact.loc[a, "cumulative_good_gain"]) + float(exact.loc[b, "cumulative_good_gain"])
        add((a, b), "positive_good_gain_complement", 200.0 + gain, float(corr.loc[a, b]))

    tested: set[tuple[str, ...]] = set()
    if SET_LEDGER.exists():
        old = pd.read_csv(SET_LEDGER, encoding="utf-8-sig")
        old = old[
            old["policy_sha256_before"].astype(str).eq(str(state["policy_sha256"]))
            & old["trial_type"].astype(str).eq("joint_ablation")
        ]
        tested = {tuple(sorted(str(value).split(";"))) for value in old["stations"]}

    rows = []
    for stations, item in groups.items():
        gains = [float(exact.loc[s, "cumulative_good_gain"]) if s in exact.index else np.nan for s in stations]
        rows.append(
            {
                "group_id": group_id(stations),
                "stations": ";".join(stations),
                "station_count": len(stations),
                "reasons": ";".join(sorted(item["reasons"])),
                "priority_score": float(item["score"]),
                "max_abs_residual_correlation": item["max_abs_residual_correlation"],
                "sum_exact_cumulative_good_gain": float(np.nansum(gains)),
                "policy_sha256": state["policy_sha256"],
                "accepted_parent_run": parent.name,
                "audit_run": audit_run.name,
                "already_tested": stations in tested,
            }
        )
    queue = pd.DataFrame(rows).sort_values(
        ["already_tested", "priority_score", "station_count", "stations"], ascending=[True, False, True, True]
    )
    queue.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    pending = queue[~queue["already_tested"]]
    lines = [
        "# Interaction candidate queue",
        "",
        f"- accepted parent: `{parent.name}`",
        f"- policy SHA256: `{state['policy_sha256']}`",
        f"- source audit: `{audit_run.name}`",
        f"- candidates: {len(queue)}; pending: {len(pending)}",
        "",
        "| group | stations | reasons | score | |residual corr| | exact good gain sum | tested |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in queue.itertuples(index=False):
        lines.append(
            f"| {row.group_id} | {row.stations} | {row.reasons} | {row.priority_score:.3f} | "
            f"{row.max_abs_residual_correlation:.3f} | {row.sum_exact_cumulative_good_gain:+.0f} | {row.already_tested} |"
        )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"queue": str(OUTPUT), "groups": len(queue), "pending": len(pending)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
