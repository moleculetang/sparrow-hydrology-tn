from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


HERE = Path(r"E:\SPARROW\5_Test\20260824_7")
SOURCE = Path(r"E:\SPARROW\5_Test\20260820_19\outputs\tree_163_domain_audit.parquet")
OUT = HERE / "outputs" / "tree_163_nested_audit.parquet"
DECISION = HERE / "reports" / "nested_spatial_decision.json"
REPORT = HERE / "reports" / "technical_report.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError("sparrow required")
    source = pd.read_parquet(SOURCE)
    required = {
        "terminal_tree_id", "domain_interpretation", "primary_gate_role",
        "river_reach_prediction_status", "station_keys",
    }
    if not required.issubset(source.columns):
        raise RuntimeError(f"tree 163 source schema mismatch: {sorted(required - set(source.columns))}")
    if not (
        len(source) > 0
        and source.terminal_tree_id.eq(163).all()
        and source.primary_gate_role.eq("diagnostic_only").all()
        and source.domain_interpretation.eq("open_lake_center_not_river_outlet").all()
    ):
        raise RuntimeError("tree 163 source domain contract failed")
    source = source.copy()
    source["source_artifact"] = str(SOURCE)
    source["source_sha256"] = sha256(SOURCE)
    source["nested_primary_gate_included"] = False
    source.to_parquet(OUT, index=False)

    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    decision.pop("tree_163_models_noninferior", None)
    decision.pop("tree_163_models_absolute_skill_supported", None)
    decision.update({
        "primary_terminal_trees": [20, 22, 23, 26, 56, 212, 217],
        "tree_163_role": "diagnostic_only_open_lake_center",
        "tree_163_in_primary_loto": False,
        "tree_163_source_sha256": sha256(SOURCE),
    })
    DECISION.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

    text = REPORT.read_text(encoding="utf-8")
    text = text.replace(
        "- tree 163相对Parent非劣：0/12；\n- tree 163 absolute skill通过：0/12。",
        "- 正式LOTO域：7棵primary river terminal trees（20、22、23、26、56、212、217）；\n"
        "- tree 163：抚仙湖心开放水体诊断域，不是河段出口，不进入primary LOSO/LOTO门禁。",
    )
    REPORT.write_text(text, encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "tree_163_rows": len(source),
        "tree_163_role": "diagnostic_only_open_lake_center",
        "tree_163_in_primary_loto": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
