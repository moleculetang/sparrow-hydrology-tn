from __future__ import annotations

import hashlib
import json
import os
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260813_36"
PARENT = ROOT / "5_Test" / "20260813_31" / "inputs" / "parent_indata.parquet"
MONTHLY = ROOT / "5_Test" / "20260813_31" / "inputs" / "merged_discharge" / "merged_2006_2022_monthly.parquet"
OUTPUT = RUN / "inputs" / "A1_indata.parquet"
EXPECTED_PARENT_SHA = "3678cd8a950290a1962f27c9e702ae84087f594e5e3bdc05332644dc200ec26f"
M3S_TO_CFS = 35.3146667


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def norm(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().replace(" ", "").replace("　", "")
    text = text.replace("（", "(").replace("）", ")")
    return text[:-1] if text.endswith("站") else text


def main() -> None:
    if Path(sys.prefix).resolve() != Path(r"D:\ProgramData\anaconda3\envs\sparrow").resolve():
        raise RuntimeError(f"wrong runtime: {sys.prefix}")
    if sha256(PARENT) != EXPECTED_PARENT_SHA:
        raise RuntimeError("parent SHA mismatch")
    parent = pd.read_parquet(PARENT)
    monthly = pd.read_parquet(MONTHLY)
    monthly["station_norm"] = monthly["station_key"].map(norm)
    source = monthly.loc[
        monthly["station_norm"].eq("江边街(二)")
        & monthly["year"].between(2006, 2022),
        ["year", "month", "q_m3s", "n_days", "source_group", "station_entity"],
    ].copy()
    if len(source) != 204 or source[["year", "month"]].duplicated().any():
        raise RuntimeError(f"江边街 monthly completeness failed: {len(source)}")
    source["Q_new_cfs"] = source["q_m3s"].astype(float) * M3S_TO_CFS
    out = parent.copy()
    mask = out["comid"].eq(193)
    if int(mask.sum()) != 204:
        raise RuntimeError(f"Reach 193 calendar rows: {int(mask.sum())}")
    old = out.loc[mask, ["comid", "year", "month", "q_site", "Q_obsv_cfs"]].copy()
    replacement = out.loc[mask, ["year", "month"]].merge(
        source, on=["year", "month"], how="left", validate="one_to_one"
    )
    if replacement["Q_new_cfs"].isna().any():
        raise RuntimeError("replacement has missing Q")
    out.loc[mask, "q_site"] = "江边街（二）站"
    out.loc[mask, "station_id"] = "江边街（二）站"
    out.loc[mask, "Q_obsv_cfs"] = replacement["Q_new_cfs"].to_numpy(float)

    non_target = ~mask
    compare_cols = [c for c in parent.columns if c not in []]
    if not parent.loc[non_target, compare_cols].reset_index(drop=True).equals(
        out.loc[non_target, compare_cols].reset_index(drop=True)
    ):
        raise RuntimeError("non-target rows changed")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT, index=False)
    new = out.loc[mask, ["comid", "year", "month", "q_site", "Q_obsv_cfs"]].copy()
    audit = old.merge(new, on=["comid", "year", "month"], suffixes=("_old", "_new"))
    audit = audit.merge(source, on=["year", "month"], how="left")
    audit.to_csv(RUN / "reach193_observation_replacement.csv", index=False, encoding="utf-8-sig")
    payload = {
        "parent_sha256": sha256(PARENT),
        "output_sha256": sha256(OUTPUT),
        "rows": int(len(out)),
        "reach193_rows": int(mask.sum()),
        "old_observed_months": int(old["Q_obsv_cfs"].notna().sum()),
        "new_observed_months": int(new["Q_obsv_cfs"].notna().sum()),
        "old_station_labels": sorted(
            old["q_site"].dropna().astype(str).unique().tolist()
        ),
        "new_station_labels": sorted(
            new["q_site"].dropna().astype(str).unique().tolist()
        ),
        "non_target_rows_identical": True,
        "expected_outer_oof_rows": 8822,
        "source_months": int(len(source)),
        "source_groups": sorted(source["source_group"].astype(str).unique().tolist()),
    }
    (RUN / "input_rebuild_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
