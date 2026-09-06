"""Build auditable PRB municipal-WWTP nitrogen model inputs.

Spatial allocation is supplied by the Chen 2012 PRB TDN anchors.  Temporal
change is the province-level terminal-aquatic TN load ratio independently
derived from Wang 2006-2019.  MEE inventories support plant identity/activity
QA and a conservative 2012 name crosswalk; anonymous Wang rows are never
matched to named plants.
"""
from __future__ import annotations

import calendar
from difflib import SequenceMatcher
import json
import math
from pathlib import Path
import re

import geopandas as gpd
import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260817_9"
STAGED = RUN / "inputs" / "staged"
READY = RUN / "inputs" / "model_ready" / "point_sources"
QA = RUN / "inputs" / "qa"
YEARS = range(2006, 2020)
MEE_YEARS = range(2007, 2015)
CHEN_PROVINCE = {"Guangxi Zhuang Autonomous Region": "Guangxi"}
TN_SCENARIOS = {
    "tdn_equals_tn_lower_bound": 1.0,
    "tdn_fraction_of_tn_0.9": 0.9,
    "tdn_fraction_of_tn_0.8": 0.8,
}


def normalize_name(value: object) -> str:
    text = str(value) if pd.notna(value) else ""
    text = text.replace("有限责任公司", "有限公司").replace("（", "(").replace("）", ")")
    text = re.sub(r"[\s\-—_·,，。.;；:：'\"“”‘’()（）\[\]【】]", "", text)
    return text.lower()


def normalize_city(value: object) -> str:
    text = normalize_name(value)
    return re.sub(r"(市|地区|自治州|区|县)$", "", text)


def deduplicate_chen(anchor: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = [
        "organization_code", "project_name", "longitude", "latitude",
        "capacity_source_value", "tdn_2012_t_n_yr",
    ]
    source = anchor.sort_values("chen_fid").copy()
    groups = source.groupby(keys, dropna=False, sort=False)
    audit = groups.agg(
        retained_chen_fid=("chen_fid", "min"),
        source_rows=("chen_fid", "size"),
        source_chen_fids=("chen_fid", lambda x: ",".join(map(str, sorted(x)))),
    ).reset_index()
    keep_ids = set(audit.retained_chen_fid)
    out = source[source.chen_fid.isin(keep_ids)].copy()
    counts = audit.set_index("retained_chen_fid").source_rows
    fids = audit.set_index("retained_chen_fid").source_chen_fids
    out["source_exact_duplicate_count"] = out.chen_fid.map(counts).astype(int)
    out["source_group_chen_fids"] = out.chen_fid.map(fids)
    return out, audit[audit.source_rows.gt(1)].copy()


def reach_quality(anchor: pd.DataFrame) -> pd.DataFrame:
    out = anchor.copy()
    same = out.catchment_reach_id.eq(out.nearest_reach_id)
    out["reach_assignment_quality"] = np.select(
        [same & out.distance_to_nearest_reach_m.le(2000), same & out.distance_to_nearest_reach_m.le(5000), same],
        ["A", "B", "C"],
        default="D",
    )
    out["model_reach_id"] = out.catchment_reach_id.where(same).astype("Int64")
    out["reach_model_inclusion"] = same
    out["reach_assignment_rule"] = "catchment reach retained only when independently nearest reach agrees"
    return out


def wang_province_year() -> pd.DataFrame:
    wang = pd.read_parquet(READY / "wang_terminal_aquatic_tn_records_2006_2019.parquet")
    out = wang.groupby(["province", "year"], as_index=False).agg(
        wang_records=("wang_row_id", "size"),
        wang_terminal_aquatic_tn_kg_n_yr=("tn_out_from_effluent_n2o_kg_n_yr", "sum"),
        wang_terminal_aquatic_volume_m3_yr=("wastewater_volume_m3_yr", "sum"),
    )
    out["wang_tn_concentration_mg_n_l"] = (
        out.wang_terminal_aquatic_tn_kg_n_yr * 1000 / out.wang_terminal_aquatic_volume_m3_yr
    )
    baseline = out[out.year.eq(2012)].set_index("province")
    out["wang_2012_tn_kg_n_yr"] = out.province.map(baseline.wang_terminal_aquatic_tn_kg_n_yr)
    out["wang_2012_volume_m3_yr"] = out.province.map(baseline.wang_terminal_aquatic_volume_m3_yr)
    out["wang_2012_concentration_mg_n_l"] = out.province.map(baseline.wang_tn_concentration_mg_n_l)
    out["province_tn_load_factor_relative_2012"] = out.wang_terminal_aquatic_tn_kg_n_yr / out.wang_2012_tn_kg_n_yr
    out["province_volume_factor_relative_2012"] = out.wang_terminal_aquatic_volume_m3_yr / out.wang_2012_volume_m3_yr
    out["province_concentration_factor_relative_2012"] = (
        out.wang_tn_concentration_mg_n_l / out.wang_2012_concentration_mg_n_l
    )
    return out


def read_mee() -> pd.DataFrame:
    frames = []
    for year in MEE_YEARS:
        path = STAGED / "mee_grid_candidates" / f"mee_{year}_grid_repaired.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def mee_activity_validation(mee: pd.DataFrame, wang: pd.DataFrame) -> pd.DataFrame:
    mee = mee.copy()
    mee["days_in_inventory_year"] = mee.source_year.map(lambda y: 366 if calendar.isleap(int(y)) else 365)
    mee["annualized_mean_flow_m3_yr"] = mee.mean_daily_flow_m3_d_repaired * mee.days_in_inventory_year
    activity = mee.groupby(["province_english", "source_year"], dropna=False, as_index=False).agg(
        mee_inventory_rows=("facility_id_grid_sequence", "size"),
        mee_rows_with_model_activity_fields=("model_activity_fields_ok", "sum"),
        mee_annualized_mean_flow_m3_yr=("annualized_mean_flow_m3_yr", "sum"),
        mee_design_capacity_m3_d=("design_capacity_m3_d_repaired", "sum"),
    ).rename(columns={"province_english": "province", "source_year": "year"})
    activity["mee_activity_field_coverage_fraction"] = (
        activity.mee_rows_with_model_activity_fields / activity.mee_inventory_rows
    )
    compare = activity.merge(
        wang[["province", "year", "wang_terminal_aquatic_volume_m3_yr"]],
        on=["province", "year"], how="left", validate="one_to_one",
    )
    compare["mee_to_wang_terminal_aquatic_volume_ratio"] = (
        compare.mee_annualized_mean_flow_m3_yr / compare.wang_terminal_aquatic_volume_m3_yr
    )
    return compare


def chen_mee_crosswalk(anchor: pd.DataFrame, mee_2012: pd.DataFrame) -> pd.DataFrame:
    mee = mee_2012.copy()
    mee["name_normalized"] = mee.facility_name_repaired.map(normalize_name)
    mee["city_normalized"] = mee.get("city_repaired", pd.Series(index=mee.index, dtype=object)).map(normalize_city)
    records = []
    for _, chen in anchor.iterrows():
        province = CHEN_PROVINCE.get(str(chen.province), str(chen.province))
        candidates = mee[mee.province_english.eq(province)]
        chen_name = normalize_name(chen.project_name)
        chen_city = normalize_city(chen.city)
        scored = []
        for index, candidate in candidates.iterrows():
            name_ratio = SequenceMatcher(None, chen_name, candidate.name_normalized).ratio()
            city_match = bool(chen_city and candidate.city_normalized and (
                chen_city in candidate.city_normalized or candidate.city_normalized in chen_city
            ))
            chen_capacity = float(chen.capacity_source_value) * 10000.0
            mee_capacity = candidate.design_capacity_m3_d_repaired
            if pd.notna(mee_capacity) and chen_capacity > 0 and mee_capacity > 0:
                capacity_ratio = float(mee_capacity / chen_capacity)
                capacity_score = math.exp(-abs(math.log(capacity_ratio)))
            else:
                capacity_ratio, capacity_score = np.nan, 0.0
            score = 0.80 * name_ratio + 0.10 * float(city_match) + 0.10 * capacity_score
            scored.append((score, name_ratio, city_match, capacity_ratio, index))
        scored.sort(reverse=True, key=lambda x: x[0])
        if not scored:
            records.append({"chen_fid": chen.chen_fid, "match_quality": "U", "match_score": np.nan})
            continue
        best = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        candidate = mee.loc[best[4]]
        margin = best[0] - second_score
        capacity_ok_a = pd.notna(best[3]) and 0.5 <= best[3] <= 2.0
        capacity_ok_b = pd.notna(best[3]) and (1 / 3) <= best[3] <= 3.0
        if best[1] >= 0.98 and (best[2] or capacity_ok_a):
            quality = "A"
        elif best[1] >= 0.86 and margin >= 0.05 and (best[2] or capacity_ok_b):
            quality = "B"
        else:
            quality = "C"
        records.append({
            "chen_fid": int(chen.chen_fid),
            "chen_project_name": chen.project_name,
            "chen_province": province,
            "chen_city": chen.city,
            "chen_capacity_m3_d_assuming_source_10k_m3_d": float(chen.capacity_source_value) * 10000.0,
            "mee_2012_facility_id_grid_sequence": int(candidate.facility_id_grid_sequence),
            "mee_2012_source_page": int(candidate.source_page),
            "mee_2012_facility_name": candidate.facility_name_repaired,
            "mee_2012_city": candidate.get("city_repaired", ""),
            "mee_2012_design_capacity_m3_d": candidate.design_capacity_m3_d_repaired,
            "mee_2012_mean_daily_flow_m3_d": candidate.mean_daily_flow_m3_d_repaired,
            "name_similarity": best[1],
            "city_match": best[2],
            "capacity_ratio_mee_to_chen": best[3],
            "match_score": best[0],
            "score_margin_to_second": margin,
            "match_quality": quality,
            "model_use_allowed": quality in {"A", "B"},
        })
    return pd.DataFrame(records)


def build_annual(anchor: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    base = anchor.copy()
    base["wang_province"] = base.province.map(lambda x: CHEN_PROVINCE.get(str(x), str(x)))
    year_rows = []
    factor_index = factors.set_index(["province", "year"])
    for year in YEARS:
        current = base.copy()
        keys = list(zip(current.wang_province, [year] * len(current)))
        current["year"] = year
        current["province_tn_load_factor_relative_2012"] = [
            factor_index.at[key, "province_tn_load_factor_relative_2012"] if key in factor_index.index else np.nan
            for key in keys
        ]
        current["province_volume_factor_relative_2012"] = [
            factor_index.at[key, "province_volume_factor_relative_2012"] if key in factor_index.index else np.nan
            for key in keys
        ]
        current["province_concentration_factor_relative_2012"] = [
            factor_index.at[key, "province_concentration_factor_relative_2012"] if key in factor_index.index else np.nan
            for key in keys
        ]
        year_rows.append(current)
    annual_base = pd.concat(year_rows, ignore_index=True)
    scenarios = []
    for label, dissolved_fraction in TN_SCENARIOS.items():
        current = annual_base.copy()
        current["tn_scenario"] = label
        current["assumed_effluent_tdn_fraction_of_tn"] = dissolved_fraction
        current["tn_load_t_n_yr"] = (
            current.tdn_2012_t_n_yr
            * current.province_tn_load_factor_relative_2012
            / dissolved_fraction
        )
        current["tn_load_kg_n_yr"] = current.tn_load_t_n_yr * 1000.0
        scenarios.append(current)
    out = pd.concat(scenarios, ignore_index=True)
    out["temporal_method"] = "Chen2012_anchor_times_Wang_province_terminal_aquatic_TN_load_ratio"
    return out


def monthly_outputs(annual: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    included = annual[annual.reach_model_inclusion].copy()
    records = []
    for month in range(1, 13):
        current = included.copy()
        current["month"] = month
        current["days_in_year"] = current.year.map(lambda y: 366 if calendar.isleap(int(y)) else 365)
        current["days_in_month"] = [calendar.monthrange(int(y), month)[1] for y in current.year]
        current["tn_load_kg_n_month"] = (
            current.tn_load_kg_n_yr * current.days_in_month / current.days_in_year
        )
        records.append(current)
    facility = pd.concat(records, ignore_index=True).sort_values(["chen_fid", "tn_scenario", "year", "month"])
    reach = facility.groupby(
        ["model_reach_id", "year", "month", "tn_scenario", "assumed_effluent_tdn_fraction_of_tn"],
        as_index=False,
    ).agg(
        facilities=("chen_fid", "nunique"),
        tn_load_kg_n_month=("tn_load_kg_n_month", "sum"),
    )
    reach["tn_load_t_n_month"] = reach.tn_load_kg_n_month / 1000.0
    reach["monthly_allocation"] = "annual load multiplied by calendar days in month / days in year"
    return facility, reach


def main() -> None:
    READY.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    original = pd.read_parquet(READY / "chen_wwtp_2012_prb_anchor.parquet")
    anchor, duplicate_audit = deduplicate_chen(original)
    anchor = reach_quality(anchor)
    wang = wang_province_year()
    mee = read_mee()
    mee.to_parquet(STAGED / "mee_wwtp_inventory_2007_2014_repaired.parquet", index=False)
    activity = mee_activity_validation(mee, wang)
    crosswalk = chen_mee_crosswalk(anchor, mee[mee.source_year.eq(2012)])
    annual = build_annual(anchor, wang)
    facility_month, reach_month = monthly_outputs(annual)

    anchor.to_parquet(READY / "prb_wwtp_2012_anchor_master.parquet", index=False)
    source_geo = gpd.read_file(READY / "chen_wwtp_2012_prb_anchor.gpkg")
    source_geo = source_geo[source_geo.chen_fid.isin(anchor.chen_fid)].drop_duplicates("chen_fid")
    added = [
        "chen_fid", "source_exact_duplicate_count", "source_group_chen_fids",
        "reach_assignment_quality", "model_reach_id", "reach_model_inclusion", "reach_assignment_rule",
    ]
    source_geo = source_geo.drop(columns=[column for column in added[1:] if column in source_geo], errors="ignore")
    source_geo = source_geo.merge(anchor[added], on="chen_fid", how="left", validate="one_to_one")
    source_geo.to_file(READY / "prb_wwtp_2012_anchor_master.gpkg", driver="GPKG")
    annual.to_parquet(READY / "prb_wwtp_tn_annual_2006_2019.parquet", index=False)
    facility_month.to_parquet(READY / "prb_wwtp_tn_monthly_facility_2006_2019.parquet", index=False)
    reach_month.to_parquet(READY / "prb_wwtp_tn_monthly_reach_2006_2019.parquet", index=False)
    activity.to_parquet(READY / "mee_province_year_activity_validation_2007_2014.parquet", index=False)
    crosswalk.to_parquet(READY / "chen_mee_2012_crosswalk.parquet", index=False)
    duplicate_audit.to_csv(QA / "chen_exact_duplicate_groups.csv", index=False, encoding="utf-8-sig")
    crosswalk.to_csv(QA / "chen_mee_2012_crosswalk.csv", index=False, encoding="utf-8-sig")
    activity.to_csv(QA / "mee_wang_province_year_activity_comparison.csv", index=False, encoding="utf-8-sig")
    wang.to_parquet(READY / "wang_province_year_temporal_factors_2006_2019.parquet", index=False)

    central = annual[annual.tn_scenario.eq("tdn_equals_tn_lower_bound")]
    central_2012 = central[central.year.eq(2012)]
    base_difference = float((central_2012.tn_load_t_n_yr - central_2012.tdn_2012_t_n_yr).abs().max())
    monthly_check = facility_month.groupby(["chen_fid", "year", "tn_scenario"], as_index=False).tn_load_kg_n_month.sum()
    annual_check = annual[annual.reach_model_inclusion][["chen_fid", "year", "tn_scenario", "tn_load_kg_n_yr"]]
    closure = annual_check.merge(monthly_check, on=["chen_fid", "year", "tn_scenario"], validate="one_to_one")
    monthly_difference = float((closure.tn_load_kg_n_yr - closure.tn_load_kg_n_month).abs().max())
    qa = {
        "chen_source_rows_in_prb": int(len(original)),
        "chen_exact_duplicate_rows_removed": int(len(original) - len(anchor)),
        "chen_unique_anchor_rows": int(len(anchor)),
        "reach_quality_counts": {str(k): int(v) for k, v in anchor.reach_assignment_quality.value_counts().items()},
        "reach_approved_anchor_rows": int(anchor.reach_model_inclusion.sum()),
        "reach_excluded_D_rows": int((~anchor.reach_model_inclusion).sum()),
        "annual_rows_all_scenarios": int(len(annual)),
        "facility_month_rows_approved_reaches": int(len(facility_month)),
        "reach_month_rows": int(len(reach_month)),
        "central_2012_max_abs_difference_from_deduplicated_chen_t_n_yr": base_difference,
        "monthly_to_annual_max_abs_closure_difference_kg_n": monthly_difference,
        "crosswalk_quality_counts": {str(k): int(v) for k, v in crosswalk.match_quality.value_counts().items()},
        "crosswalk_model_allowed_rows": int(crosswalk.model_use_allowed.fillna(False).sum()),
        "mee_activity_rows": int(len(activity)),
        "method_release": "PASS" if base_difference < 1e-9 and monthly_difference < 1e-6 else "FAIL",
        "interpretation": "Central scenario treats Chen TDN as a lower-bound TN-equivalent. The 0.9 and 0.8 dissolved-fraction cases are sensitivity scenarios, not measured fractions.",
    }
    if qa["method_release"] != "PASS":
        raise RuntimeError(json.dumps(qa, ensure_ascii=False, indent=2))
    (QA / "prb_wwtp_tn_model_input_qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
