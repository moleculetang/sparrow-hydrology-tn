"""Stage and audit municipal WWTP sources for the PRB nitrogen model.

This is deliberately a source-foundation workflow.  It creates the spatial
2012 Chen anchor and province/process/year Wang summaries, but does *not*
pretend that anonymous Wang records can be matched to individual WWTPs or
that reported TN removal equals effluent TN mass.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point


ROOT = Path(r"E:\SPARROW")
UNSORTED = ROOT / "0_reach_topology" / "data" / "未整理"
RAW = ROOT / "0_reach_topology" / "data" / "raw" / "point_sources" / "wastewater"
RUN = ROOT / "5_Test" / "20260817_9"
BASE = ROOT / "5_Test" / "20260810_1" / "inputs" / "spatial_corrected"
PDFINFO = Path(r"C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdfinfo.exe")

MEE_URLS = {
    2007: "https://www.mee.gov.cn/gkml/hbb/bgg/200910/W020080407541655278681.pdf",
    2008: "https://www.mee.gov.cn/gkml/hbb/bgg/200910/W020090622322728138651.pdf",
    2009: "https://www.mee.gov.cn/gkml/hbb/bgg/201003/W020100326526568082581.pdf",
    2010: "https://www.mee.gov.cn/gkml/hbb/bgg/201104/W020110420407642318640.pdf",
    2011: "https://www.mee.gov.cn/gkml/hbb/bgg/201204/W020120424390773833974.pdf",
    2012: "https://www.mee.gov.cn/gkml/hbb/bgg/201305/W020130508476747765965.pdf",
    2013: "https://www.mee.gov.cn/gkml/hbb/bgg/201404/W020140415399348916037.pdf",
}
MEE_EXPECTED_FACILITIES = {2007: 1178, 2008: 1521, 2009: 1916, 2010: 2739, 2011: 3184, 2012: 3836, 2013: 4136, 2014: 4436}


def move_once(source: Path, target: Path) -> Path:
    if target.exists():
        return target
    if not source.exists():
        raise FileNotFoundError(f"Missing source and target: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))
    return target


def pdf_metadata(path: Path) -> dict[str, object]:
    text = subprocess.check_output([str(PDFINFO), str(path)], text=True, errors="replace")
    fields: dict[str, object] = {"path": str(path), "bytes": path.stat().st_size}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key, value = key.strip().lower(), value.strip()
        if key == "pages":
            fields["pages"] = int(value)
        elif key == "encrypted":
            fields["encrypted"] = value
        elif key == "page size":
            fields["page_size"] = value
        elif key == "title":
            fields["title"] = value
    if fields.get("encrypted") != "no" or int(fields.get("pages", 0)) < 1:
        raise RuntimeError(f"Unreadable or encrypted PDF: {path}")
    return fields


def stage_sources() -> tuple[Path, Path, Path]:
    chen = move_once(
        UNSORTED / "es8b07352_si_002.xlsx",
        RAW / "chen_2012_wwtp" / "data" / "es8b07352_si_002.xlsx",
    )
    wang = move_once(
        UNSORTED / "20062019WWTPGHGemissionsinChina.xlsx",
        RAW / "wang_wwtp_2006_2019" / "data" / "20062019WWTPGHGemissionsinChina.xlsx",
    )
    mee2014 = move_once(
        UNSORTED / "W020150609575919731164.pdf",
        RAW / "mee_wwtp_inventory" / "2014" / "archives" / "W020150609575919731164.pdf",
    )
    for year in MEE_URLS:
        candidates = list((RAW / "mee_wwtp_inventory" / str(year) / "archives").glob("*.pdf"))
        if len(candidates) != 1 or candidates[0].stat().st_size < 10_000:
            raise RuntimeError(f"MEE {year} PDF download is absent or implausibly small")
    return chen, wang, mee2014


def clean_chen(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, dtype={"Organization code": "string"})
    out = pd.DataFrame({
        "chen_fid": pd.to_numeric(raw["FID"], errors="raise").astype("int64"),
        "chen_objectid": pd.to_numeric(raw["OBJECTID"], errors="coerce"),
        "county": raw["County"].astype("string").str.strip(),
        "city": raw["City"].astype("string").str.strip(),
        "province": raw["Province"].astype("string").str.strip(),
        "organization_code": raw["Organization code"].astype("string").str.strip(),
        "project_name": raw["ProjectName"].astype("string").str.strip(),
        "longitude": pd.to_numeric(raw["Longitude"], errors="coerce"),
        "latitude": pd.to_numeric(raw["Latitude"], errors="coerce"),
        "address": raw["Address"].astype("string").str.strip(),
        "capacity_source_value": pd.to_numeric(raw["Capacity"], errors="coerce"),
        "n_removal_fraction": pd.to_numeric(raw["Nrevoval"], errors="coerce"),
        "treatment_technology": raw["Treatment technology"].astype("string").str.strip(),
        "din_2012_kton_n_yr": pd.to_numeric(raw["DIN"], errors="coerce"),
        "don_2012_kton_n_yr": pd.to_numeric(raw["DON"], errors="coerce"),
    })
    out["tdn_2012_kton_n_yr"] = out.din_2012_kton_n_yr + out.don_2012_kton_n_yr
    out["tdn_2012_t_n_yr"] = out.tdn_2012_kton_n_yr * 1000.0
    valid = (
        out.longitude.between(73, 136) & out.latitude.between(15, 55)
        & out[["din_2012_kton_n_yr", "don_2012_kton_n_yr"]].ge(0).all(axis=1)
        & out.project_name.notna() & out.project_name.ne("")
    )
    if not valid.all():
        raise RuntimeError(f"Chen source has {int((~valid).sum())} invalid location/load/name rows")
    if out.chen_fid.duplicated().any():
        raise RuntimeError("Chen FID is not a unique source-row key")
    return out


def clean_wang(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="Firm level GHG emissions")
    raw.columns = [" ".join(str(c).split()) for c in raw.columns]
    required = {
        "NO.", "Year", "Province", "WWTP or other facilities", "Major category of treatment process",
        "Discharge pathway", "Wastewater Volume (10, 000 m3)", "TN removed (kg COD removed/year)",
        "Bio_N2O (t CO2eq/year)", "Effluent_N2O (t CO2eq/year)",
    }
    missing = required - set(raw.columns)
    if missing:
        raise RuntimeError(f"Wang sheet schema changed: missing {sorted(missing)}")
    out = pd.DataFrame({
        "wang_row_id": pd.to_numeric(raw["NO."], errors="raise").astype("int64"),
        "year": pd.to_numeric(raw["Year"], errors="raise").astype("int16"),
        "province": raw["Province"].astype("string").str.strip(),
        "facility_type": raw["WWTP or other facilities"].astype("string").str.strip(),
        "process_category": raw["Major category of treatment process"].astype("string").str.strip(),
        "process_subcategory_1": raw["1st subcategory of treatment process"].astype("string").str.strip(),
        "process_subcategory_2": raw["2nd subcategory of treatment process"].astype("string").str.strip(),
        "discharge_pathway": raw["Discharge pathway"].astype("string").str.strip(),
        "wastewater_volume_10k_m3_yr": pd.to_numeric(raw["Wastewater Volume (10, 000 m3)"], errors="coerce"),
        # Preserve the published header because its COD wording conflicts with
        # the field name. Unit interpretation is held behind a QA gate.
        "tn_removed_source_value": pd.to_numeric(raw["TN removed (kg COD removed/year)"], errors="coerce"),
        "tn_removed_source_header": "TN removed (kg COD removed/year)",
        "bio_n2o_t_co2eq_yr": pd.to_numeric(raw["Bio_N2O (t CO2eq/year)"], errors="coerce"),
        "effluent_n2o_t_co2eq_yr": pd.to_numeric(raw["Effluent_N2O (t CO2eq/year)"], errors="coerce"),
    })
    out["wastewater_volume_m3_yr"] = out.wastewater_volume_10k_m3_yr * 10_000.0
    valid = out.year.between(2006, 2019) & out.wastewater_volume_m3_yr.ge(0) & out.tn_removed_source_value.ge(0)
    if not valid.all() or out.wang_row_id.duplicated().any():
        raise RuntimeError("Wang source fails row-key, date, or non-negative quantity checks")
    return out


def chen_prb_anchor(chen: pd.DataFrame) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    catchments = gpd.read_file(BASE / "reach_catchments.shp").loc[:, ["reach_id", "geometry"]]
    reaches = gpd.read_file(BASE / "reaches_topology.shp").loc[:, ["reach_id", "geometry"]]
    points = gpd.GeoDataFrame(chen.copy(), geometry=[Point(xy) for xy in zip(chen.longitude, chen.latitude)], crs="EPSG:4326").to_crs(catchments.crs)
    joined = gpd.sjoin(points, catchments.rename(columns={"reach_id": "catchment_reach_id"}), how="inner", predicate="within").drop(columns="index_right")
    nearest_ids, distances = [], []
    for point in joined.geometry:
        nearest_index = reaches.geometry.distance(point).to_numpy().argmin()
        nearest_ids.append(int(reaches.iloc[nearest_index].reach_id))
        distances.append(float(reaches.iloc[nearest_index].geometry.distance(point)))
    joined["nearest_reach_id"] = nearest_ids
    joined["distance_to_nearest_reach_m"] = distances
    joined["spatial_role"] = "2012_TDN_spatial_anchor"
    audit = joined.drop(columns="geometry").loc[:, [
        "chen_fid", "project_name", "province", "city", "longitude", "latitude", "catchment_reach_id",
        "nearest_reach_id", "distance_to_nearest_reach_m", "tdn_2012_t_n_yr",
    ]].copy()
    return joined, audit


def source_audit(chen_path: Path, wang_path: Path, mee2014_path: Path, chen: pd.DataFrame, wang: pd.DataFrame, anchor: gpd.GeoDataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = [
        {
            "source": "Chen 2012 WWTP", "role": "2012 spatial TDN anchor", "coverage": "2012", "records": len(chen),
            "location": str(chen_path), "status": "ready; TDN is modelled source output, not measured TN", "url_or_identifier": "10.1021/acs.est.8b07352",
        },
        {
            "source": "Wang municipal WWTP GHG", "role": "province/process/year temporal intensity", "coverage": "2006-2019", "records": len(wang),
            "location": str(wang_path), "status": "ready for anonymous aggregation; individual-plant matching prohibited", "url_or_identifier": "10.1038/s41597-022-01439-7",
        },
    ]
    for year, url in MEE_URLS.items():
        path = next((RAW / "mee_wwtp_inventory" / str(year) / "archives").glob("*.pdf"))
        rows.append({"source": f"MEE WWTP inventory {year}", "role": "plant identity, operation date, design and mean daily flow", "coverage": str(year), "records": np.nan, "expected_official_facilities": MEE_EXPECTED_FACILITIES[year], "location": str(path), "status": "PDF structurally readable; OCR/staged-table extraction required", "url_or_identifier": url, **pdf_metadata(path)})
    rows.append({"source": "MEE WWTP inventory 2014", "role": "plant identity, operation date, design and mean daily flow", "coverage": "2014", "records": np.nan, "expected_official_facilities": MEE_EXPECTED_FACILITIES[2014], "location": str(mee2014_path), "status": "PDF structurally readable; OCR/staged-table extraction required", "url_or_identifier": "https://www.mee.gov.cn/gkml/hbb/bgg/201506/t20150609_303209.htm", **pdf_metadata(mee2014_path)})
    rows.append({"source": "PRB Chen spatial subset", "role": "candidate point-source master seed", "coverage": "2012", "records": len(anchor), "location": str(RUN / "inputs" / "model_ready" / "point_sources" / "chen_wwtp_2012_prb_anchor.gpkg"), "status": "ready as candidate locations; outfall-to-reach assignment still requires audit", "url_or_identifier": "derived"})
    return pd.DataFrame(rows)


def main() -> None:
    chen_path, wang_path, mee2014_path = stage_sources()
    for directory in (RUN / "inputs" / "staged", RUN / "inputs" / "model_ready" / "point_sources", RUN / "inputs" / "qa"):
        directory.mkdir(parents=True, exist_ok=True)
    chen = clean_chen(chen_path)
    wang = clean_wang(wang_path)
    anchor, reach_audit = chen_prb_anchor(chen)
    chen.to_parquet(RUN / "inputs" / "staged" / "chen_wwtp_2012.parquet", index=False)
    wang.to_parquet(RUN / "inputs" / "staged" / "wang_wwtp_activity_2006_2019.parquet", index=False)
    anchor.to_file(RUN / "inputs" / "model_ready" / "point_sources" / "chen_wwtp_2012_prb_anchor.gpkg", driver="GPKG")
    anchor.drop(columns="geometry").to_parquet(RUN / "inputs" / "model_ready" / "point_sources" / "chen_wwtp_2012_prb_anchor.parquet", index=False)
    reach_audit.to_csv(RUN / "inputs" / "qa" / "chen_prb_reach_assignment_audit.csv", index=False, encoding="utf-8-sig")
    grouping = ["year", "province", "facility_type", "process_category", "discharge_pathway"]
    summary = wang.groupby(grouping, dropna=False, as_index=False).agg(
        records=("wang_row_id", "size"), wastewater_volume_m3_yr=("wastewater_volume_m3_yr", "sum"),
        tn_removed_source_value=("tn_removed_source_value", "sum"), effluent_n2o_t_co2eq_yr=("effluent_n2o_t_co2eq_yr", "sum"),
    )
    summary.to_parquet(RUN / "inputs" / "model_ready" / "point_sources" / "wang_province_process_pathway_year_2006_2019.parquet", index=False)
    audit = source_audit(chen_path, wang_path, mee2014_path, chen, wang, anchor)
    audit.to_csv(RUN / "inputs" / "qa" / "source_audit.csv", index=False, encoding="utf-8-sig")
    qa = {
        "chen_rows": len(chen), "chen_unique_fid": int(chen.chen_fid.nunique()), "chen_prb_candidates": len(anchor),
        "chen_prb_total_tdn_2012_t_n_yr": float(anchor.tdn_2012_t_n_yr.sum()), "wang_rows": len(wang),
        "wang_year_rows": {str(int(y)): int(n) for y, n in wang.groupby("year").size().items()},
        "wang_summary_groups": len(summary), "mee_expected_facilities": MEE_EXPECTED_FACILITIES,
        "method_gate": "No plant-level 2006-2019 TN effluent load is produced until the N2O emission-factor mapping is verified from the original Wang methods/supplement and MEE PDFs are table-QAed.",
    }
    (RUN / "inputs" / "qa" / "source_audit_summary.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
