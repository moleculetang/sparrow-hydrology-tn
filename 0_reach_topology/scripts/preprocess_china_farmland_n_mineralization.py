from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RAW_ROOT = ROOT / "0_reach_topology/data/raw/agriculture/nitrogen_legacy/china_farmland_n_mineralization_zenodo_19998062"
SOURCE = RAW_ROOT / "archives/Research Data(1).xlsx"
SOURCE_METADATA = RAW_ROOT / "metadata/zenodo_record_19998062.json"
OUT_ROOT = ROOT / "0_reach_topology/data/processed/agricultural_n_legacy_constraints/china_farmland_n_mineralization_zenodo_19998062"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def require_environment() -> None:
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if conda_env.lower() != "sparrow" and "envs\\sparrow" not in sys.executable.lower():
        raise RuntimeError(f"Run in conda environment 'sparrow', got {sys.executable!r} / {conda_env!r}")


def normalize_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    # Several unit symbols in the source headers are mojibake under different
    # spreadsheet engines.  The publisher's first 21 columns have a fixed
    # documented order, so bind those positions explicitly and retain the
    # original workbook unchanged in raw/archives.
    keep = [
        "source_row_id", "article_number", "province", "site_city", "region",
        "longitude_deg", "latitude_deg", "soil_use_type", "fertilization_type",
        "map_mm_year", "mae_mm_year", "mat_deg_c", "soc_mg_kg", "stn_mg_kg",
        "som_mg_kg", "cn_ratio", "ph", "incubation_temperature_deg_c",
        "incubation_days", "potential_mineralizable_n_mg_kg",
        "mineralizable_fraction_total_soil_n_percent",
    ]
    if frame.shape[1] < len(keep):
        raise RuntimeError(f"Metadata sheet has {frame.shape[1]} columns; expected at least {len(keep)}")
    out = frame.iloc[:, : len(keep)].copy()
    out.columns = keep
    for column in keep:
        if column not in {"province", "site_city", "region", "soil_use_type", "fertilization_type"}:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out["longitude_source_deg"] = out["longitude_deg"]
    out["latitude_source_deg"] = out["latitude_deg"]
    swap = (
        out["longitude_deg"].between(15.0, 55.0)
        & out["latitude_deg"].between(70.0, 140.0)
    )
    out.loc[swap, ["longitude_deg", "latitude_deg"]] = out.loc[
        swap, ["latitude_deg", "longitude_deg"]
    ].to_numpy()
    out["coordinate_swap_corrected"] = swap
    out["source_dataset"] = "zenodo_19998062"
    out["data_role"] = "external_mineralization_endpoint_not_reach_forcing"
    out["k_a_directly_observed"] = False
    return out


def normalize_effects(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.rename(
        columns={
            frame.columns[0]: "source_row_id",
            "Article Number": "article_number",
            "Publication Year": "publication_year",
            "Agricultural soil use type": "soil_use_type",
            "Agricultural soil fertilization type": "fertilization_type",
            "Sampling Time": "sampling_time",
            "yi": "effect_size_yi",
            "vi": "effect_size_variance",
        }
    ).copy()
    for column in ["source_row_id", "article_number", "publication_year", "effect_size_yi", "effect_size_variance"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["source_dataset"] = "zenodo_19998062"
    return out


def normalize_references(frame: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "Language": "language",
        "Article Number": "article_number",
        "Article": "article_title",
        "Journal": "journal",
        "Author": "authors",
        "Publication Year": "publication_year",
        "Volume(Issue)": "volume_issue",
        "Page": "pages",
    }
    out = frame.rename(columns=columns)[list(columns.values())].copy()
    out["article_number"] = pd.to_numeric(out["article_number"], errors="coerce")
    out["publication_year"] = pd.to_numeric(out["publication_year"], errors="coerce")
    for column in ["language", "article_title", "journal", "authors", "volume_issue", "pages"]:
        out[column] = out[column].astype("string")
    out["source_dataset"] = "zenodo_19998062"
    return out


def study_balanced_summary(trials: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "potential_mineralizable_n_mg_kg",
        "mineralizable_fraction_total_soil_n_percent",
        "incubation_days",
        "incubation_temperature_deg_c",
        "stn_mg_kg",
        "soc_mg_kg",
    ]
    by_study = trials.groupby("article_number", dropna=False)[metrics].median().reset_index()
    rows: list[dict[str, float | int | str]] = []
    for metric in metrics:
        values = by_study[metric].dropna().to_numpy(float)
        rows.append(
            {
                "metric": metric,
                "study_count": int(values.size),
                "study_balanced_mean": float(np.mean(values)) if values.size else np.nan,
                "study_balanced_median": float(np.median(values)) if values.size else np.nan,
                "p05": float(np.quantile(values, 0.05)) if values.size else np.nan,
                "p95": float(np.quantile(values, 0.95)) if values.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    require_environment()
    if not SOURCE.exists() or not SOURCE_METADATA.exists():
        raise FileNotFoundError("Zenodo source workbook or official metadata is missing")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    workbook = pd.ExcelFile(SOURCE)
    expected_sheets = ["Metadata", "Effects Size", "References"]
    if workbook.sheet_names != expected_sheets:
        raise RuntimeError(f"Unexpected workbook sheets: {workbook.sheet_names}")

    trials = normalize_metadata(pd.read_excel(SOURCE, sheet_name="Metadata"))
    effects = normalize_effects(pd.read_excel(SOURCE, sheet_name="Effects Size"))
    references = normalize_references(pd.read_excel(SOURCE, sheet_name="References"))
    summary = study_balanced_summary(trials)

    atomic_parquet(trials, OUT_ROOT / "mineralization_trials.parquet")
    atomic_parquet(effects, OUT_ROOT / "published_effect_sizes.parquet")
    atomic_parquet(references, OUT_ROOT / "references.parquet")
    atomic_parquet(summary, OUT_ROOT / "study_balanced_mineralization_summary.parquet")

    duplicate_trial_keys = int(trials.duplicated(["source_row_id"]).sum())
    missing_reference_ids = sorted(
        set(trials.article_number.dropna().astype(int))
        - set(references.article_number.dropna().astype(int))
    )
    ratio_check = (
        100.0 * trials.potential_mineralizable_n_mg_kg / trials.stn_mg_kg
        - trials.mineralizable_fraction_total_soil_n_percent
    ).abs()
    ratio_inconsistent = ratio_check > 0.01
    coordinate_valid = (
        trials.longitude_deg.between(70.0, 140.0)
        & trials.latitude_deg.between(15.0, 55.0)
    )
    qa = {
        "dataset_id": "china_farmland_n_mineralization_zenodo_19998062",
        "status": "PASS" if (
            len(trials) == 946
            and len(effects) == 197
            and len(references) == 95
            and duplicate_trial_keys == 0
            and not missing_reference_ids
            and bool(coordinate_valid.all())
        ) else "FAIL",
        "grain": {
            "trials": "one published experimental row",
            "effects": "one published effect-size row",
            "references": "one source article",
        },
        "counts": {
            "trials": int(len(trials)),
            "effect_sizes": int(len(effects)),
            "references": int(len(references)),
            "unique_articles_in_trials": int(trials.article_number.nunique()),
            "duplicate_trial_source_rows": duplicate_trial_keys,
            "trial_article_ids_missing_from_references": missing_reference_ids,
            "coordinate_swap_corrections": int(trials.coordinate_swap_corrected.sum()),
            "nmr_vs_100_n0_over_stn_inconsistent_gt_0_01_rows": int(ratio_inconsistent.sum()),
            "nmr_ratio_inconsistency_article_ids": sorted(
                trials.loc[ratio_inconsistent, "article_number"].dropna().astype(int).unique().tolist()
            ),
        },
        "validity": {
            "longitude_range": [float(trials.longitude_deg.min()), float(trials.longitude_deg.max())],
            "latitude_range": [float(trials.latitude_deg.min()), float(trials.latitude_deg.max())],
            "incubation_days_range": [float(trials.incubation_days.min()), float(trials.incubation_days.max())],
            "nmr_percent_range": [
                float(trials.mineralizable_fraction_total_soil_n_percent.min()),
                float(trials.mineralizable_fraction_total_soil_n_percent.max()),
            ],
            "coordinates_all_within_broad_china_bounds_after_explicit_swap_correction": bool(coordinate_valid.all()),
            "nmr_vs_100_n0_over_stn_max_abs": float(ratio_check.max()),
        },
        "scientific_role": {
            "allowed": [
                "study-balanced external mineralization endpoint",
                "prior-predictive check for potential mineralizable N and soil/environment strata",
            ],
            "forbidden": [
                "direct 230-Reach forcing",
                "claiming that Nmr is an observed Active-pool first-order k_A",
                "using rows both to set a prior and claim independent validation",
            ],
            "critical_semantic_note": "Nmr is the authors' reported mineralizable fraction endpoint. It is close to 100*N0/STN for most rows but differs in 10 rows from article 71, so no identity is imposed. In all cases it is not a directly observed first-order turnover rate of the model Active/Fresh pool.",
        },
        "source": {
            "data_doi": "10.5281/zenodo.19998062",
            "paper_doi": "10.1007/s11104-026-08649-7",
            "license": "CC BY 4.0",
            "workbook_sha256": sha256(SOURCE),
            "metadata_sha256": sha256(SOURCE_METADATA),
        },
        "outputs": {
            path.name: sha256(path)
            for path in sorted(OUT_ROOT.glob("*.parquet"))
        },
    }
    atomic_json(qa, OUT_ROOT / "qa.json")
    print(json.dumps({"status": qa["status"], "output": str(OUT_ROOT)}, ensure_ascii=False, indent=2))
    if qa["status"] != "PASS":
        raise RuntimeError("Mineralization dataset QA failed")


if __name__ == "__main__":
    main()
