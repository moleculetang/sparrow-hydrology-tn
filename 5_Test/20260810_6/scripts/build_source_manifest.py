from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from common import RUN, load_config, sha256_file, write_json
from runtime_guard import assert_sparrow_runtime


REQUIRED = {
    "hani_gross_inputs_processed": {
        "doi": "10.1594/PANGAEA.942069",
        "license": "CC BY 4.0",
        "purpose": "1860-2019 fertilizer, manure and atmospheric deposition",
        "expected": "inputs/processed/historical_agricultural_n_inputs_hani_1860_2019.parquet",
    },
    "hyde32_processed": {
        "doi": "HYDE 3.2.1",
        "license": "verify at download",
        "purpose": "historical population and cropland/pasture fractions",
        "expected": "inputs/processed/historical_land_use_population_hyde32_1860_2019.parquet",
    },
    "net_surplus_components_missing": {
        "doi": "crop removal source TBD; 10.5281/zenodo.7133340 (BNF); 10.5061/dryad.hx3ffbgkh (national budget constraint)",
        "license": "source-specific",
        "purpose": "crop harvest N removal and biological fixation required before gross inputs can become a Legacy-N net surplus",
        "expected": "inputs/raw/net_surplus_components",
    },
    "soilgrids": {
        "doi": "SoilGrids 2.0",
        "license": "ODbL",
        "purpose": "0-100 cm nitrogen, SOC and bulk-density SON priors",
        "expected": "inputs/processed/soilgrids_son_priors_by_reach.parquet",
    },
    "hydrowaste": {
        "doi": "10.6084/m9.figshare.14847786.v1",
        "license": "CC BY 4.0",
        "purpose": "modern wastewater treatment plant locations and service attributes",
        "expected": "inputs/raw/hydrowaste/extracted/HydroWASTE_v10.csv",
    },
    "reservoir_inventory": {
        "doi": "HydroLAKES / Global Dam Watch / verified GRanD mapping",
        "license": "source-specific",
        "purpose": "reservoir area and storage for TN attenuation",
        "expected": "inputs/raw/reservoir_inventory",
    },
}


def main() -> None:
    runtime = assert_sparrow_runtime()
    config = load_config()
    rows: list[dict[str, object]] = []
    for name, metadata in REQUIRED.items():
        expected = RUN / str(metadata["expected"])
        if expected.is_file():
            files = [expected]
        elif expected.is_dir():
            files = [item for item in expected.rglob("*") if item.is_file()]
        else:
            files = []
        rows.append(
            {
                "name": name,
                "doi": metadata["doi"],
                "license": metadata["license"],
                "purpose": metadata["purpose"],
                "expected_path": str(expected),
                "status": "ready" if files else "missing",
                "files": [
                    {"path": str(item.relative_to(RUN)).replace("\\", "/"), "bytes": item.stat().st_size, "sha256": sha256_file(item)}
                    for item in files
                ],
            }
        )
    payload = {
        "run_id": config["run_id"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime,
        "sources": rows,
    }
    write_json(RUN / "inputs" / "baseline_manifest" / "source_manifest.json", payload)
    with (RUN / "inputs" / "baseline_manifest" / "source_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=["name", "doi", "license", "purpose", "expected_path", "status", "file_count", "bytes"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**{key: row[key] for key in writer.fieldnames[:6]}, "file_count": len(row["files"]), "bytes": sum(item["bytes"] for item in row["files"])})
    print(json.dumps({"ready": [row["name"] for row in rows if row["status"] == "ready"], "missing": [row["name"] for row in rows if row["status"] == "missing"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
