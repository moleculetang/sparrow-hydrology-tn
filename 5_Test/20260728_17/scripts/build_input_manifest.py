from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = Path(__file__).resolve().parents[1]
OUT = RUN / "inputs_manifest"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shapefile_parts(path: Path) -> list[Path]:
    return sorted(path.parent.glob(f"{path.stem}.*"))


def source_files() -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []

    exact = [
        ("topology", ROOT / "0_reach_topology" / "results" / "tables" / "reach_summary.csv"),
        ("topology", ROOT / "0_reach_topology" / "results" / "tables" / "topology_edges.csv"),
        (
            "precipitation",
            ROOT
            / "0_reach_topology"
            / "data"
            / "processed"
            / "rainfall2_prb"
            / "chm_pre_v2_monthly_by_reach_2006_2022.csv",
        ),
        (
            "cmfd",
            ROOT
            / "0_reach_topology"
            / "data"
            / "processed"
            / "cmfd_prb"
            / "cmfd_monthly_by_reach_2006_2022.csv",
        ),
        (
            "early_discharge",
            ROOT
            / "1_Inputs"
            / "DischargeData"
            / "monthly_mean_2006_2009"
            / "DischargeData_2006_2009.xlsx",
        ),
    ]
    rows.extend(exact)

    for path in shapefile_parts(
        ROOT / "0_reach_topology" / "results" / "vectors" / "reach_catchments.shp"
    ):
        rows.append(("reach_catchments", path))
    for path in shapefile_parts(
        ROOT / "0_reach_topology" / "data" / "raw" / "vector" / "PRB水文站_全部.shp"
    ):
        rows.append(("station_locations", path))

    era5 = (
        ROOT
        / "0_reach_topology"
        / "data"
        / "raw"
        / "era5_land"
        / "monthly_prb_buffer"
    )
    for year in range(2006, 2023):
        rows.append(("era5_aet", era5 / f"era5_land_monthly_prb_buffer_{year}.nc"))

    discharge_root = ROOT / "1_Inputs" / "DischargeData"
    for label in ("complete_2010_2022", "noncomplete_2010_2022"):
        for path in sorted((discharge_root / label).rglob("*.csv")):
            rows.append((label, path))

    for path in sorted((RUN / "inputs" / "source_metadata").glob("*")):
        if path.is_file():
            rows.append(("run_source_metadata", path))

    unique: dict[str, tuple[str, Path]] = {}
    for category, path in rows:
        unique[str(path.resolve())] = (category, path)
    return [unique[key] for key in sorted(unique)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    missing: list[str] = []
    for category, path in source_files():
        if not path.exists():
            missing.append(str(path))
            records.append(
                {
                    "category": category,
                    "path": str(path),
                    "exists": False,
                    "size_bytes": None,
                    "modified_utc": None,
                    "sha256": None,
                }
            )
            continue
        stat = path.stat()
        records.append(
            {
                "category": category,
                "path": str(path),
                "exists": True,
                "size_bytes": int(stat.st_size),
                "modified_utc": datetime.utcfromtimestamp(stat.st_mtime).isoformat(
                    timespec="seconds"
                ),
                "sha256": sha256(path),
            }
        )

    frame = pd.DataFrame(records)
    frame.to_csv(
        OUT / "source_file_inventory.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "run_id": RUN.name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "file_count": int(len(frame)),
        "existing_file_count": int(frame["exists"].sum()),
        "missing_file_count": int((~frame["exists"]).sum()),
        "total_bytes": int(frame["size_bytes"].fillna(0).sum()),
        "missing_files": missing,
    }
    (OUT / "source_manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
