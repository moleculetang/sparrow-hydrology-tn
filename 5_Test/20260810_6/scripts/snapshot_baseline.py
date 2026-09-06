from __future__ import annotations

import csv
from pathlib import Path
import shutil

from common import RUN, load_config, sha256_file, write_json
from runtime_guard import assert_sparrow_runtime


def required_baseline_files(base: Path) -> list[Path]:
    relative = [
        "config.json",
        "README.md",
        "validation.json",
        "inputs/covariate_backbone.parquet",
        "inputs/indata.parquet",
        "inputs/climate_corrected/cmfd/cmfd_monthly_by_reach_2006_2022.parquet",
        "inputs/topology/topology_edges.csv",
        "inputs/spatial_corrected/reaches_topology.shp",
        "inputs/spatial_corrected/reaches_topology.shx",
        "inputs/spatial_corrected/reaches_topology.dbf",
        "inputs/spatial_corrected/reaches_topology.prj",
        "inputs/spatial_corrected/reaches_topology.cpg",
        "inputs/spatial_corrected/reach_catchments.shp",
        "inputs/spatial_corrected/reach_catchments.shx",
        "inputs/spatial_corrected/reach_catchments.dbf",
        "inputs/spatial_corrected/reach_catchments.prj",
        "inputs/spatial_corrected/reach_catchments.cpg",
        "scripts/components/fit_monthly_bayes_seasonal_hysteresis.py",
        "scripts/runtime_guard.py",
    ]
    files = [base / item for item in relative]
    missing = [str(item) for item in files if not item.exists()]
    if missing:
        raise FileNotFoundError("Frozen baseline files missing:\n" + "\n".join(missing))
    return files


def main() -> None:
    runtime = assert_sparrow_runtime()
    config = load_config()
    base = Path(config["base_run"])
    files = required_baseline_files(base)
    snapshot = RUN / "inputs" / "baseline_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for source in files:
        relative = source.relative_to(base)
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hash = sha256_file(source)
        destination_hash = sha256_file(destination)
        if source_hash != destination_hash:
            raise RuntimeError(f"Hash mismatch after snapshot: {relative}")
        rows.append(
            {
                "relative_path": str(relative).replace("\\", "/"),
                "source_path": str(source),
                "snapshot_path": str(destination),
                "bytes": source.stat().st_size,
                "sha256": source_hash,
            }
        )
    manifest_dir = RUN / "inputs" / "baseline_manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with (manifest_dir / "baseline_snapshot.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        manifest_dir / "baseline_snapshot.json",
        {
            "run_id": config["run_id"],
            "base_run": str(base),
            "runtime": runtime,
            "file_count": len(rows),
            "files": rows,
        },
    )
    print(f"Snapshotted {len(rows)} frozen baseline files to {snapshot}")


if __name__ == "__main__":
    main()
