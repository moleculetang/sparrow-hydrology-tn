from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from runtime_guard import assert_sparrow_runtime


RUNTIME = assert_sparrow_runtime()
RUN = Path(__file__).resolve().parents[1]
BACKUP = RUN / "backup_before_correction"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    rows = []
    for path in sorted(p for p in BACKUP.rglob("*") if p.is_file()):
        rows.append({"relative_path": str(path.relative_to(RUN)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    external = [
        Path(r"E:\SPARROW\0_reach_topology\work\rasters\flow_dir.tif"),
        Path(r"E:\SPARROW\0_reach_topology\data\raw\rainfall_2\CHM_PRE V2\monthly\CHM_PRE_V2_monthly.nc"),
        Path(r"E:\SPARROW\0_reach_topology\data\raw\CMFD\temp_CMFD_V0200_B-01_01mo_010deg_195101-202412.nc"),
        Path(r"E:\SPARROW\0_reach_topology\data\raw\era5_land\monthly_prb_buffer\era5_land_monthly_prb_buffer_2006.nc"),
    ]
    ext_rows = [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in external]
    pd.DataFrame(rows).to_csv(RUN / "backup_manifest.csv", index=False, encoding="utf-8-sig")
    payload = {"runtime": RUNTIME, "backup_file_count": len(rows), "backup_bytes": sum(r["bytes"] for r in rows), "files": rows, "external_large_sources": ext_rows}
    (RUN / "backup_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"backup_file_count": len(rows), "backup_bytes": payload["backup_bytes"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
