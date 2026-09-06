from pathlib import Path
import json
import re
import unicodedata

import geopandas as gpd
import pandas as pd

ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260823_14"


def key(value):
    text = unicodedata.normalize("NFKC", str(value)).strip().replace(" ", "").replace("\u3000", "")
    text = text.replace("(", "（").replace(")", "）")
    if text.endswith("_2"):
        text = text[:-2] + "（二）"
    if text.endswith("_3"):
        text = text[:-2] + "（三）"
    if text.endswith("_4"):
        text = text[:-2] + "（四）"
    return re.sub(r"站$", "", text)


monthly = pd.read_parquet(RUN / "outputs" / "all_monthly_discharge_after_exclusions.parquet")
active = set(monthly.loc[monthly.Q_obsv_cfs.notna(), "station_norm"].astype(str))
rows = []
for label in ["existing", "all", "hydrostation"]:
    paths = list((ROOT / "0_reach_topology" / "data" / "raw" / "vector" / "stations" / label).glob("*.shp"))
    if len(paths) != 1:
        rows.append({"source": label, "error": f"found {paths}"})
        continue
    gdf = gpd.read_file(paths[0])
    candidates = [c for c in gdf.columns if c != "geometry" and gdf[c].dtype == object]
    name_col = next((c for c in ["Station", "STATION", "NAME", "station", "name"] if c in gdf.columns), candidates[0] if candidates else None)
    if name_col is None:
        rows.append({"source": label, "path": str(paths[0]), "crs": str(gdf.crs), "point_count": len(gdf), "columns": list(gdf.columns), "error": "no text name column"})
        continue
    keys = set(gdf[name_col].map(key))
    rows.append({
        "source": label, "path": str(paths[0]), "crs": str(gdf.crs), "point_count": len(gdf),
        "name_column": name_col, "unique_keys": len(keys), "active_exact_matches": len(keys & active),
        "active_unmatched": len(active - keys), "duplicate_keys": int(gdf[name_col].map(key).duplicated().sum()),
    })
(RUN / "reports" / "station_coordinate_source_inventory.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(rows, ensure_ascii=False, indent=2))
