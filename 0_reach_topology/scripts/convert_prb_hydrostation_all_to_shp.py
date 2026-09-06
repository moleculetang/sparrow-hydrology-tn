from pathlib import Path
import re

import geopandas as gpd
import pandas as pd


INPUT_XLSX = Path(r"F:\VIEIRA\PRB_Preprocessing") / ("PRB-hydrostation" + "\u5168" + ".xlsx")
OUTPUT_DIR = Path(r"E:\SPARROW\0_reach_topology\data\raw\vector\stations\all")
OUTPUT_BASE = "PRB" + "\u6c34\u6587\u7ad9" + "_" + "\u5168\u90e8"
OUTPUT_SHP = OUTPUT_DIR / f"{OUTPUT_BASE}.shp"
AUDIT_CSV = OUTPUT_DIR / f"{OUTPUT_BASE}_audit.csv"


def parse_dms(value):
    """Parse degree-minute-second-like strings to decimal degrees."""
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    if not numbers:
        return None

    parts = [float(x) for x in numbers[:3]]
    deg = parts[0]
    minute = parts[1] if len(parts) >= 2 else 0.0
    second = parts[2] if len(parts) >= 3 else 0.0
    sign = -1.0 if "-" in text or any(x in text.upper() for x in ("W", "S")) else 1.0
    return sign * (abs(deg) + minute / 60.0 + second / 3600.0)


def remove_existing_shapefile(path):
    suffixes = [".shp", ".shx", ".dbf", ".prj", ".cpg", ".sbn", ".sbx", ".qix", ".fix", ".shp.xml"]
    stem = path.with_suffix("")
    for suffix in suffixes:
        candidate = stem.with_suffix(suffix)
        if candidate.exists():
            candidate.unlink()


def main():
    if not INPUT_XLSX.exists():
        raise FileNotFoundError(f"Input workbook not found: {INPUT_XLSX}")

    df = pd.read_excel(INPUT_XLSX, header=None, names=["Station", "Lon_raw", "Lat_raw"])
    df = df.dropna(how="all").copy()
    df["Station"] = df["Station"].astype(str).str.strip()
    df["Lon_dd"] = df["Lon_raw"].map(parse_dms)
    df["Lat_dd"] = df["Lat_raw"].map(parse_dms)

    df["valid_coord"] = (
        df["Lon_dd"].between(70, 140, inclusive="both")
        & df["Lat_dd"].between(15, 55, inclusive="both")
    )
    audit = df.copy()
    audit.to_csv(AUDIT_CSV, index=False, encoding="utf-8-sig")

    valid = df[df["valid_coord"]].copy()
    if valid.empty:
        raise ValueError("No valid station coordinates were parsed from the workbook.")

    gdf = gpd.GeoDataFrame(
        valid[["Station", "Lon_raw", "Lat_raw", "Lon_dd", "Lat_dd"]].copy(),
        geometry=gpd.points_from_xy(valid["Lon_dd"], valid["Lat_dd"]),
        crs="EPSG:4326",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    remove_existing_shapefile(OUTPUT_SHP)
    gdf.to_file(OUTPUT_SHP, driver="ESRI Shapefile", encoding="UTF-8")

    read_back = gpd.read_file(OUTPUT_SHP)
    print(f"input_rows={len(df)}")
    print(f"valid_points={len(gdf)}")
    print(f"invalid_rows={len(df) - len(gdf)}")
    print(f"output={OUTPUT_SHP}")
    print(f"audit={AUDIT_CSV}")
    print(f"crs={read_back.crs}")
    print(f"bounds={tuple(round(x, 6) for x in read_back.total_bounds)}")


if __name__ == "__main__":
    main()
