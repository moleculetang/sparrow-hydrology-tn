from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    reproduction = json.loads((ROOT / "reports" / "reproduction_gate.json").read_text(encoding="utf-8"))
    extrapolation = json.loads((ROOT / "terminal_gate.json").read_text(encoding="utf-8"))
    figures = []
    for stem in ["figure_1_temporal_extrapolation", "figure_2_large_station_hydrographs"]:
        paths = {suffix: ROOT / "figures" / f"{stem}.{suffix}" for suffix in ["svg", "pdf", "tiff", "png"]}
        svg_text = paths["svg"].read_text(encoding="utf-8")
        pdf_bytes = paths["pdf"].read_bytes()
        pdf_pages = len(re.findall(rb"/Type\s*/Page\b", pdf_bytes))
        with Image.open(paths["tiff"]) as image:
            tiff_size = list(image.size)
            dpi = [float(x) for x in image.info.get("dpi", (0, 0))]
        with Image.open(paths["png"]) as image:
            png_size = list(image.size)
        figures.append({
            "stem": stem,
            "all_formats_exist_nonempty": all(path.exists() and path.stat().st_size > 0 for path in paths.values()),
            "svg_contains_editable_text": "<text" in svg_text,
            "pdf_pages": pdf_pages,
            "tiff_pixels": tiff_size,
            "tiff_dpi": dpi,
            "png_pixels": png_size,
        })
    manifest_exists = (ROOT / "input_code_manifest.json").exists()
    checks = {
        "independent_reproduction_pass": reproduction["terminal"] == "P1_INDEPENDENT_REPRODUCTION_PASS",
        "prediction_difference_exact_zero": reproduction["max_abs_prediction_difference_cfs"] == 0.0,
        "temporal_extrapolation_complete": extrapolation["terminal"] == "TEMPORAL_EXTRAPOLATION_STABLE",
        "all_extrapolation_gates_true": all(extrapolation["hard_conditions"].values()),
        "input_code_manifest_exists": manifest_exists,
        "figure_formats_complete": all(row["all_formats_exist_nonempty"] for row in figures),
        "svg_text_editable": all(row["svg_contains_editable_text"] for row in figures),
        "pdf_single_page": all(row["pdf_pages"] == 1 for row in figures),
        "tiff_600_dpi": all(min(row["tiff_dpi"]) >= 599 for row in figures),
    }
    payload = {
        "terminal": "INDEPENDENT_P1_BASELINE_AND_2019_2022_VALIDATION_COMPLETE"
        if all(checks.values()) else "FINAL_DELIVERY_AUDIT_FAILURE",
        "checks": checks,
        "figures": figures,
        "scientific_terminal": extrapolation["terminal"],
        "evaluation_label": extrapolation["evaluation_label"],
    }
    (ROOT / "final_delivery_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["terminal"] == "FINAL_DELIVERY_AUDIT_FAILURE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
