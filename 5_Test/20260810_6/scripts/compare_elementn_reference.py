from __future__ import annotations

import json

from scipy.io import loadmat
import numpy as np

from common import RUN, write_json
from runtime_guard import assert_sparrow_runtime


MAPPINGS = {
    "ma": "Ma_s_t",
    "mp": "Mp_s_t",
    "mmin": "Mmin_s_t",
    "leaching_rate": "Js_s_t_unit_area_rate",
    "subsurface_load": "Subsurface_N_load_t_unit_area",
    "wastewater_load": "WWTP_N_load_t_unit_area",
    "total_load": "Total_N_load_Mout_t_unit_area",
}


def main() -> None:
    runtime = assert_sparrow_runtime()
    python_result = np.load(RUN / "outputs" / "elementn_python_reference.npz")
    matlab_result = loadmat(RUN / "outputs" / "elementn_matlab_reference.mat")
    checks: dict[str, dict[str, float]] = {}
    for python_name, matlab_name in MAPPINGS.items():
        py = np.asarray(python_result[python_name]).squeeze()
        ml = np.asarray(matlab_result[matlab_name]).squeeze()
        if py.shape != ml.shape:
            raise AssertionError(f"Shape mismatch for {python_name}: {py.shape} != {ml.shape}")
        absolute = float(np.max(np.abs(py - ml)))
        scale = max(float(np.max(np.abs(ml))), 1.0)
        relative = absolute / scale
        checks[python_name] = {"max_absolute_error": absolute, "max_relative_error": relative}
    max_relative = max(item["max_relative_error"] for item in checks.values())
    summary = {"runtime": runtime, "checks": checks, "max_relative_error": max_relative, "pass": max_relative < 1.0e-8}
    write_json(RUN / "reports" / "elementn_equivalence_gate.json", summary)
    if not summary["pass"]:
        raise AssertionError(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
