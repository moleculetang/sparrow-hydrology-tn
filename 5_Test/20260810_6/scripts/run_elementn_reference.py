from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import RUN, load_config, write_json
from legacy_n.elementn_reference import run_elementn_reference
from runtime_guard import assert_sparrow_runtime


def main() -> None:
    runtime = assert_sparrow_runtime()
    config = load_config()
    result = run_elementn_reference(Path(config["elementn_root"]))
    outputs = RUN / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outputs / "elementn_python_reference.npz",
        years=result.years,
        ma=result.ma,
        mp=result.mp,
        mmin=result.mmin,
        leaching_rate=result.leaching_rate,
        subsurface_load=result.subsurface_load,
        wastewater_load=result.wastewater_load,
        total_load=result.total_load,
    )
    aggregate = pd.DataFrame(
        {
            "model_year": result.years,
            "ma_mean_kg_ha": result.ma.mean(axis=0),
            "mp_mean_kg_ha": result.mp.mean(axis=0),
            "mmin_mean_kg_ha": result.mmin.mean(axis=0),
            "leaching_mean_kg_ha_year": result.leaching_rate.mean(axis=0),
            "subsurface_kg_ha_year": result.subsurface_load,
            "wastewater_kg_ha_year": result.wastewater_load,
            "total_kg_ha_year": result.total_load,
        }
    )
    aggregate.to_csv(outputs / "elementn_python_reference.csv", index=False, encoding="utf-8-sig")
    summary = {
        "runtime": runtime,
        "years": int(len(result.years)),
        "s_bins": int(result.ma.shape[0]),
        "total_final_kg_ha_year": float(result.total_load[-1]),
        "subsurface_final_kg_ha_year": float(result.subsurface_load[-1]),
        "wastewater_final_kg_ha_year": float(result.wastewater_load[-1]),
    }
    write_json(RUN / "reports" / "elementn_python_reference.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
