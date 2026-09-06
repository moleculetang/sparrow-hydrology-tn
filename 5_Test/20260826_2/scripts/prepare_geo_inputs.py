from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW")
SOURCE = ROOT / "5_Test" / "20260814_9" / "inputs" / "model_ready" / "static" / "reach_static_attributes.parquet"
TARGET = ROOT / "5_Test" / "20260826_2" / "inputs" / "parent_static_snapshot.csv"

TARGET.parent.mkdir(parents=True, exist_ok=True)
frame = pd.read_parquet(SOURCE).sort_values("reach_id").reset_index(drop=True)
if len(frame) != 230 or frame.reach_id.nunique() != 230:
    raise RuntimeError("Expected 230 unique parent static rows")
frame.to_csv(TARGET, index=False)
print(TARGET)
