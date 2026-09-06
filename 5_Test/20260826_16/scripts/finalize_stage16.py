"""Finish Stage-16 prose artifacts without rerunning completed training."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260826_16"
REPORTS = RUN / "reports"


def main() -> None:
    decision = json.loads((REPORTS / "stage16_decision.json").read_text(encoding="utf-8"))
    runs = pd.read_parquet(RUN / "outputs" / "seed_run_summary.parquet")
    passing = [
        name for name, value in decision["candidates"].items()
        if value["status"] == "PASS_TEMPORAL_GATE"
    ]
    report = f"""# 20260826_16 Q-only静态DPL-HBV

状态：`{decision['status']}`。本阶段只使用2010–2015流量优化、2016 early stopping和2017–2018锁定时间评价；没有读取2019–2022或TN。

实际4018天递归基准为CPU 2.12秒、GPU 6.98秒，因此正式训练使用CPU。两个候选、三个种子共享完全相同的优化器、先验、边界与停止规则，epoch 0均严格复现父模型。

通过时间门的候选：`{passing}`。通过时间门只允许进入整棵河树零目标历史评价，不代表空间晋级，也不改变快/中/慢分量的未识别状态。

```text
{runs.to_string(index=False)}
```
"""
    (REPORTS / "technical_report.md").write_text(report, encoding="utf-8")
    (RUN / "README.md").write_text(
        "# 20260826_16\n\nQ-only static differentiable parameter-learning HBV temporal experiment. Spatial promotion is impossible in this folder.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
