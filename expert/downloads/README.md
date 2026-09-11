# TN专家拟合包：完整ZIP下载

[直接下载 tn_challenge_24fcf199.zip](https://github.com/moleculetang/sparrow-hydrology-tn/raw/refs/heads/codex/review-shared-closure-20260911/expert/downloads/tn_challenge_24fcf199.zip)

此ZIP完整封装PR #2中 **`24fcf199b2d5e922de750905f011a774c899e012`** 的 `expert/tn_challenge`，包括25个实际NPZ、CSV、模型代码、拟合/评价入口和参照验证，共91文件；没有Git LFS指针，也不需要逐个下载NPZ。文件大小 **74,894,313 bytes（74.89 MB）**。

SHA256：

```text
4ab8fd41046cd1983b5a5cc762a32a11073c7594847513ae2bdb1abd8ebedd4a
```

另附 [校验和文件](tn_challenge_24fcf199.zip.sha256) 和 [解压独立运行验证](tn_challenge_24fcf199.validation.json)。所有归档文件与原版本清单逐项一致，解压后在conda `sparrow` 中运行 `verify.py` 已通过。此次仅解决数据下载交付，不改变模型、参数或科学结果。

解压到一个新文件夹后，**在该文件夹根目录**执行：

```powershell
conda --no-plugins run -n sparrow python expert/tn_challenge/verify.py
conda --no-plugins run -n sparrow python expert/tn_challenge/run.py fit --variant M0 --start 0 --max-calls 1000 --seconds 3600 --out expert/tn_challenge/runs/m0_0
```

若没有 `sparrow` 环境，先执行 `conda env create -f expert/tn_challenge/environment.yml`。完整拟合方法、输入单位及限制在ZIP内README和DATA_DICTIONARY。ZIP保留 `expert/tn_challenge/` 目录结构；其README引用的包外历史报告可在 [PR #2](https://github.com/moleculetang/sparrow-hydrology-tn/pull/2) 和 [失败原因报告](https://github.com/moleculetang/sparrow-hydrology-tn/blob/24fcf199b2d5e922de750905f011a774c899e012/expert/WHY_MANY_ROUNDS_FAILED.md) 查阅，不是运行该子集的依赖。

17个主评价站中，2023和2024分别仅12站满足年度NSE条件。缺月站没有被删除，所有原始留出记录继续参与适用的误差统计；不将17站全部算作年度NSE分母。这个ZIP是原版本的可运行交付，不是新训练出的模型。
