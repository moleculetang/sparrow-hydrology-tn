# 代码导航、运行与审阅

## 真实调用链

从 `snapshot/scripts/campaign_model.py:for_job` 开始：加载本折数组、train.parquet和design → `make_model(kind='D29_BE')` → `HFEndpoints` → `TemporalEndpoints` → `Endpoints` → `Matched` → `ScientificModel`。逐日物理状态仍由 `Transport` 和 `RiverN` 完整递推，不通过观测插补状态。

|问题|代码入口|
|---|---|
|v3载入、正式前缀及2025切片|[bootstrap_global](../snapshot/scripts/bootstrap_global.py)、[verify_extension_contract](../snapshot/scripts/verify_extension_contract.py)|
|训练站月并集、空间闭包、尺度、标签血缘|[prepare_global](../snapshot/scripts/prepare_global.py) `main/closure`|
|异常阈值、日覆盖及按实际次数月均|[prepare_hf](../snapshot/scripts/prepare_hf.py) `classify/products`；本轮只调用这些辅助，不运行旧七站main|
|数据身份、文件封存|[seal_global](../snapshot/scripts/seal_global.py)、[native_runtime](../snapshot/scripts/native_runtime.py)|
|字段白名单、日边界、月聚合|[temporal_model](../snapshot/scripts/temporal_model.py) `clean_metadata/daily_routing_view/aggregate_daily/TemporalEndpoints`|
|日观测索引|[hf_model](../snapshot/scripts/hf_model.py) `HFEndpoints.daily_metadata`|
|β端点及固定17归一化|[campaign_model](../snapshot/scripts/campaign_model.py) `Endpoints/Matched/build_design/make_model`|
|区域映射、原始参数界、基础先验|[model](../snapshot/vendor/expert/tn_challenge/model.py) `Predictor/Objective`|
|8维D29乘子及先验|[scientific_models](../snapshot/vendor/transfer_research/scientific_models.py) `ScientificModel`|
|M/L完整前向和伴随|[closures](../snapshot/vendor/research/closures.py) `scan/reverse/Transport`|
|被动来源及比例反馈导数|[tagged_transport](../snapshot/vendor/research/tagged_transport.py) `tag_scan/tag_reverse`|
|河网、水库、站界及伴随|[routing](../snapshot/vendor/expert/tn_challenge/routing.py) `route/RiverN/boundary_mass`|
|OU积分及解析导数|[support_integral](../snapshot/vendor/expert/tn_challenge/support_integral.py) `phi/coefficients/BoundaryFactor`|
|拟合、标签屏障、合法保存点|[fit_worker](../snapshot/scripts/fit_worker.py)、[serial_solvers](../snapshot/scripts/serial_solvers.py)|
|派发、资源、恢复与退出审计|[campaign_controller](../snapshot/scripts/campaign_controller.py)、[recover_controller](../snapshot/scripts/recover_controller.py)、[audit_job](../snapshot/scripts/audit_job.py)|
|完整域/公共边界验收|`validate_global.py`, `validate_global_inputs.py`, `validate_common_domain.py`, `validate_common_outputs.py`, `test_boundaries.py`|
|冻结预测和评价|`finalize_global.py`, `hf_metrics.py`, `recompute_selected_routing.py`, `supplement_global.py`|
|专家解读与独立审计|`report_global.py`, `synthesize_global.py`, `independent_global.py`, `account_computation.py`|

保留完整脚本目录，是为了保存运行与恢复证据，**不要按文件名排序全部执行**。`bootstrap.py/prepare_hf.main/seal_hf/validate_small/validate_mix/finalize_hf`等为历史平台入口；`recover_*`是事件恢复工具，不应在新运行中默认执行。vendor内其他候选模型及OS-MIX代码为继承依赖，28条拟合唯一选择D29_BE/OU。

## 无私有数据可运行的检查

本包另加测试，不修改任何冻结算法文件。使用已安装的conda sparrow环境，仓库根目录运行：

```powershell
conda run --no-capture-output -n sparrow python -B experiments/global_tn_20260916/tests/test_public_kernels.py
```

需要NumPy、pandas、PyTorch、Numba；精确实测版本见 [requirements-tested.txt](../requirements-tested.txt) 和 [environment.json](../evidence/environment.json)。文件是环境记录，不保证任何平台的二进制wheel都可直接安装。代码明确断言环境目录名为sparrow。测试真实内核的OU解析极限/细分/导数、M/L和四源和守恒、完整合成历史方向导数、共用水库只释放一次及反向、MATCH权重梯度、η分解、零水拒绝、元数据去标签和全部快照哈希/语法。

这些测试只加载人工数组，不访问本地监测归档。完整1961—2024梯度、116站评价及资源压力需要真实数据，不能从合成测试外推。公开证据JSON是原实验结果，专家仍可要求在受控数据环境独立重算。

## 完整重算的必要条件

这是一份**精确代码审阅快照，不是下载后即能全量训练的发行版**。`native_runtime`使用Windows kernel32/psapi；控制器未移植Linux。源码保留原相对目录契约：新的运行目录位于`SPARROW/5_Test`下，v3、20260915_5、auto_4h依原层级可读。部分恢复和历史脚本还记录原绝对路径；不可指向其他路径后声称哈希相同。

数据持有者需提供：v3的release_loader和完整catalog所指数组/拓扑、原始月报模型表；auto_4h canonical分区和空间/对应登记；20260915_5的冻结支持、参考元数据、预处理、公共域数组及配置。所有源先校验哈希。原始数据不随PR发布，取得权限与交付渠道须由数据持有者安排。

在独立新运行目录复制snapshot代码后，按以下阶段重新生成，不将 `evidence/configs` 当作当前时间的可执行工作配置：

1. `bootstrap_global.py`：建时钟、数组视图及初始配置（拒绝已经初始化的目录）。
2. `prepare_global.py`：建本折训练表、目标权重、登记与jobs；随后核验输入、2025合同、全域及公共域回归。
3. `seal_global.py`：封存身份；按controller要求完成其余启动验收和实测资源画像。查看该脚本和controller对启动报告的精确依赖，不跳过门禁。
4. `campaign_controller.py`：28条依赖路径；每次退出由audit_job独立重算。恢复程序只在对应异常存在且身份检查通过时运行。
5. 参数与预测冻结后运行global后处理、敏感性、报告、independent_global和计算核算。保留逐子任务退出与资源追加日志。

本PR没有偷偷补齐缺失数组、重写输入哈希或用旧拟合点代替新训练。公开 `source_manifest` 是发布身份；本地训练的sealed manifest则是科学输入身份，二者用途不同。

## 发布边界

仅源码、文档、配置、汇总审计、形状与哈希入Git；禁止raw标签、parquet/npy/pickle、模型检查点或数据压缩包。既有仓库没有在此PR中新授数据或代码许可证；公开可见性不应被解读为原始数据授权。原研究报告中的实验相对链接指向未公开产物时，需要在本地归档查阅。
