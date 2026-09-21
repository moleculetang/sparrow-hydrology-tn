# 实际入口和依赖

## 优先审阅20_6的原D29与候选

`snapshot/20260920_6/scripts/campaign_model.py:for_job` → `make_model` → `HFEndpoints` → `TemporalEndpoints` → `Endpoints` → `Matched` → `ScientificModel` → `Predictor`。状态调制在`StateModulated.flux_parameters`封装父方法，不改变M/L核。

|职责|相对20_6快照的代码|
|---|---|
|折内清洗、并集与权重|scripts/prepare.py、prepare_hf.py|
|H1数组/β映射/17归一化|scripts/campaign_model.py|
|基础风险、区域映射、先验与目标|vendor/expert/tn_challenge/model.py|
|D29软饱和8基函数|vendor/transfer_research/scientific_models.py；vendor/research/closures.py|
|M/L前向与完整伴随|vendor/research/closures.py:scan/reverse/Transport|
|被动来源与反馈导数|vendor/research/tagged_transport.py|
|河网、水库、站界与伴随|vendor/expert/tn_challenge/routing.py、support_integral.py；vendor/research/mix_routing.py|
|日读出、月聚合、标签白名单|scripts/temporal_model.py、hf_model.py|
|候选风险乘子|scripts/state_modulated.py|
|拟合、嵌套保存点、优化器|scripts/fit_worker.py、serial_solvers.py|
|完整历史验收/独立重算|scripts/preflight.py、extended_acceptance.py、independent_audit.py|
|同支持配对与重采样|scripts/evaluation.py、audit_evaluation.py|
|新增方向与二阶诊断|scripts/direction_diagnostic.py|

17_5活跃准备入口为prepare_bridge/build_domain，18_1为bootstrap_fct8/prepare_fct8及fct8_model。它们目录中保留的global、HF旧平台main、2025和其他kind辅助代码不应逐个执行；实际配置以对应configs/jobs.json为准。恢复脚本不是新实验入口。

19_4的work/xi.py、19_5的work/xi_k.py、20_1/20_2的work/dp_kernel.py与closures_dp*.py、20_3的scripts/kernel.py、20_4的bypass_kernel.py、20_5的family_kernel.py是结构审阅重点。runtime/common模块记录实际导入路径，不能因同名模块存在就任意互换。

## H1代码血缘

17_3/b1_run_product → dependencies/5_Test/20260917_2/scripts/a2h_refit_arm → 17_3/scripts/forks的父拟合、区域化和spinup，再调用28_9/28_34/28_35的导出、长期水文和TN接口。17_5/build_domain先注入本轮audit_inputs，再调用fc_data/fc_legacy，从而生成H1全部配套量。

dependencies保存这些跨轮算法以及结构筛查依赖的16_2代码，不代表本次审阅新增那些实验。evidence/import_inventory.json列出静态导入；dependency_resolved.json记录同名模块的人工核对选择。import表是静态辅助，不冒充每个历史辅助脚本都曾执行的动态trace。

源快照保留原Windows路径和冻结哈希。公开适配工具仅处理路径，具体变更在REPRODUCTION.md登记；不要直接从任意目录启动历史controller。
