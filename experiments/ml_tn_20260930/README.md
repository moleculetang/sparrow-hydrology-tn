# 纯机器学习TN：20260930_1完整实验审阅

本目录公开已完成实验的运行源码、冻结配置、完整逐站汇总、文献证据与专家报告。200个有效筛选身份、189条正式逻辑路径全部完成；45条空间2024路径复用合法2023检查点。11类学习模型全部尝试，失败与替代保留。它是回顾性审阅包，不是数据完整的预测部署产品。

## 先读结果

- [专家机器学习与全域验证报告](reports/专家机器学习与全域验证报告.md)
- [振幅控制因子数值诊断](reports/振幅控制因子数值诊断.md)
- [实际方法与偏离](reports/实际方法与偏离.md)
- [模型路径与观测支持说明](reports/模型路径与观测支持说明.md)
- [全配置统计](outputs/expert_review/configuration_summary.csv)、[逐站指标](outputs/evaluation/all_station_metrics/INDEX.md)、[配对抽样汇总](outputs/expert_review/paired_uncertainty_summary/INDEX.md)

训练截止2022/2023后评价2023/2024。内部验证选择的直接CatBoost三种子均值官方月NSE中位数0.1726/0.2887，HF日0.1725/0.3325，月内0.1674/0.2092；联合GraphTCN日0.4037/0.1906，没有两折一致胜出。官方月116站中分别78/85站满足年度NSE覆盖门，日15站。230河段读出不等于230个实测验证站。

## 代码与设计

`mltn/models.py`包含树与MLP/LSTM/GRU/因果TCN/Transformer/GraphTCN；图模型使用冻结有向一跳驱动视图与因果TCN，不能称任意可学习图网络。`train.py`执行月/日直接预测；`joint.py`实现0.8官方月报+0.2 HF月内异常的联合日模型。`auxiliary.py`、`mltn/history.py`隔离历史TN可见性。NH4/DO、灰箱预测、氮库存与反推输入排除。

`config/design.json`、筛选/正式清单及`outputs/frozen_*selection.json`固定配置选择；2022年1—9月选配置、10—12月定集成。2023/2024不用于挑模型。S23/S24为已知拓扑上移除支流及下游缓冲全部标签的区域检验。源脚本逐字节保留，准备与一次性迁移脚本内旧目录路径属于当时私有工程依赖，不能在任意克隆上直接启动。

联合树种子转发修复已经真实重训原矩阵；当前XGBoost选定c3，旧c7及重复种子结果不再是正式证据。`repair_joint_seed.py`是封存修复记录，不应再次运行覆盖实验。公开修复回执与失败台账供检查。

## 无私有数据的验收

Python3.11，CPU；依赖见[requirements-review.txt](requirements-review.txt)。PyTorch按本机/设备安装官方兼容构建，不将CUDA版本写成通用依赖保证。环境原版本记录在`evidence/environment_*.json`；本次实测版本在`tests/public_validation.json`。

```text
cd experiments/ml_tn_20260930
python -m pip install -r requirements-review.txt
python -B tests/run_public.py
python -B tests/verify_publication.py
```

测试调用真实聚合目标/梯度、网络因果核、恢复和合成拟合入口；文件屏障禁止读取私有实验，合成临时文件单独清理。GPU可用时固定权重门检查CPU/GPU，但不重新训练真实路径。源码/表格/文档哈希、CSV分片行数与报告内部链接另核对。测试通过不等于真实TN改善或全局最优。

## 完整复现和部署边界

真实运行仍须[私有输入契约](docs/REPRODUCTION.md)中的冻结特征、站界、标签及父检查点；不下载、不插补TN、不提供假实测默认值。原`acceptance.py`含真实输入验收，需要私有文件；公开测试只选其中可独立运行的合成门。`prepare.py`需要原H1、独立产品及认证支持。

`controller.py`和`mltn/resources.py`为原部署源码，需机器侧共享资源登记器。完整登记器快照在`vendor/resource_registry.py`；接入方法见复现说明。公开clone不能自动绕过资源租约或重新派发封存job。168小时为登记上限，本轮已提前完成；`study.json`中的publication=false记录完成当时状态，本PR为其后授权发布，不改写原实验。

大CSV保持全部行、按原顺序分片，INDEX及manifest登记来源哈希；公开报告仅修订链接和标注，不改科学数值。原始读数、输入、完整预测和权重不公开。方法源码、实际证据和本次公开验证必须分别理解。
