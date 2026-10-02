# 复现条件与原部署接口

无需私有数据可运行tests/run_public.py和tests/verify_publication.py；真实训练/独立效果复算则需要完整授权实验输入。源码没有因此修改。

`mltn/common.py`从源码位置计算ROOT。ROOT/data至少需要feature_identity.json、reach_features.npy（或identity中的array_file）、adjacency.npy、station_water.npy、station_registry.parquet、spatial_blocks.json、monthly_tn_accepted.parquet、hf_daily_accepted.parquet；读数独立复算另需hf_readings.parquet。身份、年月、单位和采用年份由prepare.py写入，数据未包括在此PR。

prepare.py原工程相对依赖：同级20260928_1中的冻结legacy22数组登记和拓扑、20260917_5/data/domains/FULL24C的原H1，以及已验收农业、LUH3、MOD17、沉降等产品。报告说明年度产品的回顾性可见性。不得把路径存在等同数据身份合格。

检查点推理需要各job的可信checkpoint.pkl、模型booster/估计器或weights.pt；resume.pt为完整优化器/RNG状态。不能加载不可信pickle。公开原源码不附检查点或逐日观测/预测。

机器侧资源接口：Linux在ROOT.parent/.compute_coordination/resource_registry.py，Windows在ROOT.parent/compute_coordination/resource_registry.py，使用共享机器状态。vendor/resource_registry.py是本轮已审计源码供系统管理员接入；不附运行状态、租约、私钥或机器凭据，不在测试中修改实际共享登记器。仅阅读代码不触发计算。

原controller、本机/工作站迁移器和修复器保存当时路径和所有权约束。新授权实验必须更换实验ID并冻结清单、资源门与输入身份；不能在public clone中直接把原已完成ID作为新任务。小批量优先本机，大批量使用实测吞吐与共享资源门决定工作站；不建立定时任务、不自动合并。

公开真实结果是汇总审计证据，无法在没有原数据和模型权重时重新算出。原实际复核回执与本次合成测试分开保存；大表分片可按manifest顺序用pandas.read_csv逐片读取后concat（不重复表头）。
