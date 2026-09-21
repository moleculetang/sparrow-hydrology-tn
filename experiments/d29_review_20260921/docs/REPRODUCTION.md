# 可执行范围与复现

## 公开测试

README三条命令运行实际归档数值函数上的合成输入。测试不读取私有资料，不运行优化器，不生成科学配置。测试结果写在发布目录validation，和evidence归档验收分开。

## 私有数据持有者的固定参数复算

```powershell
conda run --no-capture-output -n sparrow python -B experiments/d29_review_20260921/tools/replay_private.py --data-root E:/SPARROW --tag F24_X_s0 --check-only
conda run --no-capture-output -n sparrow python -B experiments/d29_review_20260921/tools/replay_private.py --data-root E:/SPARROW --tag F24_X_s0 --output-dir D:/review-output
```

data-root是含5_Test归档的根目录。接口支持20_6全部8个保存点；先列出所需文件并校验身份，缺项立即退出。复算完整1961—2024前向、训练目标、梯度及物理账本，不重拟合。输出只写显式output-dir；禁止指向私有原实验目录。F23目标仅含2021—2022，完整历史尾部用于因果一致性。

适配器使用公开快照，将模块RUN重绑定到显式归档，只替换load_data中H1数组目录这个路径常量；公式、观测、权重及先验不改。训练标签屏障在读取训练文件/哈希前安装；不借此执行原controller或后台任务。适配器自己的结果和代码与原样快照分开。

完整重新训练17_5/18_1/20_6还需要原v3/auto_4h、登记支持、全部派生数组、FCT8月基函数及原Windows conda环境。公开配置是归档证据，不是缺资料时可直接启动的任务。原控制器尚未做Linux移植；公开可见性不意味着数据或代码获得新的许可。

## 公开与省略

上传源码、来源清单、参数终值、标准化/先验常数、配置、汇总统计和审计。省略原始逐条TN、水文数组、栅格、完整逐日预测、状态检查点及缓存。资料哈希和shape可以检查身份，不能替代资料数值。

source_manifest核验归档文件字节；delivery_manifest核验本次发布全部新增文件。PR推送后以Git tree SHA和提交SHA核验远端，不把本地测试成功当作上传成功。

本次另在本地私有数据上实际验证F24_X_s0与F23_X_s1固定参数复算：目标差均为0、物理检查通过。结果位于validation/private_replay_*.json；仅公开汇总，不上传使用的原始数据。这是此次发布适配器验证，与此前归档审计分开。
