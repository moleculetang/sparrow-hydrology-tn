# D29主线与近期结构对照：专家入口

本包对应已经完成的实验；没有新增拟合、修改公式或重新校准水文。核心问题是：保持守恒和H1水文条件后，增加的水质结构是否提供真实浓度预测收益？

## 阅读顺序

1. [模型方程、30参数、先验与站界](docs/MODEL.md)。
2. [输入、水文谱系、清洗与训练方法](docs/DATA_TRAINING.md)。
3. [各轮干预、结果和证据边界](docs/EXPERIMENTS.md)。
4. [实际调用链和运行入口](docs/CODE_MAP.md)。
5. [历史报告更正及限制](docs/CORRECTIONS.md)、[复算方式](docs/REPRODUCTION.md)。

## 当前结论

20_6数值与物理验收通过，但没有跨时期预测收益。状态调制使2023日RMSE降低约2.38%，2024升高约1.41%；两折事件振幅误差增加。总目标下降约73%和88%来自净先验下降。新增方向约95%可由原30参数的局部敏感度列空间解释，但并非完全重复。

2024原D29日RMSE为1.10837 mg/L，15站NSE中位数−0.06717；116站月报RMSE为0.74584 mg/L，85个合格站NSE中位数−0.46813。数值收敛不等于预测可用，也不证明全局最优。

## 无原始数据快速验证

在仓库根目录、已有conda sparrow环境执行：

```powershell
conda run --no-capture-output -n sparrow python -B experiments/d29_review_20260921/tests/test_public_kernels.py
conda run --no-capture-output -n sparrow python -B experiments/d29_review_20260921/tests/test_candidates.py
conda run --no-capture-output -n sparrow python -B experiments/d29_review_20260921/tests/test_evaluation_public.py
```

测试使用人工输入和归档实际函数，不读取原始监测。完整训练仍依赖私有数据及原Windows运行环境。实测软件版本见[environment.json](evidence/environment.json)。

`snapshot`按轮次保留代码原字节；`dependencies`保存跨轮依赖，并不表示重新开展那些实验。`evidence`保存归档证据；本次公开测试结果另存`validation`。每个归档文件的来源与SHA256见[source_manifest.json](evidence/source_manifest.json)。旧报告中的本地链接可能指向未公开产物，应按代码导航阅读，不能当作公开数据链接。
