# 39河段完整计算补包与CatBoost重放补包

本目录补充原小型审阅包未交付的计算资产；不重复上传旧包，不启动训练。两份原始ZIP通过2 MiB分片保存，Git克隆或下载PR分支后，在本目录执行：

```sh
python -B assemble.py --extract
```

也可使用 `--package case39` 或 `--package catboost`。仅使用Python标准库，逐片、整包SHA256与ZIP CRC都核验；拒绝不安全路径和冲突覆盖。需要约1.5 GB可用空间。恢复目录为 `restored/`。

## 39河段计算包

`case39_compute_1961_2018.zip`：H1统一230河段中的39河段闭合子图、4座水库、1961—2018完整历史。含水文/水库日过程、农业输入与预算、物理静态属性、来源与初态研究情景实际数值，以及仅登记交付资产的 `catalog/unified_input_index.json`。17站映射和417条月度记录身份不含TN标签值。少量230河段年度供体参考用于复算，不是全域日数组。

C1/C2在1961—2015明确使用C0；2016—2018才使用自身时序。来源倍率固定1；初态是研究情景，不是实测或旧拟合检查点。现代NPP历史代理的限制见包内说明。

**沉降只交付原月质量；日沉降生产入口仍全局阻塞。补文件不解除此前零雨月湿沉降冲突的限制，因此本包不构成完整日过程灰箱训练准入。**

```sh
cd restored/case39_compute_1961_2018
python -B scripts/verify_package.py
```

## CatBoost重放包

`catboost_replay_F23_F24_seed1729.zip`：日/月×F23/F24共4模型，seed1729；CBM、预处理、特征数组、记录及划分索引、无TN标签的历史预测和对应源码。F23的2023记录是历史测试，F24的2023记录已参与训练；不能将二者当独立预测能力比较。仅移除CBM内部训练损失等元数据，树与预测不变，保留原件/审阅件双哈希。未重新训练或调参，不含原始TN或成绩表。

```sh
cd restored/catboost_replay_F23_F24_seed1729
python -B scripts/validate_package.py
python -B scripts/catboost_replay.py
```

32,510条历史预测在新解压目录全部精确重放（最大绝对误差0）。包内环境与固定版本是重放依据。上述脚本不使用另一包的绝对路径。

## 完整性与范围

`delivery_manifest.json`登记分片与整包字节数和SHA256；附原独立审查及验证回执。组装只恢复已验收的原ZIP，不修改科学内容。NSE不适用于传输及无标签重放验收。原始TN标签、凭据、SSH密钥与环境缓存均不在本次新增目录。公开上传由用户明确授权；PR不自动合并。
