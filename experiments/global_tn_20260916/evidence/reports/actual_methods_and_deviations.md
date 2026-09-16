# 实际方法与偏离

预处理第一次重算在无TN标准化浮点归约的精确字典比较处停止；改为1e-12相对/绝对容差核验后仍保留原冻结标准化值。启动前隔离测试发现Windows反斜线导致清单分区筛选不生效；读取屏障已拒绝外折文件，尚未启动拟合。将清单路径统一为POSIX表示后真实身份测试通过。上述修复没有修改目标、科学起点、数据阈值或重置预算。

完整验收同时持有多个模型，原预约使用了该进程6.56 GiB峰值。随后对最大FULL25日目标及116站完整审计单独测量，单worker峰值为约3.17 GiB；按它与实测拟合峰值的最大值×1.2重新预约。只重启控制器并接管原PID，未停止拟合worker、变更每worker身份或重置调用/时钟。RESOURCE_PROFILE_FIXED_POINT是零次优化的原预设1资源测试，不是第29条拟合路径。重启后的收据曾因重复time字段写入失败，控制器已成功接管；独立核验随后补记了收据。证据见resource_reservation_change/verified.json及追加事件日志。

## 实际准入

| scope | stations | hf_stations | months | hf_months | hf_days | floor | reference_stations | spatial_block |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T24_L | 34 | 7 | 1140 | 122 | 2752 | 0.107494 | 34 | — |
| T24_G | 116 | 15 | 3674 | 269 | 6194 | 0.107494 | 34 | — |
| S56 | 104 | 13 | 3285 | 235 | 5432 | 0.0997733 | 22 | 56 |
| S113 | 100 | 12 | 3142 | 212 | 4821 | 0.107494 | 34 | 113 |
| S191 | 98 | 11 | 3059 | 194 | 4503 | 0.100944 | 26 | 191 |
| T25S_L | 34 | 7 | 1501 | 191 | 4063 | 0.100611 | 34 | — |
| T25S_G | 116 | 15 | 4819 | 419 | 9079 | 0.100611 | 34 | — |

## 执行与预算

总预算72小时，最后4小时收尾。时钟起点1789489730.0741248；本报告时刻1789536829.2187672。完整conda sparrow、CPU float64、每worker一线程；按实测峰值×1.2与增长空间派发，没有固定worker上限或定时任务。优先2024时间与空间任务，之后2025敏感性。资源与每次派发、退让、退出事件追加于work/resource_events.jsonl。

| tag | status | calls | objective | pg | numerical_sufficient | physical_reasonable |
| --- | --- | --- | --- | --- | --- | --- |
| T24_L_M_s0 | NUMERICALLY_SUFFICIENT | 2658 | 0.852981 | 2.84578e-07 | True | True |
| T24_L_M_s1 | NUMERICALLY_SUFFICIENT | 2516 | 0.852981 | 8.37095e-07 | True | True |
| T24_L_D_s0 | NUMERICALLY_SUFFICIENT | 1877 | 0.902526 | 8.76142e-07 | True | True |
| T24_L_D_s1 | NUMERICALLY_SUFFICIENT | 2447 | 0.902526 | 9.29658e-07 | True | True |
| T24_G_M_s0 | NUMERICALLY_SUFFICIENT | 2967 | 1.18538 | 1.92355e-06 | True | True |
| T24_G_M_s1 | NUMERICALLY_SUFFICIENT | 3063 | 1.18538 | 8.02849e-07 | True | True |
| T24_G_D_s0 | NUMERICALLY_SUFFICIENT | 2210 | 1.20749 | 3.67031e-07 | True | True |
| T24_G_D_s1 | NUMERICALLY_SUFFICIENT | 3035 | 1.20749 | 6.94573e-07 | True | True |
| S56_M_s0 | NUMERICALLY_SUFFICIENT | 3042 | 1.06569 | 5.53139e-07 | True | True |
| S56_M_s1 | NUMERICALLY_SUFFICIENT | 2855 | 1.06569 | 1.63717e-06 | True | True |
| S56_D_s0 | NUMERICALLY_SUFFICIENT | 2184 | 1.08637 | 1.00338e-06 | True | True |
| S56_D_s1 | NUMERICALLY_SUFFICIENT | 3016 | 1.08637 | 6.84434e-07 | True | True |
| S113_M_s0 | BUDGET_STOPPED | 3195 | 1.17219 | 4.57727e-06 | True | True |
| S113_M_s1 | NUMERICALLY_SUFFICIENT | 3356 | 1.17219 | 4.16553e-07 | True | True |
| S113_D_s0 | NUMERICALLY_SUFFICIENT | 2474 | 1.19225 | 4.55266e-07 | True | True |
| S113_D_s1 | NUMERICALLY_SUFFICIENT | 3324 | 1.19225 | 8.64646e-07 | True | True |
| S191_M_s0 | NUMERICALLY_SUFFICIENT | 2719 | 1.18591 | 6.24558e-07 | True | True |
| S191_M_s1 | NUMERICALLY_SUFFICIENT | 3752 | 1.18591 | 5.59004e-07 | True | True |
| S191_D_s0 | NUMERICALLY_SUFFICIENT | 1508 | 1.20069 | 7.90033e-07 | True | True |
| S191_D_s1 | NUMERICALLY_SUFFICIENT | 3320 | 1.34019 | 2.86985e-07 | True | True |
| T25S_L_M_s0 | NUMERICALLY_SUFFICIENT | 1142 | 0.928469 | 4.82682e-07 | True | True |
| T25S_L_M_s1 | NUMERICALLY_SUFFICIENT | 3313 | 0.859824 | 1.84178e-06 | True | True |
| T25S_L_D_s0 | NUMERICALLY_SUFFICIENT | 2354 | 0.914283 | 3.14087e-07 | True | True |
| T25S_L_D_s1 | NUMERICALLY_SUFFICIENT | 1087 | 0.982096 | 1.41021e-07 | True | True |
| T25S_G_M_s0 | NUMERICALLY_SUFFICIENT | 2991 | 1.18785 | 7.12695e-07 | True | True |
| T25S_G_M_s1 | NUMERICALLY_SUFFICIENT | 2678 | 1.18785 | 1.2204e-06 | True | True |
| T25S_G_D_s0 | NUMERICALLY_SUFFICIENT | 1891 | 1.21264 | 1.03834e-06 | True | True |
| T25S_G_D_s1 | NUMERICALLY_SUFFICIENT | 2893 | 1.21264 | 4.92666e-07 | True | True |

## 证据等级与缺项

本轮为算法隔离的回顾性检验。2024为正式留出；2025仅1—11月月报敏感性，水文/PET延伸及重复2024氮源与需求不能与2024合并认证。2025无合格高频站月。未知月报采用等日均值仍属条件假设；同名同坐标不能认证全部站址历史和审核血缘。自然日对应混合日界的冻结水文日期是日尺度近似，未建立四小时物理模型。

来源标签覆盖仅继承试点，不扩展动力学。新域来自v3正式产品，230河段全部检查字段差异为零；与原68河段完整支持的公共预测/目标/梯度回归通过。原始与旧实验只读。所有条件性未完成路径按path_status.csv保留，不把检查点当作完成。

## 报告渲染与后续时间授权

首轮报告渲染因sparrow环境缺少tabulate而停止；改为实验目录内的Markdown表格函数，不安装依赖、不改预测或选点。随后重跑报告与独立审计。用户追加允许时间不足且未收敛的原路径续算；本轮28条独立审计全部数值充分，未触发额外优化。S113_M_s0虽保留预算停止状态，其保存点梯度已满足原标准。

图件复核后的重复生成曾在重新打开既有PNG时发生Invalid argument；没有修改文件权限，改为仅复用与既有清单哈希一致的图件，未匹配清单的图件仍重新生成。冻结预测身份保持不变。各后处理子任务现在使用单独追加日志，避免多进程共享重定向日志造成诊断丢失。

## 检查点写入中断与恢复

T24_G_M_s1在Windows原子替换latest.json时持续30秒拒绝访问。异常退出前保存了1393次累计调用及11212.44秒活动时间；同内容原子替换与哈希恢复测试随后通过。外部句柄的具体来源未证实。控制器短暂停止派发并接管其他原进程，该路径从原状态恢复，未增加起点、变更核心代码或重置预算。中断点的审计若已启动则另存为恢复证据，不用它完成主路径。详见checkpoint_io_recovery。

T24_G_D_s1随后在替换status.json时发生同类访问异常，保留1555次调用及13721.12秒活动时间后恢复。证据见checkpoint_status_io_recovery；两次中断均不算新增科学路径，重复发生的Windows文件访问风险未被证明彻底消除。

T25S_G_D_s1在精修阶段替换latest.json时也发生同类异常，保留2713次调用及5296.06秒活动时间后从原状态恢复。证据见t25_checkpoint_io_recovery。这是第三次文件访问中断，不增加逻辑拟合路径。

## 计算计数的已知偏离

拟合计数包含优化、差分与恢复检查；独立审计与报告重算未纳入这些计数。computation_accounting.csv/json另列每条路径已知审计接口调用、选定点重算及观察到的审计时长。总调用仅能给出下界，不能声称所有计算均已满足每路径调用或活动时间上限。实验总时钟没有重置。

## 地理摄取支持与风险标记

N/H/X沿用原物理河段集合；OTHER为新增月报站完整上游支持的并集，允许与原区域重叠，各区域摄取比例不得相加。ALL覆盖230河段，空间面板使用各外层登记的支持掩码。详见evaluation_physical_region_registration.json。风险表中的persistent_overestimate仅指正偏差超过0.5倍RMSE的高估候选，并不证明每月持续高估。
