# 全域月水平＋高频日变化联合校准：专家诊断

28条登记路径中，28条数值充分，28条有通过原物理容差的结果。数值不足路径保留预测诊断，但其配对结论为条件性。

本轮为算法隔离的回顾性检验。2024为正式留出；2025仅1—11月月报敏感性，水文/PET延伸及重复2024氮源与需求不能与2024合并认证。2025无合格高频站月。未知月报采用等日均值仍属条件假设；同名同坐标不能认证全部站址历史和审核血缘。自然日对应混合日界的冻结水文日期是日尺度近似，未建立四小时物理模型。

## 固定比较的结果

- **S113_D minus S113_M**：2024分组日RMSE下降0组、上升2组；月报组合门槛通过0/2组。不同区域和空间面板分别判断，不能用组数代替站点收益或独立流域样本量。
- **S191_D minus S191_M**：2024分组日RMSE下降1组、上升0组；月报组合门槛通过0/3组。不同区域和空间面板分别判断，不能用组数代替站点收益或独立流域样本量。
- **S56_D minus S56_M**：2024分组日RMSE下降1组、上升0组；月报组合门槛通过0/1组。不同区域和空间面板分别判断，不能用组数代替站点收益或独立流域样本量。
- **T24_G_D minus T24_G_M**：2024分组日RMSE下降2组、上升2组；月报组合门槛通过0/4组。不同区域和空间面板分别判断，不能用组数代替站点收益或独立流域样本量。
- **T24_G_D minus T24_L_D**：2024分组日RMSE下降3组、上升1组；月报组合门槛通过2/4组。不同区域和空间面板分别判断，不能用组数代替站点收益或独立流域样本量。
- **T24_G_M minus T24_L_M**：2024分组日RMSE下降3组、上升1组；月报组合门槛通过2/4组。不同区域和空间面板分别判断，不能用组数代替站点收益或独立流域样本量。
- **T24_L_D minus T24_L_M**：2024分组日RMSE下降4组、上升0组；月报组合门槛通过0/4组。不同区域和空间面板分别判断，不能用组数代替站点收益或独立流域样本量。
- **T25S_G_D minus T25S_G_M**：2025年1—11月月报敏感性组合门槛通过0/4组；无合格日尺度主评价，与2024分开解释。
- **T25S_G_D minus T25S_L_D**：2025年1—11月月报敏感性组合门槛通过1/4组；无合格日尺度主评价，与2024分开解释。
- **T25S_G_M minus T25S_L_M**：2025年1—11月月报敏感性组合门槛通过1/4组；无合格日尺度主评价，与2024分开解释。
- **T25S_L_D minus T25S_L_M**：2025年1—11月月报敏感性组合门槛通过0/4组；无合格日尺度主评价，与2024分开解释。

| contrast | scale | panel | cohort | stations | eligible_NSE | delta_NSE | delta_r | RMSE_ratio | numerically_sufficient |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T24_L_D minus T24_L_M | daily | TIME_2024 | H | 5 | 5 | 0.0518656 | 0.0232357 | 0.963492 | True |
| T24_L_D minus T24_L_M | daily | TIME_2024 | N | 2 | 2 | 0.0438729 | 0.0353297 | 0.984576 | True |
| T24_L_D minus T24_L_M | daily | TIME_2024 | OTHER | 7 | 7 | 0.0239914 | 0.00695591 | 0.979205 | True |
| T24_L_D minus T24_L_M | daily | TIME_2024 | X | 1 | 1 | 0.0602551 | 0.0148243 | 0.956171 | True |
| T24_L_D minus T24_L_M | monthly_HF | TIME_2024 | H | 5 | 5 | 0.0672599 | 0.0165036 | 0.968957 | True |
| T24_L_D minus T24_L_M | monthly_HF | TIME_2024 | N | 2 | 2 | -0.0234614 | 0.0596907 | 0.992366 | True |
| T24_L_D minus T24_L_M | monthly_HF | TIME_2024 | OTHER | 7 | 6 | 0.0274528 | 0.000990162 | 0.985025 | True |
| T24_L_D minus T24_L_M | monthly_HF | TIME_2024 | X | 1 | 1 | 0.163332 | 0.0489484 | 0.879978 | True |
| T24_L_D minus T24_L_M | monthly_PUB | TIME_2024 | H | 17 | 16 | 0.042601 | -0.0103872 | 0.98977 | True |
| T24_L_D minus T24_L_M | monthly_PUB | TIME_2024 | N | 17 | 12 | 0.0715327 | 0.0940553 | 0.996607 | True |
| T24_L_D minus T24_L_M | monthly_PUB | TIME_2024 | OTHER | 75 | 53 | -0.0491678 | 0.0117685 | 0.991796 | True |
| T24_L_D minus T24_L_M | monthly_PUB | TIME_2024 | X | 7 | 4 | 0.144057 | 0.0144872 | 0.932121 | True |
| T24_L_D minus T24_L_M | weekly | TIME_2024 | H | 5 | 5 | 0.0638086 | 0.0240474 | 0.964964 | True |
| T24_L_D minus T24_L_M | weekly | TIME_2024 | N | 2 | 2 | 0.0197977 | 0.0678552 | 0.988562 | True |
| T24_L_D minus T24_L_M | weekly | TIME_2024 | OTHER | 7 | 7 | -0.0533854 | 0.00626652 | 0.978751 | True |
| T24_L_D minus T24_L_M | weekly | TIME_2024 | X | 1 | 1 | 0.0542243 | 0.0158189 | 0.95199 | True |
| T24_G_M minus T24_L_M | daily | TIME_2024 | H | 5 | 5 | 0.698122 | 0.0688497 | 0.876738 | True |
| T24_G_M minus T24_L_M | daily | TIME_2024 | N | 2 | 2 | -0.0293709 | 0.245582 | 0.96485 | True |
| T24_G_M minus T24_L_M | daily | TIME_2024 | OTHER | 7 | 7 | -0.636675 | 0.127732 | 0.221233 | True |
| T24_G_M minus T24_L_M | daily | TIME_2024 | X | 1 | 1 | -0.300816 | 0.0401276 | 1.195 | True |
| T24_G_M minus T24_L_M | monthly_HF | TIME_2024 | H | 5 | 5 | 1.73031 | 0.00849742 | 0.929185 | True |
| T24_G_M minus T24_L_M | monthly_HF | TIME_2024 | N | 2 | 2 | -0.26369 | 0.0984364 | 1.02591 | True |
| T24_G_M minus T24_L_M | monthly_HF | TIME_2024 | OTHER | 7 | 6 | -0.623715 | 0.259312 | 0.20483 | True |
| T24_G_M minus T24_L_M | monthly_HF | TIME_2024 | X | 1 | 1 | -0.285904 | 0.0333641 | 1.18109 | True |
| T24_G_M minus T24_L_M | monthly_PUB | TIME_2024 | H | 17 | 16 | 0.379438 | -0.0240327 | 1.06883 | True |
| T24_G_M minus T24_L_M | monthly_PUB | TIME_2024 | N | 17 | 12 | -1.27336 | 0.201102 | 1.43591 | True |
| T24_G_M minus T24_L_M | monthly_PUB | TIME_2024 | OTHER | 75 | 53 | 2.0838 | 0.113672 | 0.296723 | True |
| T24_G_M minus T24_L_M | monthly_PUB | TIME_2024 | X | 7 | 4 | 1.48906 | 0.106841 | 0.413705 | True |
| T24_G_M minus T24_L_M | weekly | TIME_2024 | H | 5 | 5 | 0.865422 | 0.066731 | 0.924926 | True |
| T24_G_M minus T24_L_M | weekly | TIME_2024 | N | 2 | 2 | -0.205109 | 0.291619 | 0.994261 | True |
| T24_G_M minus T24_L_M | weekly | TIME_2024 | OTHER | 7 | 7 | -0.788307 | 0.148912 | 0.220203 | True |
| T24_G_M minus T24_L_M | weekly | TIME_2024 | X | 1 | 1 | -0.447524 | 0.0132533 | 1.33171 | True |
| T24_G_D minus T24_L_D | daily | TIME_2024 | H | 5 | 5 | 0.656122 | 0.0565663 | 0.906292 | True |
| T24_G_D minus T24_L_D | daily | TIME_2024 | N | 2 | 2 | 0.053433 | 0.253435 | 0.950649 | True |
| T24_G_D minus T24_L_D | daily | TIME_2024 | OTHER | 7 | 7 | -0.633284 | 0.138584 | 0.226248 | True |
| T24_G_D minus T24_L_D | daily | TIME_2024 | X | 1 | 1 | -0.373171 | 0.0257487 | 1.25729 | True |
| T24_G_D minus T24_L_D | monthly_HF | TIME_2024 | H | 5 | 5 | 1.67724 | 0.000409275 | 0.956095 | True |
| T24_G_D minus T24_L_D | monthly_HF | TIME_2024 | N | 2 | 2 | -0.244173 | 0.0798951 | 1.03116 | True |
| T24_G_D minus T24_L_D | monthly_HF | TIME_2024 | OTHER | 7 | 6 | -0.613275 | 0.255952 | 0.208519 | True |
| T24_G_D minus T24_L_D | monthly_HF | TIME_2024 | X | 1 | 1 | -0.464833 | -0.0209408 | 1.3525 | True |
| T24_G_D minus T24_L_D | monthly_PUB | TIME_2024 | H | 17 | 16 | 0.343648 | -0.0076537 | 1.08401 | True |
| T24_G_D minus T24_L_D | monthly_PUB | TIME_2024 | N | 17 | 12 | -1.56806 | 0.112819 | 1.4396 | True |
| T24_G_D minus T24_L_D | monthly_PUB | TIME_2024 | OTHER | 75 | 53 | 2.12257 | 0.0939714 | 0.298704 | True |
| T24_G_D minus T24_L_D | monthly_PUB | TIME_2024 | X | 7 | 4 | 1.34463 | 0.104213 | 0.445671 | True |
| T24_G_D minus T24_L_D | weekly | TIME_2024 | H | 5 | 5 | 0.814708 | 0.0544214 | 0.954681 | True |
| T24_G_D minus T24_L_D | weekly | TIME_2024 | N | 2 | 2 | -0.136309 | 0.279313 | 0.9842 | True |
| T24_G_D minus T24_L_D | weekly | TIME_2024 | OTHER | 7 | 7 | -0.703526 | 0.14437 | 0.225406 | True |
| T24_G_D minus T24_L_D | weekly | TIME_2024 | X | 1 | 1 | -0.514951 | -0.0018549 | 1.40784 | True |
| T24_G_D minus T24_G_M | daily | TIME_2024 | H | 5 | 5 | 0.00986571 | 0.0109523 | 0.99597 | True |
| T24_G_D minus T24_G_M | daily | TIME_2024 | N | 2 | 2 | 0.126677 | 0.0431827 | 0.970085 | True |
| T24_G_D minus T24_G_M | daily | TIME_2024 | OTHER | 7 | 7 | 0.0273828 | 0.0178078 | 1.0014 | True |
| T24_G_D minus T24_G_M | daily | TIME_2024 | X | 1 | 1 | -0.0121001 | 0.000445354 | 1.00601 | True |
| T24_G_D minus T24_G_M | monthly_HF | TIME_2024 | H | 5 | 5 | 0.0141896 | 0.00841546 | 0.997019 | True |
| T24_G_D minus T24_G_M | monthly_HF | TIME_2024 | N | 2 | 2 | -0.00394426 | 0.0411494 | 0.997442 | True |
| T24_G_D minus T24_G_M | monthly_HF | TIME_2024 | OTHER | 7 | 6 | 0.0378934 | -0.0023703 | 1.00277 | True |
| T24_G_D minus T24_G_M | monthly_HF | TIME_2024 | X | 1 | 1 | -0.0155969 | -0.00535645 | 1.00769 | True |
| T24_G_D minus T24_G_M | monthly_PUB | TIME_2024 | H | 17 | 16 | 0.0068111 | 0.00599176 | 1.00382 | True |
| T24_G_D minus T24_G_M | monthly_PUB | TIME_2024 | N | 17 | 12 | -0.223172 | 0.00577277 | 0.999164 | True |
| T24_G_D minus T24_G_M | monthly_PUB | TIME_2024 | OTHER | 75 | 53 | -0.0103954 | -0.00793162 | 0.998415 | True |
| T24_G_D minus T24_G_M | monthly_PUB | TIME_2024 | X | 7 | 4 | -0.000377073 | 0.0118595 | 1.00414 | True |
| T24_G_D minus T24_G_M | weekly | TIME_2024 | H | 5 | 5 | 0.0130944 | 0.0117379 | 0.996008 | True |
| T24_G_D minus T24_G_M | weekly | TIME_2024 | N | 2 | 2 | 0.0885975 | 0.0555495 | 0.978559 | True |
| T24_G_D minus T24_G_M | weekly | TIME_2024 | OTHER | 7 | 7 | 0.0313958 | 0.00172414 | 1.00188 | True |
| T24_G_D minus T24_G_M | weekly | TIME_2024 | X | 1 | 1 | -0.0132031 | 0.000710646 | 1.00641 | True |
| T25S_L_D minus T25S_L_M | monthly_PUB | TIME_2025_SENSITIVITY | H | 17 | 16 | -0.0203911 | 0.00686619 | 0.988789 | True |
| T25S_L_D minus T25S_L_M | monthly_PUB | TIME_2025_SENSITIVITY | N | 17 | 12 | 0.00502544 | 0.0303776 | 1.00506 | True |
| T25S_L_D minus T25S_L_M | monthly_PUB | TIME_2025_SENSITIVITY | OTHER | 75 | 54 | -0.421553 | 0.0248435 | 1.00091 | True |
| T25S_L_D minus T25S_L_M | monthly_PUB | TIME_2025_SENSITIVITY | X | 7 | 5 | 0.00777198 | 0.0297097 | 0.933605 | True |
| T25S_G_M minus T25S_L_M | monthly_PUB | TIME_2025_SENSITIVITY | H | 17 | 16 | 0.00736763 | -0.156084 | 1.14295 | True |
| T25S_G_M minus T25S_L_M | monthly_PUB | TIME_2025_SENSITIVITY | N | 17 | 12 | -0.803782 | 0.0272956 | 1.4735 | True |
| T25S_G_M minus T25S_L_M | monthly_PUB | TIME_2025_SENSITIVITY | OTHER | 75 | 54 | 2.32755 | 0.138302 | 0.287338 | True |
| T25S_G_M minus T25S_L_M | monthly_PUB | TIME_2025_SENSITIVITY | X | 7 | 5 | 0.00352004 | 0.236346 | 0.498982 | True |
| T25S_G_D minus T25S_L_D | monthly_PUB | TIME_2025_SENSITIVITY | H | 17 | 16 | 0.0184834 | -0.182063 | 1.15753 | True |
| T25S_G_D minus T25S_L_D | monthly_PUB | TIME_2025_SENSITIVITY | N | 17 | 12 | -1.03878 | -0.0115571 | 1.46739 | True |
| T25S_G_D minus T25S_L_D | monthly_PUB | TIME_2025_SENSITIVITY | OTHER | 75 | 54 | 2.75434 | 0.103472 | 0.286896 | True |
| T25S_G_D minus T25S_L_D | monthly_PUB | TIME_2025_SENSITIVITY | X | 7 | 5 | -0.00563532 | 0.247882 | 0.533622 | True |
| T25S_G_D minus T25S_G_M | monthly_PUB | TIME_2025_SENSITIVITY | H | 17 | 16 | -0.00927532 | -0.0191128 | 1.00141 | True |
| T25S_G_D minus T25S_G_M | monthly_PUB | TIME_2025_SENSITIVITY | N | 17 | 12 | -0.229972 | -0.00847498 | 1.0009 | True |
| T25S_G_D minus T25S_G_M | monthly_PUB | TIME_2025_SENSITIVITY | OTHER | 75 | 54 | 0.00524123 | -0.00998709 | 0.999374 | True |
| T25S_G_D minus T25S_G_M | monthly_PUB | TIME_2025_SENSITIVITY | X | 7 | 5 | -0.00138338 | 0.0412456 | 0.998418 | True |
| S56_D minus S56_M | daily | SPACE_HELD | N | 2 | 2 | 0.0189354 | 0.0150347 | 0.992342 | True |
| S56_D minus S56_M | daily | SPACE_TRAIN | H | 5 | 5 | 0.0116968 | 0.0138904 | 0.99469 | True |
| S56_D minus S56_M | daily | SPACE_TRAIN | OTHER | 7 | 7 | 0.0204372 | 0.0267894 | 0.997767 | True |
| S56_D minus S56_M | daily | SPACE_TRAIN | X | 1 | 1 | -0.00241815 | 0.00976298 | 1.0012 | True |
| S56_D minus S56_M | monthly_HF | SPACE_HELD | N | 2 | 2 | 0.0260848 | 0.00655989 | 0.99278 | True |
| S56_D minus S56_M | monthly_HF | SPACE_TRAIN | H | 5 | 5 | 0.0146542 | 0.00949848 | 0.996076 | True |
| S56_D minus S56_M | monthly_HF | SPACE_TRAIN | OTHER | 7 | 6 | 0.0303793 | 0.0119607 | 0.998462 | True |
| S56_D minus S56_M | monthly_HF | SPACE_TRAIN | X | 1 | 1 | -0.00392555 | 0.00639079 | 1.00196 | True |
| S56_D minus S56_M | monthly_PUB | SPACE_HELD | N | 12 | 8 | 0.0502985 | 0.018898 | 0.996739 | True |
| S56_D minus S56_M | monthly_PUB | SPACE_TRAIN | H | 17 | 16 | -0.00190358 | 0.00576122 | 1.00092 | True |
| S56_D minus S56_M | monthly_PUB | SPACE_TRAIN | N | 5 | 4 | 0.0299628 | 0.017843 | 0.998071 | True |
| S56_D minus S56_M | monthly_PUB | SPACE_TRAIN | OTHER | 75 | 53 | 0.00593413 | -0.0243823 | 0.99808 | True |
| S56_D minus S56_M | monthly_PUB | SPACE_TRAIN | X | 7 | 4 | -0.0144304 | 0.0278383 | 1.00312 | True |
| S56_D minus S56_M | weekly | SPACE_HELD | N | 2 | 2 | 0.0204977 | 0.0168998 | 0.991853 | True |
| S56_D minus S56_M | weekly | SPACE_TRAIN | H | 5 | 5 | 0.0147879 | 0.0174128 | 0.994854 | True |
| S56_D minus S56_M | weekly | SPACE_TRAIN | OTHER | 7 | 7 | 0.0258921 | 0.0036696 | 0.99762 | True |
| S56_D minus S56_M | weekly | SPACE_TRAIN | X | 1 | 1 | -0.00180949 | 0.0107678 | 1.00088 | True |
| S113_D minus S113_M | daily | SPACE_BUFFER | OTHER | 2 | 2 | -0.0532971 | 0.00362044 | 1.01265 | True |
| S113_D minus S113_M | daily | SPACE_HELD | X | 1 | 1 | -0.00634573 | 0.00855022 | 1.00476 | True |
| S113_D minus S113_M | daily | SPACE_TRAIN | H | 5 | 5 | 0.0117021 | 0.00897935 | 0.994436 | True |
| S113_D minus S113_M | daily | SPACE_TRAIN | N | 2 | 2 | 0.0600345 | 0.0250201 | 0.982503 | True |
| S113_D minus S113_M | daily | SPACE_TRAIN | OTHER | 5 | 5 | 0.00755441 | 0.0250941 | 0.998023 | True |
| S113_D minus S113_M | monthly_HF | SPACE_BUFFER | OTHER | 2 | 2 | -0.1006 | 0.00827852 | 1.01594 | True |
| S113_D minus S113_M | monthly_HF | SPACE_HELD | X | 1 | 1 | -0.00360046 | 0.00785892 | 1.00328 | True |
| S113_D minus S113_M | monthly_HF | SPACE_TRAIN | H | 5 | 5 | 0.0194014 | 0.00798377 | 0.995162 | True |
| S113_D minus S113_M | monthly_HF | SPACE_TRAIN | N | 2 | 2 | 0.0471767 | 0.00793414 | 0.987047 | True |
| S113_D minus S113_M | monthly_HF | SPACE_TRAIN | OTHER | 5 | 4 | 0.00451429 | 0.0149238 | 0.999065 | True |
| S113_D minus S113_M | monthly_PUB | SPACE_BUFFER | OTHER | 11 | 10 | -0.00624658 | 0.0089968 | 0.994721 | True |
| S113_D minus S113_M | monthly_PUB | SPACE_HELD | X | 5 | 3 | 0.00451366 | 0.0296964 | 0.992122 | True |
| S113_D minus S113_M | monthly_PUB | SPACE_TRAIN | H | 17 | 16 | 0.0114755 | -0.0145095 | 0.999815 | True |
| S113_D minus S113_M | monthly_PUB | SPACE_TRAIN | N | 17 | 12 | 0.0142591 | 0.00755253 | 0.997712 | True |
| S113_D minus S113_M | monthly_PUB | SPACE_TRAIN | OTHER | 64 | 43 | 0.00167336 | 0.0139993 | 0.997823 | True |
| S113_D minus S113_M | monthly_PUB | SPACE_TRAIN | X | 2 | 1 | -0.030804 | -0.0681188 | 1.01599 | True |
| S113_D minus S113_M | weekly | SPACE_BUFFER | OTHER | 2 | 2 | -0.0519694 | 0.00365474 | 1.0117 | True |
| S113_D minus S113_M | weekly | SPACE_HELD | X | 1 | 1 | -0.00788961 | 0.0102904 | 1.00651 | True |
| S113_D minus S113_M | weekly | SPACE_TRAIN | H | 5 | 5 | 0.0158804 | 0.00934157 | 0.994691 | True |
| S113_D minus S113_M | weekly | SPACE_TRAIN | N | 2 | 2 | 0.0561944 | 0.0238192 | 0.983834 | True |
| S113_D minus S113_M | weekly | SPACE_TRAIN | OTHER | 5 | 5 | 0.00295971 | 0.0229371 | 0.998712 | True |
| S191_D minus S191_M | daily | SPACE_HELD | H | 4 | 4 | 0.0396444 | 0.00478157 | 0.995308 | True |
| S191_D minus S191_M | daily | SPACE_TRAIN | H | 1 | 1 | -0.107599 | 0.00356862 | 1.00801 | True |
| S191_D minus S191_M | daily | SPACE_TRAIN | N | 2 | 2 | 0.0670381 | 0.0369211 | 0.977595 | True |
| S191_D minus S191_M | daily | SPACE_TRAIN | OTHER | 7 | 7 | 0.00343728 | 0.0292578 | 0.994893 | True |
| S191_D minus S191_M | daily | SPACE_TRAIN | X | 1 | 1 | 0.0143533 | 0.00782636 | 0.992457 | True |
| S191_D minus S191_M | monthly_HF | SPACE_HELD | H | 4 | 4 | 0.118737 | 0.00673853 | 0.994651 | True |
| S191_D minus S191_M | monthly_HF | SPACE_TRAIN | H | 1 | 1 | -0.141114 | 0.0107039 | 1.00721 | True |
| S191_D minus S191_M | monthly_HF | SPACE_TRAIN | N | 2 | 2 | 0.0602655 | 0.0219089 | 0.98378 | True |
| S191_D minus S191_M | monthly_HF | SPACE_TRAIN | OTHER | 7 | 6 | -0.00600508 | 0.0128132 | 0.995047 | True |
| S191_D minus S191_M | monthly_HF | SPACE_TRAIN | X | 1 | 1 | 0.0261385 | 0.0121953 | 0.988162 | True |
| S191_D minus S191_M | monthly_PUB | SPACE_BUFFER | H | 2 | 2 | 0.0534076 | 0.00234687 | 0.996998 | True |
| S191_D minus S191_M | monthly_PUB | SPACE_BUFFER | OTHER | 10 | 8 | 0.0339794 | 0.0167935 | 0.994176 | True |
| S191_D minus S191_M | monthly_PUB | SPACE_HELD | H | 6 | 6 | 0.0213129 | 0.0017414 | 0.996079 | True |
| S191_D minus S191_M | monthly_PUB | SPACE_TRAIN | H | 9 | 8 | 0.000983558 | -0.0282704 | 1.00163 | True |
| S191_D minus S191_M | monthly_PUB | SPACE_TRAIN | N | 17 | 12 | 0.0173541 | -0.0171489 | 0.99869 | True |
| S191_D minus S191_M | monthly_PUB | SPACE_TRAIN | OTHER | 65 | 45 | 0.0157951 | 0.0185923 | 0.997116 | True |
| S191_D minus S191_M | monthly_PUB | SPACE_TRAIN | X | 7 | 4 | 0.00814005 | 0.00178932 | 0.999722 | True |
| S191_D minus S191_M | weekly | SPACE_HELD | H | 4 | 4 | 0.0453515 | 0.00139236 | 0.995013 | True |
| S191_D minus S191_M | weekly | SPACE_TRAIN | H | 1 | 1 | -0.119497 | 0.00772935 | 1.00895 | True |
| S191_D minus S191_M | weekly | SPACE_TRAIN | N | 2 | 2 | 0.0629816 | 0.0385942 | 0.978444 | True |
| S191_D minus S191_M | weekly | SPACE_TRAIN | OTHER | 7 | 7 | 0.00744726 | 0.0295937 | 0.994877 | True |
| S191_D minus S191_M | weekly | SPACE_TRAIN | X | 1 | 1 | 0.0151596 | 0.00677483 | 0.992209 | True |

## 目标与参数补偿

月水平采用唯一站月来源：合格HF优先，否则月报；M/D共同月份与方差完全相同。D仅增加HF月内中心化误差，不重复计月均值。L/G共有站点的数据相同，但1/S随合法站数变化，这是登记的站等权设计。

数据项、先验项和全部30参数见fit_summary.csv；两入口MAP差见start_stability.csv。不同目标的MAP大小不可直接用来宣称预测更优，梯度小也不保证同一全局解。

## 时间收益与空间收益

时间折G包含棉江训练标签，棉江2024结果属于训练地点的时间预测。空间能力仅看S56、S113、S191的SPACE_HELD；SPACE_BUFFER为禁用下游诊断，SPACE_TRAIN不混入空间成绩。三个支流并不足以代表众多独立流域样本。交互量见factor_interaction.parquet。

逐站NSE资格及总覆盖、均值偏差、中心化误差、幅度、月峰位相、高浓度缺口均保留在station_metrics.csv。月内残差分解见monthly_error_components.parquet；严重退化和持续高估见risk_stations.csv，不因退化删除站点。日/周/月尺度分开，不将浓度相加解释为负荷。

## 过程与支持限制

高低流分组使用冻结训练水量阈值，相关记录见flow_diagnostics.parquet。同河段站的共同日期浓度差和比例见same_reach_diagnostics.parquet，不假定同时观测属于同一水团。

每路径daily_station_mass_water.parquet保存日质量与水量；monthly_physical_ledger.parquet保存全域库存、入河、摄取、需求和损失；network_ledger.parquet保存终端与水库库存。四源标签仅继承158、225试点范围，不称全域已识别历史来源。预测改善不证明真实legacy来源或施肥日历已识别。

## 稳定性

CHM/CMFD比较同时报告公共日期与覆盖变化，time_support_direction.csv列示收益反转。出现反转的站点应称时间支持下尚不可辨别，不挑选最好的日界。维护状态及训练删除前资料只作固定参数诊断。整月同步重采样仅描述本资料稳定性，不是未来预测或机制置信区间。

## 逐路径数值状态

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

## 月水平与月内训练损失

| tag | month_level | within_month | prior | objective | pg |
| --- | --- | --- | --- | --- | --- |
| T24_L_M_s0 | 0.79995 | 9.22695e-34 | 0.0530307 | 0.852981 | 2.84578e-07 |
| T24_L_D_s0 | 0.801853 | 0.0475914 | 0.0530824 | 0.902526 | 8.76142e-07 |
| T24_G_M_s1 | 1.13266 | 2.17519e-33 | 0.0527206 | 1.18538 | 8.02849e-07 |
| T24_G_D_s0 | 1.13252 | 0.0215168 | 0.0534507 | 1.20749 | 3.67031e-07 |
| S56_M_s0 | 1.01665 | 1.57696e-33 | 0.0490343 | 1.06569 | 5.53139e-07 |
| S56_D_s0 | 1.016 | 0.0205277 | 0.0498419 | 1.08637 | 1.00338e-06 |
| S113_M_s1 | 1.12922 | 1.79302e-33 | 0.0429611 | 1.17219 | 4.16553e-07 |
| S113_D_s0 | 1.12834 | 0.0198113 | 0.0440978 | 1.19225 | 4.55266e-07 |
| S191_M_s1 | 1.08204 | 2.50172e-33 | 0.103865 | 1.18591 | 5.59004e-07 |
| S191_D_s0 | 1.08271 | 0.0145749 | 0.103406 | 1.20069 | 7.90033e-07 |
| T25S_L_M_s1 | 0.810345 | 1.11525e-33 | 0.0494791 | 0.859824 | 1.84178e-06 |
| T25S_L_D_s0 | 0.809642 | 0.0515513 | 0.0530903 | 0.914283 | 3.14087e-07 |
| T25S_G_M_s0 | 1.13635 | 2.39488e-33 | 0.0515008 | 1.18785 | 7.12695e-07 |
| T25S_G_D_s1 | 1.1359 | 0.0242378 | 0.0525007 | 1.21264 | 4.92666e-07 |

逐站月贡献保存在training_loss_components.parquet。G相较L扩大了站数，也改变具有日资料站点在总目标中的占比；共有站仍采用同一来源与尺度，但整体1/S归一化会变化。因此G-D与L-D的差异应结合覆盖和贡献构成解释，不能全归因为某一新增站点的因果作用。

选定参数的完整历史物理重算见selected_routing_recompute.json；selected_routing目录包含各河段入口/出口、各水库捕获/释放/库存，以及观测年份每日源输入—快慢入河—出口响应链。OU沿用158、225号试点积分范围，其他河段仍使用原表达。

## 棉江旧点的有限外推检查

七站仅覆盖五个河段；棉江有3/7个静态属性超出这五河段的范围。该判断描述环境覆盖，不表示超出原26参考河段的所有标准化范围，也不单独证明退化原因。

| field | mianjiang_raw | training_min | training_max | standardized | outside_seven_station_support |
| --- | --- | --- | --- | --- | --- |
| log_awc_0_200_mm | 5.4619 | 5.34635 | 5.65063 | -0.636296 | False |
| bulk_density_0_30_g_cm3 | 1.07029 | 1.06013 | 1.26587 | -1.57078 | False |
| glhymps_log10_permeability_m2 | -11.8825 | -14.298 | -12.389 | 1.95097 | True |
| glhymps_porosity | 0.0611979 | 0.0755643 | 0.179058 | -1.15016 | True |
| log_dem_slope | -11.1934 | -8.7713 | -5.37792 | -2.24445 | True |
| log_predev_annual_precipitation_mm | 7.13609 | 6.66402 | 7.51162 | -0.304976 | False |
| log_predev_annual_pet_mm | 6.46894 | 6.29445 | 6.78152 | -0.322598 | False |

四个旧保存点均从1961重新传播，只作诊断，没有参与新路径初始化或预热库存。它们在棉江的动员、库存、快慢通量及损失差异如下；端点β本身不能概括所有参数补偿。

| old_tag | beta | hazard_max | M_end | L_end | fast_kg | slow_kg | loss_kg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F23_M_HF_s1 | 0.727619 | 0.00296112 | 4.44892e+06 | 69861.9 | 603588 | 268244 | 3.48508e+06 |
| F23_D_HF_s0 | 0.696522 | 0.00127493 | 3.91524e+06 | 19666.1 | 248936 | 75700.8 | 3.89381e+06 |
| F24_M_HF_s1 | 0.486893 | 0.000986219 | 1.12183e+07 | 61240.8 | 668156 | 228974 | 3.38523e+06 |
| F24_D_HF_s1 | 0.671709 | 0.00104904 | 9.03616e+06 | 34072 | 496306 | 126245 | 3.66179e+06 |

## 主要发现与具体数值

28条路径全部完成独立重算，28/28数值充分、28/28通过原物理及来源一致性要求。S113_M_s0保留预算停止状态，但其独立投影梯度4.58×10⁻⁶已达标；无需因时间再续算。

**扩域有明显但不均衡的收益，尚未解决区域偏差或棉江式外推问题。** 2024年G-D相对L-D，新增75站的站均月RMSE降至0.299倍，X组降至0.446倍，两组通过登记的完整月门槛；N组升至1.440倍，H组升至1.084倍。N组月NSE中位数从−0.560降至−2.128。

**日信息在全域联合目标中的额外收益较小。** G-D相对G-M，15个合格HF站中11站日RMSE下降；N/H组站均日RMSE分别下降约3.0%/0.4%，OTHER和棉江略升。四个原月面板组均未通过日信息增量的完整改善门槛。

**棉江并未因加入全域训练而改善。** L-D与G-D的2024日NSE分别为0.357和−0.016，RMSE从0.348升至0.438 mg/L。S113独立空间留出下，M/D日NSE为0.335/0.328；日约束没有改善这一外层。S56与S191的日RMSE仅分别下降约0.8%和0.5%，三个空间主留出块均未通过完整月门槛，绝对精度仍不足。

**收敛不等于起点稳定。** S191-D、T25S-L-M、T25S-L-D的两入口训练MAP差约11.6%、8.0%、7.4%；均按训练目标选点，其他入口结果保留。150项公共日期日界比较中，D−M的日RMSE方向反转0项，这只支持本轮所测试日界下的方向稳定。

S191-D尤其需要保留条件性结论：另一个训练MAP较差的合法入口，其空间留出日RMSE为M基准的0.617倍，而按训练MAP选中的入口为0.995倍。不能根据已见留出成绩改选，但这表明空间预测明显依赖求解所到达的局部解。

2025仅为1—11月月报敏感性延伸，OTHER继续获益、N/H仍有退化；没有2025日尺度主评价，不能与2024合并认证。全部属于回顾性检验，不自动替换主线。

详见[专家诊断](expert_diagnosis.md)、[逐站指标](station_metrics.csv)、[起点诊断](unstable_start_prediction_diagnostics.csv)和[独立审计](completion_audit.json)。

### 扩域：2024月报同队列

| cohort | stations | eligible_NSE | delta_NSE | delta_r | RMSE_ratio | monthly_gate |
| --- | --- | --- | --- | --- | --- | --- |
| H | 17 | 16 | 0.343648 | -0.0076537 | 1.08401 | False |
| N | 17 | 12 | -1.56806 | 0.112819 | 1.4396 | False |
| OTHER | 75 | 53 | 2.12257 | 0.0939714 | 0.298704 | True |
| X | 7 | 4 | 1.34463 | 0.104213 | 0.445671 | True |

### 全域新增日信息：2024日预测

| cohort | stations | eligible_NSE | delta_NSE | delta_r | RMSE_ratio | improved_RMSE_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| H | 5 | 5 | 0.00986571 | 0.0109523 | 0.99597 | 0.8 |
| N | 2 | 2 | 0.126677 | 0.0431827 | 0.970085 | 1 |
| OTHER | 7 | 7 | 0.0273828 | 0.0178078 | 1.0014 | 0.714286 |
| X | 1 | 1 | -0.0121001 | 0.000445354 | 1.00601 | 0 |

### 月水平与月内动态误差

先按站点平均站月误差、再站等权比较；比值小于1表示D优于M。

| scope | cohort | stations | month_level_ratio | within_month_ratio |
| --- | --- | --- | --- | --- |
| T24_L | H | 5 | 0.936293 | 0.923477 |
| T24_L | N | 2 | 0.987568 | 0.950493 |
| T24_L | OTHER | 7 | 0.968895 | 0.704523 |
| T24_L | X | 1 | 0.774362 | 0.95368 |
| T24_G | H | 5 | 0.995578 | 0.987211 |
| T24_G | N | 2 | 0.996618 | 0.907342 |
| T24_G | OTHER | 7 | 1.00534 | 0.993005 |
| T24_G | X | 1 | 1.01545 | 1.0062 |
| S56 | N | 2 | 0.985747 | 0.979275 |
| S113 | X | 1 | 1.00657 | 1.00129 |
| S191 | H | 4 | 0.98882 | 0.999418 |

### 三个空间主留出块

| contrast | scale | cohort | stations | eligible_NSE | delta_NSE | delta_r | RMSE_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S56_D minus S56_M | daily | N | 2 | 2 | 0.0189354 | 0.0150347 | 0.992342 |
| S56_D minus S56_M | monthly_PUB | N | 12 | 8 | 0.0502985 | 0.018898 | 0.996739 |
| S113_D minus S113_M | daily | X | 1 | 1 | -0.00634573 | 0.00855022 | 1.00476 |
| S113_D minus S113_M | monthly_PUB | X | 5 | 3 | 0.00451366 | 0.0296964 | 0.992122 |
| S191_D minus S191_M | daily | H | 4 | 4 | 0.0396444 | 0.00478157 | 0.995308 |
| S191_D minus S191_M | monthly_PUB | H | 6 | 6 | 0.0213129 | 0.0017414 | 0.996079 |

### 棉江：时间与空间面板分开

| tag | n | NSE | r | RMSE | bias | centered_MSE |
| --- | --- | --- | --- | --- | --- | --- |
| T24_L_M_s0 | 225 | 0.297213 | 0.588381 | 0.364126 | 0.0780446 | 0.126497 |
| T24_L_D_s0 | 225 | 0.357468 | 0.603205 | 0.348167 | 0.0219967 | 0.120736 |
| T24_G_M_s1 | 225 | -0.00360368 | 0.628508 | 0.435132 | -0.239577 | 0.131942 |
| T24_G_D_s0 | 225 | -0.0157038 | 0.628954 | 0.437747 | -0.242562 | 0.132786 |
| S56_M_s0 | 225 | -0.00695847 | 0.619419 | 0.435858 | -0.241387 | 0.131705 |
| S56_D_s0 | 225 | -0.00937662 | 0.629182 | 0.436381 | -0.243565 | 0.131105 |
| S113_M_s1 | 225 | 0.334678 | 0.642126 | 0.354287 | -0.0909576 | 0.117246 |
| S113_D_s0 | 225 | 0.328332 | 0.650676 | 0.355973 | -0.100628 | 0.116591 |
| S191_M_s1 | 225 | 0.0449011 | 0.32886 | 0.424486 | -0.0411735 | 0.178493 |
| S191_D_s0 | 225 | 0.0592544 | 0.336686 | 0.421285 | -0.0457864 | 0.175384 |

### 对求解入口敏感的配置

| config | legal_starts | sufficient_starts | MAP_relative_gap | selected | interpretation |
| --- | --- | --- | --- | --- | --- |
| S191_D | 2 | 2 | 0.116185 | S191_D_s0 | distinct local solutions |
| T25S_L_D | 2 | 2 | 0.0741702 | T25S_L_D_s0 | distinct local solutions |
| T25S_L_M | 2 | 2 | 0.0798361 | T25S_L_M_s1 | distinct local solutions |

| scope | tag | scale | stations | RMSE_ratio | delta_NSE_median |
| --- | --- | --- | --- | --- | --- |
| S191 | S191_D_s0 | daily | 4 | 0.995308 | 0.0396444 |
| S191 | S191_D_s0 | monthly_HF | 4 | 0.994651 | 0.118737 |
| S191 | S191_D_s0 | monthly_PUB | 6 | 0.996079 | 0.0213129 |
| S191 | S191_D_s0 | weekly | 4 | 0.995013 | 0.0453515 |
| S191 | S191_D_s1 | daily | 4 | 0.617245 | 2.3532 |
| S191 | S191_D_s1 | monthly_HF | 4 | 0.518294 | 6.31935 |
| S191 | S191_D_s1 | monthly_PUB | 6 | 0.568462 | 4.22327 |
| S191 | S191_D_s1 | weekly | 4 | 0.588241 | 3.03058 |
| T25S_L | T25S_L_D_s0 | monthly_PUB | 116 | 0.996171 | 0.0634407 |
| T25S_L | T25S_L_D_s1 | monthly_PUB | 116 | 1.03142 | -0.654133 |

这些不同局部解均达小梯度标准；不能靠延长同一局部最优点的时间保证得到另一个解。本轮没有追加入口。

### 同步整月重采样

G-D−G-M的1000次同步整月重采样分位数如下。N/H/OTHER月内中心化误差在这一重采样下倾向下降，但不等于所有区域月水平或日总误差改善；它只是已见月份的描述性稳定性，不是未来预测置信区间。

| cohort | scale | quantity | q025 | median | q975 |
| --- | --- | --- | --- | --- | --- |
| H | monthly_PUB | delta_station_mean_MSE | 0.0100181 | 0.0187035 | 0.0276024 |
| H | monthly_PUB | delta_station_mean_RMSE | 0.00193745 | 0.00528105 | 0.00937818 |
| H | monthly_PUB | delta_station_mean_absolute_bias | -0.000435997 | 0.00827227 | 0.0121539 |
| N | monthly_PUB | delta_station_mean_MSE | -0.00718144 | -0.00306701 | 0.000900043 |
| N | monthly_PUB | delta_station_mean_RMSE | -0.00319879 | -0.000656915 | 0.00198571 |
| N | monthly_PUB | delta_station_mean_absolute_bias | -0.0028424 | 0.000208577 | 0.00262816 |
| OTHER | monthly_PUB | delta_station_mean_MSE | -0.0020426 | -0.000872808 | 0.000257184 |
| OTHER | monthly_PUB | delta_station_mean_RMSE | -0.0014331 | -0.000891786 | -0.000313422 |
| OTHER | monthly_PUB | delta_station_mean_absolute_bias | -0.00122462 | -6.29499e-05 | 0.000971051 |
| X | monthly_PUB | delta_station_mean_MSE | -0.00122242 | 0.00257866 | 0.00676306 |
| X | monthly_PUB | delta_station_mean_RMSE | 0.000306199 | 0.00273562 | 0.00496983 |
| X | monthly_PUB | delta_station_mean_absolute_bias | 0.000457576 | 0.00370801 | 0.00680142 |
| H | HF_month_components | delta_daily_SSE | -0.0476936 | -0.0261931 | -0.00469742 |
| H | HF_month_components | delta_within_SSE | -0.0401292 | -0.0174726 | -0.00275167 |
| N | HF_month_components | delta_daily_SSE | -0.0209679 | -0.00744553 | 0.005122 |
| N | HF_month_components | delta_within_SSE | -0.0167928 | -0.00670443 | -0.0013606 |
| OTHER | HF_month_components | delta_daily_SSE | -0.0035709 | 0.00738268 | 0.01515 |
| OTHER | HF_month_components | delta_within_SSE | -0.00181235 | -0.000918287 | -0.000291158 |
| X | HF_month_components | delta_daily_SSE | 7.96498e-05 | 0.00179247 | 0.00496455 |
| X | HF_month_components | delta_within_SSE | -0.000403861 | 0.000265693 | 0.00148284 |
