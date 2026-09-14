# 实际方法与偏离报告

本文件与专家科学报告分开，记录实际运行和数值工程。

复用旧21参数灰箱、独立河网/水库代码和SciPy TRF/可序列化L-BFGS-B；新写库存依赖离散伴随，完整27参数差分和独立前向在训练前通过。预处理为每折preset0参考库存和训练专属损失/环境/水文尺度，绝未从T/Y算子直接续用。

训练前独立代码审查发现并修复旧200次精修上限、不完整累计预算、进程退出不即时审计及新控制路径的恢复风险。详见reports/numerical_implementation_review.md（如存在）及scripts版本哈希。两槽checkpoint和单commit manifest保护恢复；TRF残差调用先计费，缓存回放不重复计费，解析与末态检查仍计入总账。TRF初期最多3000次，保留同一路径预算给精修；这只是数值阶段分配，不是新科学起点。

每worker一线程，CPU float64，conda sparrow。控制器根据实测冷启动/完整峰值乘1.2并预约活跃增长；90%停止派发和RAM退让，85%恢复，无两worker固定上限。每个子任务退出排入独立审计，不等总控退出。正常运行不输出对话进度。

| tag | MAP | data | prior | projected_gradient | numerical_sufficient | calls | hours | reason | repair_used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F23_M0_0 | 1.99265 | 1.61396 | 0.378694 | 5.54016e-06 | True | 1295 | 0.666915 | ORIGINAL_COORDINATE_SUFFICIENT | False |
| F23_M0_1 | 1.99265 | 1.61396 | 0.378695 | 5.83059e-06 | True | 1171 | 0.577985 | ORIGINAL_COORDINATE_SUFFICIENT | False |
| F23_SC_0 | 1.99009 | 1.61209 | 0.378002 | 8.50077e-06 | True | 1052 | 0.901126 | ORIGINAL_COORDINATE_SUFFICIENT | False |
| F23_SC_1 | 1.99009 | 1.61209 | 0.378002 | 3.68901e-06 | True | 1550 | 1.42422 | ORIGINAL_COORDINATE_SUFFICIENT | False |
| F24_M0_0 | 1.89278 | 1.53543 | 0.35735 | 5.09052e-06 | True | 1302 | 0.86432 | ORIGINAL_COORDINATE_SUFFICIENT | False |
| F24_M0_1 | 1.89278 | 1.53543 | 0.357351 | 7.965e-06 | True | 1222 | 0.638315 | ORIGINAL_COORDINATE_SUFFICIENT | False |
| F24_SC_0 | 1.88903 | 1.53171 | 0.357327 | 5.3577e-06 | True | 1055 | 0.390019 | ORIGINAL_COORDINATE_SUFFICIENT | False |
| F24_SC_1 | 1.88903 | 1.53171 | 0.357325 | 6.47808e-06 | True | 1505 | 1.37227 | ORIGINAL_COORDINATE_SUFFICIENT | False |

运行事件完整见outputs/controller_events.jsonl；延长阶段只看训练轨迹，不改先验/方程。每条路径的reason、extensions、repair由reports/fits原样保存。不得因原求解器终态字符串美化收敛。

实际缺项：[]。控制器终态原因：FINITE_MATRIX_TERMINATED。科学阴性单独报告。附加事实：本轮不恢复旧closest，不实施自由E软闭合训练，不进入空间实验。

训练启动后仅修补独立审计与报告完整性，训练方程/参数/先验/数值求解代码及注册身份保持原哈希；修改记录见post_launch_audit_revision.json。修补包括冻结预测逐ID绑定、完整评价分母与mask核验、从两起点审计重建MAP选优、恢复幂等性、容量数值充分条件以及专家报告的量化状态表。没有追加训练调用或科学起点。

F24_M0_0在TRF第1202次预扣调用写检查点时发生Windows文件替换拒绝访问。该次尚未执行模型，但仍保留attempt收费。原提交检查点与1201条TRF缓存独立核验一致，最低训练MAP为1.8927782200429493；这不是模型收敛失败。恢复保持原科学起点、原方程先验和累计预算，使用另存不可变版本的I/O适配器，补记进程尾部时间，保留原失败审计与slot。受影响的F24_SC_0待两个M0结果重新完成审计后才启动；期间禁止报告读取留出标签。实际恢复过程见reports/io_recovery*与outputs/recovery_events.jsonl。

## 终态对账与实际偏离补充

控制器从首次启动至终态共2.392小时；8条路径累计执行（含独立审计）6.835 worker小时，累计计费尝试10152次，不等同于均已完成的模型调用。全部路径在首轮4000次/4小时预算内达到原坐标投影梯度门槛，未启用追加两段或参数缩放修复。

F24_M0_0的I/O恢复及其已注册依赖F24_SC_0均成功完成。13组适配器测试没有TN调用；原1201次TRF缓存只读复现，保留第1202次未实际运行的已计费尝试，并补计0.798199秒进程尾部及原失败审计时间。没有把文件锁原因武断归于杀毒程序。原worker、solver和全部科学代码哈希保持冻结。恢复未新增科学起点，F24_SC_0仍按两条M0训练MAP最优解加零系数初始化。

恢复进程的审计计费另存work/audit_checkpoints，未修改不可变恢复代文件。资源让出后的适配器自动续算未实现，本次没有发生该分支；不能把未触发分支宣称为实测验证。历史失败与阻塞记录保留在io_recovery_original及controller_state.historical_failures，现态单独与恢复审计对账。

用户取消定时巡检后，automation tn-20260911-1保持PAUSED；实际采用目标模式下原生进程句柄静默等待，恢复控制器及原控制器退出事件均已捕获。没有外部聊天发送，未重新启动健康进程。全部拟合与子任务审计结束后才解除report_hold，统一冻结预测并评价。最终统计补充只使用已有输出，没有新TN拟合、QP或完整物理重演。

独立恢复终审另发现：run函数计时不含全部解释器导入与末尾写盘。将原失败、恢复训练与审计的全部原生进程寿命相加后，F24_M0_0累计3117.439309秒（比原账多5.886154秒），F24_SC_0累计1406.373445秒（多2.304089秒），均远小于4小时。本记录作为附加计时账，不追改统一预测冻结所绑定的原审计哈希。详见io_recovery_final_review.json；不能宣称原run计时已覆盖每一秒系统开销。
