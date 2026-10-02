"""Add correction provenance and finish the authored waveform interpretation."""
import difflib
from mltn.common import ROOT,read,write,sha
from author_amplitude_report import main as author
def main():
    assert read(ROOT/'outputs/seed_repair_postprocess_done.json')['passed']
    assert read(ROOT/'evidence/joint_seed_forwarding_acceptance.json')['passed']
    author()
    old=ROOT/'superseded/seed_not_forwarded/joint.py';new=ROOT/'joint.py'
    diff=''.join(difflib.unified_diff(old.read_text(encoding='utf-8').splitlines(keepends=True),new.read_text(encoding='utf-8').splitlines(keepends=True),fromfile='before/joint.py',tofile='after/joint.py'))
    (ROOT/'evidence/joint_seed_correction.diff').write_text(diff,encoding='utf-8')
    write(ROOT/'evidence/joint_seed_source_change.json',dict(passed=True,before_sha256=sha(old),after_sha256=sha(new),diff_sha256=sha(ROOT/'evidence/joint_seed_correction.diff'),scope='seed forwarded to XGBoost and LightGBM only; neural execution branch unchanged',affected_retraining='original joint XGBoost matrix; no new input, support, target, configurations or seed'))
    actual=read(ROOT/'evidence/independent_actual_effect_metrics.json');roles=read(ROOT/'outputs/result_roles.json');diag=read(ROOT/'outputs/amplitude_diagnostic/receipt.json');repair=read(ROOT/'evidence/joint_seed_repair.json')
    method=ROOT/'reports/实际方法与偏离.md'
    text=method.read_text(encoding='utf-8')
    marker='\n## 振幅专项与联合树种子修正后的权威版本\n'
    if marker in text:text=text.split(marker)[0]
    text+=marker+f'''
46. 用户要求进一步辨认振幅控制项。本轮新增`diagnose_amplitude.py`，以逐站同支持的独立标量矩分解NSE、最优非负幅度增益和强行单位幅度；合成正确相位／错相正负控通过。{diag['station_rows']}条逐站诊断、{diag['registered_readouts']}个冻结读出、18个完整联合训练检查点，诊断本身新拟合0、选点0。标签辅助增益仅是固定波形的解释，不写回正式预测。保持官方月均的零空间方向与实际共享输出头方向分开；最后两个相邻差分步长分别通过原1e−6尺度。初次诊断的FP32月均核失败只发生于新诊断器；改为FP64再求均值，原失败日志保留，不放宽门。

47. 实际发现`joint.py`仅向NumPy/PyTorch传seed，未向联合XGBoost底层传seed；原1729/1730/1731预测相同，不能当三次随机稳定性。源码仅修正树seed传递，旧源码、33逻辑身份及统计快照保留于`superseded/seed_not_forwarded`；神经执行分支和其他直接路线不变。修正后仍只在原8配置、原2022年1—9月支持选择，联合XGBoost由c7改为c{repair['selected']}。受影响原矩阵共24次拟合、9个S24合法父点读出，33个逻辑替换身份，不增加科学网格。实际不同种子预测及同种子重复正控通过[验收](../evidence/joint_seed_forwarding_acceptance.json)。

48. 首轮修复筛选尝试工作站4进程×4线程，遇提交峰值预留门；仅停止未获租约的等待进程，父控制器相对脚本路径最初未捕获，核对PID／cwd后结束。边界已获租约的c5—c7未中断，自然完成后与c0共4条结果一并回收核验。待准入失败与移交证据保留于`superseded/seed_repair_remote_wait`。剩余20次拟合及9个读出由本机2进程×2线程完成，共享登记器和90/85门不变；另一聊天灰箱训练不中断。此处不能声称所有修复拟合都在工作站，亦不能把等待进程当正在训练。

49. `rebuild_seed_evidence.py`重建受影响230reach输出及HF月读出，完整重算所有正式指标、事件、1000次整月／两月块配对抽样；不沿用旧XGBoost预测哈希缓存。未改变输入的精确碰撞检查和未改变直接CatBoost父点的组置换诊断按哈希复用，单列[复用范围](../evidence/seed_repair_unaffected_diagnostics_reuse.json)。最终实际指标{actual['station_rows']}条站结果、{actual['scalar_checks']}个独立标量检查通过；189正式路径缺项{len(roles['missing'])}。两端修复后14项隔离门均通过。原回收包与旧12阶段回执作为修复前来源证据保留，修复后以[重算回执](../outputs/seed_repair_postprocess_done.json)、[专项报告](振幅控制因子数值诊断.md)和最终归档清单为权威。

50. 最终解释：固定当前波形时，正增益最优幅度是max(0,r)；月水平目标不直接惩罚严格零月均方向，但可能通过实际共享参数压制HF想要的增益。灰箱的库存过滤与纯ML平方误差／迁移平滑分别处理，不指定缺乏独立证据的唯一物理主因。没有评价期调幅、重新调权或重开灰箱训练。详细训练／留出、逐站、两种配对、事件、软件缺陷及尚未识别项写入[主专家报告](专家机器学习与全域验证报告.md)和专项报告。

51. 最终后处理另外修复两个工程缺陷：报告绘图仍硬编码c7，内部合法选择改c3后触发查表失败，改为冻结选择动态读取；该失败与恢复均保留，不修改数值预测。后处理父控制器持单核租约，诊断子进程继承相同亲和性后再申请自己的租约，形成PHYSICAL_CORES_RESERVED等待；核对父子PID、创建身份及命令后只释放闲置父租约、恢复可用亲和性，让子进程独立准入，无训练中断。运行器已在具有自主租约的尾部脚本前显式交接，避免嵌套租约死锁。[亲和性移交回执](../evidence/nested_postprocess_lease_handoff.json)、[报告绘图失败及修复](../evidence/report_render_c7_hardcoding_failure.json)记录范围。这两项影响完成流程，不解释水质振幅偏小。
'''
    method.write_text(text,encoding='utf-8')
    # This original append receipt describes the old import, not current metrics.
    receipt=ROOT/'evidence/historical_comparator_append.json'
    if receipt.exists():
        saved=ROOT/'superseded/seed_not_forwarded/historical_comparator_append.json'
        if not saved.exists():saved.write_bytes(receipt.read_bytes())
        write(receipt,dict(passed=True,scope='all nine unchanged historical references included in full seed-repair evaluation recomputation',old_append_receipt=saved.relative_to(ROOT).as_posix(),current_evaluation_sha256=sha(ROOT/'outputs/evaluation/evaluation_receipt.json'),no_original_XGBoost_cache_reused=True))
    write(ROOT/'evidence/amplitude_final_interpretation.json',dict(passed=True,main_report_sha256=sha(ROOT/'reports/专家机器学习与全域验证报告.md'),amplitude_report_sha256=sha(ROOT/'reports/振幅控制因子数值诊断.md'),method_sha256=sha(method),input_physical_cause_not_uniquely_identified=True,software_correction_and_numerical_diagnosis_complete=True))
    print('INTERPRETATION_SEALED',flush=True)
if __name__=='__main__':main()
