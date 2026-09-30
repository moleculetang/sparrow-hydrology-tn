"""Evidence-based source experiment report and separate numerical/table recomputation."""
import time,json,math,ast
import numpy as np,pandas as pd
import native_runtime as rt
R=rt.RUN
NAMES={'R':'原D29','U':'统一来源校正D29','G':'分组来源校正D29','S':'分源校正D29'}
def md(df):
    def fmt(x):return f'{x:.6g}' if isinstance(x,(float,np.floating)) and np.isfinite(x) else str(x)
    return '| '+' | '.join(df.columns)+' |\n|'+'|'.join(['---']*len(df.columns))+'|\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))
def write(name,s):(R/name).write_text(s,encoding='utf-8')
def main():
    selected=rt.read(R/'data/selected.json');paths=pd.read_csv(R/'reports/path_summary.csv');rows=[];checks=[];params=[];budgets=[];replayrows=[];sourceyears=[]
    def check(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
    for key,tag in selected.items():
        short,arm=key.split('_');out=R/'outputs'/tag;rec=rt.read(out/'model.json');audit=rt.read(out/'audit.json');x=rec['parameters'];eta=np.zeros(4) if arm=='R' else np.asarray(x)[30:][{'U':[0,0,0,0],'G':[0,0,0,1],'S':[0,1,2,3]}[arm]];c=np.exp(eta)
        row=dict(fold=short,model=NAMES[arm],tag=tag,objective=rec['objective'],data=rec['terms']['data'],original_prior=rec['terms'].get('original_prior',rec['terms']['prior']),new_prior=rec['terms'].get('new_prior',0),pg=rec['pg'],numerical=rec['numerical_sufficient'],fertilizer=c[0],manure=c[1],BNF=c[2],deposition=c[3],boundary=bool(np.any(np.isclose(abs(eta),math.log(4),atol=1e-7))))
        sf=out/'full_source_audit.json';row['full_source_pass']=sf.exists() and rt.read(sf)['status']=='PASS';rows.append(row)
        check('independent_'+tag,(R/'reports/independent'/f'{tag}.json').exists() and rt.read(R/'reports/independent'/f'{tag}.json')['status']=='PASS')
        check('full_source_'+tag,row['full_source_pass'])
        for name,value in zip(rec['names'],x):params.append(dict(fold=short,arm=arm,parameter=name,value=value))
        land=pd.read_parquet(out/'monthly_physical_ledger.parquet');net=pd.read_parquet(out/'network_ledger.parquet');year=2023 if short=='F23' else 2024
        for period,a,b in [('reference',1961,2020),('training',2021,year-1),('evaluation',year,year)]:
            g=land[land.year.between(a,b)];n=net[net.year.between(a,b)];last=g[(g.year==b)&(g.month==12)];lastn=n[(n.year==b)&(n.month==12)]
            rr=dict(fold=short,arm=arm,period=period)
            for field in ['source_kg','original_source_kg','uptake_kg','demand_kg','mineral_loss_kg','fast_kg','slow_kg','channel_loss_kg']:rr[field]=float(g[field].sum())
            rr.update(M_end_kg=float(last.M_end_kg.sum()),L_end_kg=float(last.L_end_kg.sum()),terminal_kg=float(n.terminal_kg.sum()),reservoir_end_kg=float(lastn.reservoir_end_kg.sum()),demand_satisfaction=float(g.uptake_kg.sum()/g.demand_kg.sum()) if g.demand_kg.sum()>0 else np.nan);budgets.append(rr)
        if (out/'full_monthly_source_ledger.parquet').exists():
            z=pd.read_parquet(out/'full_monthly_source_ledger.parquet');flux=['original_input_kg','corrected_input_kg','fast_kg','slow_kg','uptake_kg','mineral_loss_kg','channel_loss_kg'];annual=z.groupby(['year','global_reach_id','source'])[flux].sum().reset_index();stock=z[z.month==12][['year','global_reach_id','source','M_kg','L_kg']];annual=annual.merge(stock,on=['year','global_reach_id','source'],validate='one_to_one');annual['fold']=short;annual['arm']=arm;sourceyears.append(annual)
        # Recompute daily RMSE independently from final saved daily predictions.
        if (out/'HF_evaluation_days.parquet').exists():
            daily=pd.read_parquet(out/'HF_evaluation_days.parquet');daily['sq']=(daily.p-daily.y)**2
            ind=daily.groupby('station_key').sq.mean().pow(.5)
            for p in (R/'reports').glob('*/1month/station_metrics.csv'):
                tab=pd.read_csv(p);pair=p.parent.parent.name;hi,lo=pair.split('-');role='R' if arm==lo else 'X' if arm==hi else None
                if role is None:continue
                tab=tab[(tab.fold==short)&(tab.arm==role)&(tab.scale=='HF_day')].set_index('station_key');err=float(abs(ind.reindex(tab.index)-tab.rmse).max()) if len(tab) else 0
                check('independent_daily_rmse_'+key+'_'+pair,err<1e-11,err)
        if arm!='R':
            baseline=selected[short+'_R'];base=pd.read_parquet(R/'outputs'/baseline/'daily_station_mass_water.parquet');joint=pd.read_parquet(out/'daily_station_mass_water.parquet');keys=['station_key','date'];base=base[base.date.dt.year==year][keys+['concentration_mg_l']];joint=joint[joint.date.dt.year==year][keys+['concentration_mg_l']]
            z=base.merge(joint,on=keys,suffixes=('_R','_joint'),validate='one_to_one')
            for mode in ['source_only','process_only']:
                file=out/'replays'/f'{mode}.parquet'
                if not file.exists():continue
                q=pd.read_parquet(file);q=q[q.date.dt.year==year][keys+['concentration_mg_l']].rename(columns={'concentration_mg_l':mode});z=z.merge(q,on=keys,validate='one_to_one')
            if {'source_only','process_only'}.issubset(z.columns):
                z['interaction']=z.concentration_mg_l_joint-z.source_only-z.process_only+z.concentration_mg_l_R
                z['fold']=short;z['arm']=arm;z.to_parquet(out/'replays/four_corner_concentrations.parquet',index=False)
                obs=pd.read_parquet(R/'data/heldout_labels'/f'{year}_days.parquet')[keys+['y']];zz=obs.merge(z,on=keys,validate='one_to_one')
                for station,g in zz.groupby('station_key'):
                    rr=dict(fold=short,arm=arm,station_key=station,n=len(g),mean_interaction=float(g.interaction.mean()),rms_interaction=float(np.sqrt(np.mean(g.interaction**2))))
                    for col in ['concentration_mg_l_R','source_only','process_only','concentration_mg_l_joint']:rr[col+'_rmse']=float(np.sqrt(np.mean((g[col]-g.y)**2)))
                    replayrows.append(rr)
    result=pd.DataFrame(rows);result.to_csv(R/'reports/source_coefficients_and_objectives.csv',index=False);pd.DataFrame(params).to_csv(R/'reports/parameter_roles.csv',index=False);pd.DataFrame(budgets).to_csv(R/'reports/physical_period_summary.csv',index=False);pd.DataFrame(replayrows).to_csv(R/'reports/four_corner_station_diagnostics.csv',index=False)
    if sourceyears:pd.concat(sourceyears,ignore_index=True).to_parquet(R/'reports/annual_reach_source_ledger.parquet',index=False)
    for p in (R/'reports').glob('*/*month/paired_station_changes.csv'):
        g=pd.read_csv(p);core=pd.read_csv(p.parent/'core_paired_table.csv')
        for (fold,scale),a in g.groupby(['fold','scale']):
            a=a[a.nse_eligible_R&a.nse_eligible_X&a.nse_R.notna()&a.nse_X.notna()];row=core[(core.fold==fold)&(core.scale==scale)&(core.group=='ALL')].iloc[0]
            if len(a):check('paired_nse_'+str(p.relative_to(R))+'_'+fold+'_'+scale,abs((a.nse_X-a.nse_R).median()-row.nse_median_paired_change)<1e-11)
    for p in (R/'reports').glob('*/*month/bootstrap_ledger.parquet'):
        a=pd.read_parquet(p);check('bootstrap_copy_ids_'+str(p.relative_to(R)),not a.duplicated(['fold','replicate','copy_id']).any() and a.groupby('fold').replicate.nunique().eq(1000).all())
    frozen=rt.read(R/'data/prediction_freeze.json');changed=[p for p,h in frozen['files'].items() if rt.sha(R/p)!=h];check('prediction_freeze_unchanged',not changed,changed)
    manifest=rt.read(R/'evidence/inherited_manifest.json');changed=[p for p,h in manifest.items() if rt.sha(p)!=h];check('old_experiments_read_only',not changed,changed)
    for p in (R/'scripts').glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    mainpath=R/'reports/G-R/1month';text='# 专家诊断：TN约束下的氮源估计校正\n\n'
    text+='系数是依赖H1水量、D29过程和既定观测支持的来源估计校正；不是实测排放率、到河率或独立污染贡献。全部评价为已见资料的回顾性检验。\n\n## 合法选点与来源倍率\n\n'+md(result)+'\n\n'
    if mainpath.exists():
        core=pd.read_csv(mainpath/'core_paired_table.csv');events=pd.read_csv(mainpath/'event_centered_summary.csv');a=core[core.group=='ALL'];cols=['fold','scale','paired_stations','nse_paired_stations','nse_difference_of_medians','nse_median_paired_change','nse_improved_fraction','rmse_change','bias_change','abs_bias_change'];text+='## 正式主比较：分组来源校正－原D29\n\n'+md(a[cols])+'\n\n'+md(events)+'\n\n'
        for fold in ['F23','F24']:
            c=a[a.fold==fold];e=events[events.fold==fold];chosen=result[(result.fold==fold)&(result.model==NAMES['G'])]
            if not len(c) or not len(e) or not len(chosen):continue
            daily=c[c.scale=='HF_day'].iloc[0];pub=c[c.scale=='PUB_month'].iloc[0];ev=e.iloc[0];qual=bool(chosen.numerical.all() and chosen.full_source_pass.all());joint=bool(daily.rmse_change<0 and ev.amplitude_error_change<0 and ev.peak_error_change<=0 and ev.base_error_change<=0 and ev.centered_mse_X<=ev.centered_mse_R and pub.rmse_change<=0)
            text+=f'- {fold}：日RMSE变化{daily.rmse_change:+.6g} mg/L；月报NSE逐站配对差中位数{pub.nse_median_paired_change:+.6g}；事件振幅/峰值/背景误差变化{ev.amplitude_error_change:+.6g}/{ev.peak_error_change:+.6g}/{ev.base_error_change:+.6g}。数值及完整来源检查通过={qual}；上述多尺度联合方向={joint}。这些方向只形成保守继续建议，轻微反向不等于机制无效。\n'
    text+='\n## 解释与下一步\n\n统一校正识别整体供氮尺度，分组校正为主检验，四源独立仅为探索。所有辅助比较及1/2月块区间在reports各比较目录；单个系数的真实值不能由这些预测区间认证。原参数、来源和两者交互通过四角回放分开呈现，非线性交互不强制归因。\n\n源产品的年份替代、月初日历、遗漏直接来源、158/190站界限制和TN站缺少同期同站界实测流量均仍存在；来源校正可能吸收这些误差。即使改善，也仅支持后续独立空间验证，不修改原始产品或替换主线。\n'
    tasks=rt.read(R/'reports/postprocess_tasks.json');failed=[x for x in tasks if not x['passed']];text+='\n## 缺项与审计\n\n'+json.dumps(dict(missing=frozen['missing'],failed_postprocess=failed),ensure_ascii=False,indent=2)+'\n'
    write('reports/专家诊断报告.md',text)
    write('reports/实际方法与偏离.md','# 实际方法与偏离\n\n- 正式模型入口为source_corrected.py；继承脚本中的STATE_MODULATED等历史分支未进入本轮任务矩阵。倍率范围0.25—4，需求固定，四源展开先验保持嵌套。\n- 全倍率1采用原总质量加分源增量的算术顺序，保证原点精确回归；原分源求和差先以1e-6 kg核验。\n- F24较粗差分跨越原模型库存耗尽边界；保留失败日志、活跃集差异和步长收敛证据。未改公式或放宽容差；要求相邻两步长通过。\n- 系数变化作用于1961开始的全部历史，不是只修正评价期库存；训练标签仅本折，评价统一冻结后读取。\n- 12候选拟合与4基线逻辑路径；输入导数、测试、诊断回放单独计数。新增先验不是文献确定的产品误差分布。\n- 完整分源传播独立于合计核重算，但独立程序审计不是外部专家复核。重采样不重新拟合系数，不能据其区间推断来源参数置信区间。\n')
    with (R/'reports/实际方法与偏离.md').open('a',encoding='utf-8') as f:
        f.write('\n- 原标签核在完整230河段的固定非单位倍率夹具中出现2.15e-6 kg库存求和误差，未通过硬门。完整标签输出改用balanced_tags.py：同一比例分配的最大份额承担浮点余数，合计物理递推、系数和硬门不变。最大调整、逐源平衡及非负性单列于full_source_audit.json；该数值记账不解释为新增物理过程。\n- 模拟事件背景或峰值为零时，不删除事件来改善振幅指标；报告模型比例不可定义次数，该比较不签发振幅改善。峰值和背景绝对误差仍保留原共同支持。\n')
    with (R/'reports/实际方法与偏离.md').open('a',encoding='utf-8') as f:
        f.write('\n- 续算期间源标签试点记账也采用同一浮点余数分配；三个保存点的目标、梯度、预测及合计状态逐位不变，证据为maintenance_v2.json。随后修复Windows路径分隔符重复登记导致的启动身份失败，证据为manifest_recovery_v3.json。保留原优化器、科学入口、失败日志和累计预算，未以重启增加拟合。\n')
    elapsed=(time.time()-rt.read(R/'work/experiment_clock.json')['started'])/3600;passed=all(x['passed'] for x in checks) and not failed and not frozen['missing']
    rt.write(R/'reports/independent_completion_audit.json',dict(status='PASS_WITH_DECLARED_LIMITATIONS' if passed else 'INCOMPLETE_OR_FAILED_CHECKS',checks=checks,postprocess_failures=failed,missing=frozen['missing'],elapsed_hours=elapsed,external_audit=False))
    write('reports/独立完成审计.md','# 独立完成审计\n\n'+f'检查通过={passed}；累计{elapsed:.2f}小时。独立指分开程序/独立汇总，不是外部专家审查。\n\n'+ '\n'.join(f'- {x["name"]}: {x["passed"]}; {x["detail"]}' for x in checks))
    write('README.md','# D29氮源估计校正：20260921_2\n\n状态：有限路径已终止，详见缺项及独立审计。\n\n[专家报告](reports/专家诊断报告.md) · [来源倍率与目标](reports/source_coefficients_and_objectives.csv) · [独立完成审计](reports/独立完成审计.md) · [方法与偏离](reports/实际方法与偏离.md)\n\n分组来源校正为正式主比较；统一及四源校正提供增量诊断。原始数据保持不变；不认证真实排放、不自动替换主线。\n\n运行：conda sparrow，CPU float64，每worker一线程。preflight_source.py与基础设施/隔离测试→seal_source_launch.py→controller.py→finalize_source.py。配置在configs；旧目录只读，发布不在本轮范围。\n')
    files={str(p.relative_to(R)):{'sha256':rt.sha(p),'bytes':p.stat().st_size} for folder in ['scripts','configs','reports'] for p in (R/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    rt.write(R/'delivery_manifest.json',dict(files=files,prediction_freeze='data/prediction_freeze.json',selected=selected));print('REPORT_COMPLETE',passed,flush=True)
    if not passed:raise SystemExit(2)
if __name__=='__main__':main()
