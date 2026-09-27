"""Source-backed scientific figures. No residual-driven station selection."""
import json
import native_runtime as rt
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=rt.RUN;OUT=R/'reports/figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Arial','font.size':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
COL={'U':'#496d89','L3':'#b17b4b','uncorrected':'#8a8a8a','scalar_corrected':'#496d89'}
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=190,bbox_inches='tight');fig.savefig(OUT/(name+'.svg'),bbox_inches='tight');plt.close(fig)
def main():
    # Contract: distinguish constructed-input recoverability from real-data gains.
    rt.write(OUT/'figure_contract.json',dict(backend='Python matplotlib',archetype='quantitative grid',claim='Compare input distortions, scalar recoverability, negative controls, and real paired changes without treating synthetic truth as observed truth',exports=['PNG','SVG editable text'],width_inches=7.2,uncertainty='Synthetic: three fixed seed outcomes, not confidence intervals. Real: bootstrap intervals in source tables; station points retained.',sources=['outputs/synthetic/*/metrics.csv','reports/*/1month/core_paired_table.csv'],integrity='No smoothing of observed labels; no station selected by performance; all units and supports explicit'))
    files=list((R/'outputs/synthetic').glob('*_*/metrics.csv'));s=pd.concat([pd.read_csv(p) for p in files],ignore_index=True);s.to_csv(OUT/'synthetic_source_data.csv',index=False)
    modes=['uniform','month_first','shift14','previous_year','area','mass_067','combined'];labels=['Uniform days','Month start','Shift +14d','Previous year','Area allocation','Mass x0.67','Combined']
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.6),sharey=True)
    for ax,structure in zip(axs,['U','L3']):
        q=s[(s.structure==structure)&(s.fold=='F24')&(s.target=='clean')&(s.support=='observed_dates')]
        for treatment,offset in [('uncorrected',-.12),('scalar_corrected',.12)]:
            for i,mode in enumerate(modes):
                v=q[(q.input_mode==mode)&(q.treatment==treatment)].rmse.to_numpy();ax.scatter(np.full(len(v),i)+offset,v,s=19,color=COL[treatment],alpha=.6,label=treatment.replace('_',' ') if i==0 else None)
        ax.set_xticks(range(len(modes)),labels,rotation=65,ha='right');ax.set_title(structure+' generator: synthetic 2024');ax.set_ylabel('RMSE against fabricated truth (mg/L)');ax.legend(fontsize=7)
    fig.tight_layout();save(fig,'synthetic_recoverability')
    rows=[]
    for structure in ['U','L3']:
        for contrast in ['D-P','A-D','A-P']:
            p=R/f'reports/{structure}_{contrast}/1month/core_paired_table.csv'
            if p.exists():rows.append(pd.read_csv(p).assign(structure=structure,contrast=contrast))
    if rows:
        a=pd.concat(rows);a.to_csv(OUT/'real_core_source_data.csv',index=False)
        fig,axs=plt.subplots(1,2,figsize=(7.2,3.4))
        for ax,fold in zip(axs,['F23','F24']):
            q=a[(a.fold==fold)&(a.scale=='HF_day')&(a.group=='ALL')]
            for structure,offset in [('U',-.1),('L3',.1)]:
                for i,contrast in enumerate(['D-P','A-D','A-P']):
                    row=q[(q.structure==structure)&(q.contrast==contrast)]
                    if len(row):ax.scatter(i+offset,row.rmse_change.iloc[0],color=COL[structure],s=40,label=structure if i==0 else None)
            ax.axhline(0,color='.5',lw=.7);ax.set_xticks(range(3),['D minus P','A minus D','A minus P']);ax.set_ylabel('Change in station-mean daily RMSE (mg/L)');ax.set_title(fold+' real held-out observations');ax.legend()
        fig.tight_layout();save(fig,'real_daily_comparison')
    pulses=[]
    for structure in ['U','L3']:
        p=R/f'outputs/synthetic/pulses/{structure}_summary.csv'
        if p.exists():pulses.append(pd.read_csv(p))
    if pulses:
        p=pd.concat(pulses);p.to_csv(OUT/'pulse_source_data.csv',index=False)
        fig,axs=plt.subplots(1,2,figsize=(7.2,3.4))
        for structure in ['U','L3']:
            q=p[(p.structure==structure)&(p.spread_days==1)];v=q.groupby('horizon_days')[['delta_M_kg','delta_L_kg','station_rms_delta_mg_l']].mean()
            axs[0].plot(v.index,(v.delta_M_kg+v.delta_L_kg)/230,'o-',color=COL[structure],label=structure)
            axs[1].plot(v.index,v.station_rms_delta_mg_l,'o-',color=COL[structure],label=structure)
        for ax in axs:ax.set_xscale('log');ax.set_xlabel('Days after fixed-date unit input');ax.legend()
        axs[0].set_ylabel('Incremental land stock / injected N');axs[1].set_ylabel('Station RMS over first n days (mg/L)');fig.tight_layout();save(fig,'signal_propagation')
    rt.write(OUT/'render_manifest.json',dict(status='RENDERED_PENDING_VISUAL_QA',files={p.name:rt.sha(p) for p in OUT.glob('*.png')}))
if __name__=='__main__':main()
