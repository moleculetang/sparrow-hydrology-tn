"""Python-only quantitative figures; source tables and fixed supports retained."""
import native_runtime as rt
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages
R=rt.RUN

def main():
    folder=R/'figures';folder.mkdir(exist_ok=True)
    rt.write(folder/'figure_contract.json',dict(question='Does the registered L3 model improve typical stations and preserve cross-period/spatial performance?',archetype='quantitative grid',backend='Python matplotlib',size_mm=[183,160],primary='paired per-station RMSE changes',secondary='monthly NSE and event errors',source='reports/*/1month paired tables',statistics='paired stations, no significance test; 1000 frozen-model paired-month bootstrap intervals in source tables',claim='Results determine the conclusion; no prospective benefit assumed',exports=['SVG','PDF','PNG'],no_clipping=True))
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Microsoft YaHei','DejaVu Sans'],'font.size':7,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False})
    fig,axes=plt.subplots(2,2,figsize=(183/25.4,160/25.4),layout='constrained');allrows=[];event_labels=set()
    cases=[('L3-U','F23'),('L3-U','F24'),('S56_L3-U','S56'),('S113_L3-U','S113'),('S191_L3-U','S191')]
    for index,(comparison,fold) in enumerate(cases):
        path=R/'reports'/comparison/'1month/paired_station_changes.csv'
        if not path.exists():continue
        df=pd.read_csv(path);df=df[df.fold.eq(fold)];allrows.append(df.assign(comparison=comparison))
        for ax,scale,metric,title in [(axes[0,0],'HF_day','rmse','a  Daily concentration error'),(axes[0,1],'PUB_month','nse','b  Monthly NSE')]:
            g=df[df.scale.eq(scale)];v=g['delta_'+metric].dropna().to_numpy();offset=np.linspace(-.13,.13,len(v)) if len(v) else []
            ax.scatter(index+np.asarray(offset),v,s=10,color='#5686a5',alpha=.6,linewidths=0)
            if len(v):ax.plot([index-.18,index+.18],[np.median(v)]*2,color='#222222',lw=1.5)
            ax.set_title(title,loc='left');ax.axhline(0,color='#888888',lw=.6);ax.set_xticks(range(5),[x[1] for x in cases],rotation=30);ax.set_ylabel('Candidate minus baseline'+(' (mg/L)' if metric=='rmse' else ''))
        path=R/'reports'/comparison/'1month/event_centered_summary.csv'
        if path.exists():
            g=pd.read_csv(path);g=g[g.fold.eq(fold)]
            for row in g.to_dict('records'):
                for j,k in enumerate(['amplitude_error','peak_error','base_error']):
                    key=k+'_change'
                    if key in row and row.get(k+'_R',0)>0:
                        axes[1,0].scatter(index+(j-1)*.16,100*row[key]/row[k+'_R'],s=24,marker=['o','s','^'][j],color=['#5686a5','#b58d67','#8c7ca9'][j],label=k.replace('_error','') if k not in event_labels else None);event_labels.add(k)
    axes[1,0].set_title('c  Relative event errors',loc='left');axes[1,0].axhline(0,color='#888888',lw=.6);axes[1,0].set_xticks(range(5),[x[1] for x in cases],rotation=30);axes[1,0].set_ylabel('Change / baseline error (%)');axes[1,0].legend(fontsize=6)
    p=R/'reports/lowrank_paired.csv'
    if p.exists():
        g=pd.read_csv(p);g=g[g.scale.eq('daily')&g.metric.eq('logrmse')&g.target.eq('anomaly')]
        for fold,color in [('F23','#5686a5'),('F24','#b58d67')]:
            v=g[g.fold.eq(fold)].sort_values('model');axes[1,1].plot(v.model,v.median_paired_change,'o-',label=fold,color=color,ms=4)
    axes[1,1].set_title('d  Input-only low-rank forecast',loc='left');axes[1,1].axhline(0,color='#888888',lw=.6);axes[1,1].set_ylabel('Log-RMSE change vs unrestricted ridge');axes[1,1].legend(fontsize=6)
    for ext in ['svg','pdf','png']:fig.savefig(folder/f'paired_overview.{ext}',dpi=220)
    plt.close(fig)
    if allrows:pd.concat(allrows).to_csv(folder/'paired_overview_source.csv',index=False)
    counts=[]
    for frame in allrows:
        for (fold,scale),g in frame.groupby(['fold','scale']):counts.append(f'{fold} {scale}: {len(g)} paired stations; NSE eligible {(g.nse_eligible_R & g.nse_eligible_X).sum()}')
    (folder/'figure_legends.md').write_text('# Figure 1 | Regional response under fixed input and physical constraints\n\n(a) Each point is one paired station; the black line is the median paired RMSE change. Negative is better. (b) Paired monthly NSE differences on the qualified station set; positive is better. No stations or extreme values are clipped. (c) Station-median, then station-equal event errors expressed as percentage change relative to baseline: amplitude is log-ratio error; peak and background are mg/L errors before normalization. A zero baseline error has no defined relative percentage and is omitted only from this relative figure, not from the tables. (d) Median paired log-RMSE change for input-only low-rank forecasts versus the same-input unrestricted ridge, with fixed lambda=1. Contemporary TN-assisted reconstructions are excluded.\n\nNo inferential significance marks are used. Frozen-model 1000-replicate paired month intervals, and two-month-block sensitivity, are in the corresponding bootstrap tables. F23 trains on 2021–2022; F24 trains on 2021–2023; S56/S113/S191 exclude their registered TN closures and buffers and evaluate 2024.\n\n'+ '\n'.join(counts)+'\n\nEvent PDF: every eligible temporal event is shown, with original complete input/prediction time axis and TN only on observed dates. No event-specific time shift or peak-window optimization.\n',encoding='utf-8')
    curves=R/'reports/L3-U/1month/event_curves.parquet'
    pages=0
    if curves.exists():
        df=pd.read_parquet(curves)
        with PdfPages(folder/'all_temporal_events.pdf') as pdf:
            for (fold,station,event),g in df.groupby(['fold','station_key','event_rank'],sort=True):
                fig,ax=plt.subplots(figsize=(7.2,3.2),layout='constrained')
                for arm,label,color in [('R','Unified source D29','#777777'),('X','3-mode regional D29','#5686a5')]:
                    q=g[g.arm.eq(arm)].sort_values('date');ax.plot(q.date,q.p,label=label,color=color,lw=1)
                obs=g[['date','y']].drop_duplicates().dropna();ax.scatter(obs.date,obs.y,label='Observed daily TN',color='#222222',s=15,zorder=3);ax.set(ylabel='TN (mg/L)');ax.set_title(f'{fold} | {station} | event {event}',fontproperties=FontProperties(fname='C:/Windows/Fonts/msyh.ttc',size=8));ax.legend(fontsize=7);fig.autofmt_xdate();pdf.savefig(fig);plt.close(fig);pages+=1
    rt.write(folder/'figure_qa.json',dict(status='GENERATED_PENDING_VISUAL_REVIEW',event_pages=pages,all_events_included=True,editable_vector_text=True,backend='Python',source_tables_saved=True))
if __name__=='__main__':main()
