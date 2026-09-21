import os
from pathlib import Path
R=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(R/'work/matplotlib')
from runtime import *
contract=dict(conclusion='Wetness modulation modestly improves event amplitude error but misses the registered structural threshold.',evidence=['Fixed gamma response on identical events','Paired 2024 station errors','Peak and background changes, not just a ratio'],archetype='quantitative grid',backend='Python/matplotlib',dimensions_inches=[9,3],formats=['png','svg','pdf'],text='editable',statistics='Station equal weight; 2024: 15 stations / 57 events; development: 14 stations / 70 events; descriptive, no blind validation',review_risks=['Do not select gamma by 2024 score','Do not use physically blocked level controls as positive evidence','Concentration amplitudes are on matched observed days'])
put(R/'reports/figure_contract.json',contract)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.right':False,'axes.spines.top':False,'legend.frameon':False})
comp=read(R/'reports/comparisons.json');gamma=[0,.5,1,2,4];keys=['G0','G05','G1','G2','G4']
fig,axes=plt.subplots(1,3,figsize=(9,3),layout='constrained',gridspec_kw={'width_ratios':[1.2,1,1]})
source=[]
for period,color in [('2021-2023','#8a8a8a'),('2024','#0072B2')]:
 vals=[0]+[100*comp[f'H1_{k} minus H1_G0'][period]['relative_reduction'] for k in keys[1:]]
 axes[0].plot(gamma,vals,'o-',color=color,label=period)
 source.extend(dict(gamma=g,period=period,reduction_pct=v) for g,v in zip(gamma,vals))
axes[0].axhline(25,color='#b35a36',linestyle='--',linewidth=1);axes[0].text(4,25.5,'Registered threshold',ha='right',fontsize=7,color='#b35a36')
axes[0].set(xlabel='Wetness exponent, gamma',ylabel='Event-error reduction (%)',ylim=(-1,30),title='a  Fixed response range');axes[0].legend(loc='upper left',fontsize=7)
a=pd.read_parquet(R/'outputs/H1_G0/event_evaluation.parquet');b=pd.read_parquet(R/'outputs/H1_G1/event_evaluation.parquet')
a=a[a.period.eq('2024')].groupby('station_key').log_amplitude_error.median();b=b[b.period.eq('2024')].groupby('station_key').log_amplitude_error.median();s=pd.concat([a,b],axis=1,keys=['constant','wetness']).dropna()
axes[1].scatter(s.constant,s.wetness,color='#0072B2',s=20);v=max(s.max())*1.05;axes[1].plot([0,v],[0,v],color='#888888',linewidth=.8)
axes[1].set(xlabel='Constant-transfer error',ylabel='Wetness-transfer error',title='b  2024 station pairs',xlim=(0,v),ylim=(0,v));axes[1].text(.05,.95,'10/15 stations improve',transform=axes[1].transAxes,va='top',fontsize=7)
z=pd.read_csv(R/'reports/paired_base_peak_decomposition.csv');z=z[z.period.eq('2024')]
axes[2].scatter(z.minus_log_background_change,z.log_peak_change,color='#0072B2',s=10,alpha=.65);axes[2].axhline(0,color='#999999',linewidth=.7);axes[2].axvline(0,color='#999999',linewidth=.7)
axes[2].set(xlabel='-log(background candidate / baseline)',ylabel='log(peak candidate / baseline)',title='c  Ratio-change components')
out=R/'reports/figures';out.mkdir(exist_ok=True)
for ext in ('png','svg','pdf'):fig.savefig(out/f'structure_evidence.{ext}',dpi=240)
pd.DataFrame(source).to_csv(out/'panel_a_source.csv',index=False);s.to_csv(out/'panel_b_source.csv');z.to_csv(out/'panel_c_source.csv',index=False)
plt.close(fig)
print('FIGURE_COMPLETE')
