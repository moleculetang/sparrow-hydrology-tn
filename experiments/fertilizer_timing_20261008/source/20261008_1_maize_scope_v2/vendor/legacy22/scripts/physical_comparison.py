"""Compare archived physical budgets; no new forward runs or label access."""
import numpy as np,pandas as pd
import native_runtime as rt
R=rt.RUN
def main(selected):
    rows=[]
    flow=['fast_kg','slow_kg','uptake_kg','demand_kg','mineral_loss_kg','channel_loss_kg','source_kg']
    for key,tag in selected.items():
        fold,arm=key.split('_');out=R/'outputs'/tag
        land=pd.read_parquet(out/'monthly_physical_ledger.parquet');network=pd.read_parquet(out/'network_ledger.parquet')
        for year,g in land.groupby('year'):
            end=g[g.month==12];n=network[network.year==year];ne=n[n.month==12]
            row=dict(fold=fold,arm=arm,year=int(year),**{c:float(g[c].sum()) for c in flow})
            row.update(M_end_kg=float(end.M_end_kg.sum()),L_end_kg=float(end.L_end_kg.sum()),terminal_kg=float(n.terminal_kg.sum()),reservoir_end_kg=float(ne.reservoir_end_kg.sum()))
            row['uptake_demand_ratio']=row['uptake_kg']/row['demand_kg'] if row['demand_kg']>0 else np.nan
            rows.append(row)
    frame=pd.DataFrame(rows);a=frame[frame.arm.eq('R')].drop(columns='arm');b=frame[frame.arm.eq('X')].drop(columns='arm')
    pair=a.merge(b,on=['fold','year'],suffixes=('_R','_X'),validate='one_to_one')
    quantities=[k for k in a.columns if k not in ['fold','year']]
    for k in quantities:pair[k+'_change']=pair[k+'_X']-pair[k+'_R']
    pair.to_csv(R/'reports/physical_year_comparison.csv',index=False)
    summaries=[]
    for fold,year in [('F23',2023),('F24',2024)]:
        for period,years in [('reference_1961_2020',list(range(1961,2021))),('training',list(range(2021,year))),('evaluation',[year])]:
            z=pair[pair.fold.eq(fold)&pair.year.isin(years)]
            if z.empty:continue
            row=dict(fold=fold,period=period,years=len(z))
            for k in quantities:
                for arm in ['R','X']:
                    # Fluxes sum over the period; pools use its final year.
                    row[k+'_'+arm]=float(z[k+'_'+arm].iloc[-1]) if '_end_' in k else (float(z['uptake_kg_'+arm].sum()/z['demand_kg_'+arm].sum()) if k=='uptake_demand_ratio' else float(z[k+'_'+arm].sum()))
                row[k+'_change']=row[k+'_X']-row[k+'_R']
            summaries.append(row)
    pd.DataFrame(summaries).to_csv(R/'reports/physical_period_comparison.csv',index=False)
    return summaries
